#!/usr/bin/env python3
"""
predict_stock.py — 株価方向 2値分類予測（全銘柄版）
====================================================
annual_select.py の出力（annual_result.xlsx の AllCandidates シート）を
ファンダメンタルズ特徴量として利用し、yfinance の時系列から
テクニカル特徴量も組み合わせて HORIZON 日後の騰落を分類予測する。

処理フロー:
  1. データ読み込み  : annual_result.xlsx[AllCandidates] (1000銘柄超)
  2. 株価ダウンロード: yf.download() バッチ処理 + Parquet キャッシュ
  3. 特徴量生成     : テクニカル指標 + ファンダメンタルズ指標
  4. ラベル生成     : HORIZON 日後に THRESHOLD 以上上昇 → 1, それ以外 → 0
  5. モデル訓練     : Logistic Regression (基準) + Random Forest
  6. 評価・可視化   : 混同行列 / 分類レポート / ROC 曲線 / 特徴量重要度
  7. 最新シグナル   : 全銘柄の上昇確率ランキング CSV
"""

import json
import warnings
from pathlib import Path

# EDINET シグナル（オプション）
try:
    import edinet_signals as edinet
    _EDINET_AVAILABLE = True
except ImportError:
    _EDINET_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import chart_style  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
HORIZON     = 20    # 何営業日後の騰落を予測するか
THRESHOLD   = 0.03  # 上昇と判定する変化率（3% 超 → ラベル 1）
LOOKBACK    = "2y"  # yfinance で取得する期間
TRAIN_RATIO = 0.70  # 訓練データの割合（時系列分割）
BATCH_SIZE  = 50    # yf.download のバッチサイズ（銘柄数）
MIN_ROWS    = 60    # 最低必要サンプル行数
USE_EDINET  = True  # EDINET 大量保有報告書シグナルを特徴量に追加するか

_ROOT             = Path(__file__).parent.parent  # dividend-system/
FUNDAMENTALS_FILE = _ROOT / "output" / "annual_result.xlsx"
OUTPUT_DIR        = _ROOT / "output"
CACHE_DIR         = OUTPUT_DIR / "price_cache"  # Parquet キャッシュ保存先
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ファンダメンタルズ指標として使う列
FUND_COLS = [
    "利回り%(3年平均)",
    "配当性向%(複数年平均)",
    "ROE%(複数年平均)",
    "DOE%",
    "PBR",
    "理論利回り%",
    "実質PBR倍率",
    "売上成長率%",
    "負債比率",
    "利回り点",
    "安定性点",
    "配当性向点",
    "ROE点",
    "成長点",
    "財務点",
    "総合点",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. データ読み込み
# ─────────────────────────────────────────────────────────────────────────────

def load_fundamentals() -> pd.DataFrame:
    """
    annual_result.xlsx の AllCandidates シートを優先して読み込む。
    なければ Final7 シートにフォールバック。
    """
    if not FUNDAMENTALS_FILE.exists():
        raise FileNotFoundError(
            f"{FUNDAMENTALS_FILE} が見つかりません。"
            "先に annual_select.py を実行してください。"
        )

    xl     = pd.ExcelFile(FUNDAMENTALS_FILE)
    sheet  = "AllCandidates" if "AllCandidates" in xl.sheet_names else "Final7"
    df     = pd.read_excel(FUNDAMENTALS_FILE, sheet_name=sheet,
                           dtype={"ティッカー": str})

    keep   = ["ティッカー", "会社名", "業種"] + \
             [c for c in FUND_COLS if c in df.columns]
    df     = df[[c for c in keep if c in df.columns]].dropna(subset=["ティッカー"])
    df["ティッカー"] = df["ティッカー"].str.strip()

    print(f"[データ] ファンダメンタルズ: {len(df)} 銘柄 (シート: {sheet})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. 株価バッチダウンロード（Parquet キャッシュ付き）
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('.', '_')}.parquet"


def download_prices_batch(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    複数銘柄を BATCH_SIZE ずつ yf.download() で一括取得し、
    Parquet キャッシュに保存する。キャッシュ済み銘柄はスキップ。
    """
    result:       dict[str, pd.DataFrame] = {}
    to_download:  list[str]               = []

    # キャッシュ確認
    for t in tickers:
        cp = _cache_path(t)
        if cp.exists():
            try:
                result[t] = pd.read_parquet(cp)
            except Exception:
                to_download.append(t)
        else:
            to_download.append(t)

    cached = len(result)
    if cached:
        print(f"  キャッシュ使用: {cached} 銘柄")

    if not to_download:
        return result

    print(f"  ダウンロード対象: {len(to_download)} 銘柄 "
          f"(バッチサイズ={BATCH_SIZE})")

    for i in range(0, len(to_download), BATCH_SIZE):
        batch = to_download[i : i + BATCH_SIZE]
        end   = min(i + BATCH_SIZE, len(to_download))
        print(f"  [{i+1:4d}–{end:4d}/{len(to_download)}] "
              f"{batch[0]} … {batch[-1]}", end="  ", flush=True)

        try:
            raw = yf.download(
                batch,
                period=LOOKBACK,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                print("空データ")
                continue

            # 複数銘柄は MultiIndex、1銘柄は単純 Index
            if isinstance(raw.columns, pd.MultiIndex):
                close_raw = raw["Close"]
            else:
                # 1銘柄のとき columns = ["Close", "High", ...]
                close_raw = raw[["Close"]].copy()
                close_raw.columns = [batch[0]]

            ok = 0
            for t in batch:
                if t not in close_raw.columns:
                    continue
                s = close_raw[t].dropna()
                if s.empty:
                    continue
                df       = s.to_frame("close")
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_parquet(_cache_path(t))
                result[t] = df
                ok += 1

            print(f"OK {ok}/{len(batch)}")

        except Exception as e:
            print(f"エラー: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. 特徴量エンジニアリング
# ─────────────────────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """終値 (close 列) からテクニカル指標を追加する。"""
    p = df["close"]

    ma5  = p.rolling(5).mean()
    ma20 = p.rolling(20).mean()
    ma60 = p.rolling(60).mean()

    # 移動平均比率
    df["r_5_20"]  = ma5 / ma20 - 1
    df["r_5_60"]  = ma5 / ma60 - 1
    df["r_20_60"] = ma20 / ma60 - 1

    # モメンタム
    for n in [5, 10, 20, 60]:
        df[f"mom_{n}"] = p.pct_change(n)

    # ボリンジャーバンド位置
    roll20       = p.rolling(20)
    df["bb_pos"] = (p - roll20.mean()) / (roll20.std() + 1e-9)

    # RSI (14日)
    delta        = p.diff()
    gain         = delta.clip(lower=0).rolling(14).mean()
    loss         = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"]  = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD ヒストグラム
    ema12        = p.ewm(span=12, adjust=False).mean()
    ema26        = p.ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]   = macd_line - signal_line

    # ボラティリティ (14日ローリング標準偏差)
    df["vol14"]  = p.pct_change().rolling(14).std()

    return df


def build_features(
    price_df:   pd.DataFrame,
    fund_row:   pd.Series,
    ticker:     str,
    signals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """テクニカル + ファンダメンタルズ（+ EDINET）特徴量を結合してラベルを付与する。"""
    df   = price_df.copy()
    df   = add_technical_features(df)

    # 日付列（プロット用）
    df["date"] = df.index

    # ラベル: HORIZON 日後の変化率が THRESHOLD を超えたら 1
    future_ret   = df["close"].pct_change(HORIZON).shift(-HORIZON)
    df["label"]  = (future_ret > THRESHOLD).astype(int)

    # ファンダメンタルズを定数列として付与
    for col in FUND_COLS:
        if col in fund_row.index:
            df[f"fund_{col}"] = fund_row[col]

    # EDINET 大量保有報告書シグナル
    if signals_df is not None and not signals_df.empty:
        df = edinet.add_large_holder_features(df, ticker, signals_df)

    df["ticker"] = ticker

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. データセット構築
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    fund_df:    pd.DataFrame,
    price_map:  dict[str, pd.DataFrame],
    signals_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """全銘柄のデータを統合したデータセットと特徴量列リストを返す。"""
    fund_indexed = fund_df.set_index("ティッカー")
    all_frames   = []
    skip         = 0

    for ticker, price_df in price_map.items():
        if ticker not in fund_indexed.index:
            skip += 1
            continue
        feat_df = build_features(price_df, fund_indexed.loc[ticker], ticker, signals_df)
        if len(feat_df) < MIN_ROWS:
            skip += 1
            continue
        all_frames.append(feat_df)

    if not all_frames:
        raise RuntimeError("有効なデータが1銘柄も取得できませんでした。")

    print(f"  使用: {len(all_frames)} 銘柄 / スキップ: {skip} 銘柄")

    dataset = pd.concat(all_frames, ignore_index=True)

    # 特徴量列（非数値 / ラベル / 識別子 / 日付 を除く）
    exclude   = {"close", "label", "ticker", "date"}
    feat_cols = [
        c for c in dataset.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(dataset[c])
    ]

    pos_rate = dataset["label"].mean()
    print(f"  データセット: {len(dataset):,} 行 / "
          f"特徴量: {len(feat_cols)} 個 / 上昇率: {pos_rate:.1%}")
    return dataset, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 5. モデル訓練・評価
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000, class_weight="balanced",
        )),
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}


def train_and_evaluate(
    dataset:   pd.DataFrame,
    feat_cols: list[str],
) -> dict:
    """時系列順で TRAIN_RATIO 分割してモデルを訓練し評価結果を返す。"""
    dataset = dataset.sort_index()
    split   = int(len(dataset) * TRAIN_RATIO)

    X_train = dataset.iloc[:split][feat_cols].values
    y_train = dataset.iloc[:split]["label"].values
    X_test  = dataset.iloc[split:][feat_cols].values
    y_test  = dataset.iloc[split:]["label"].values

    print(f"\n[分割] 訓練: {len(X_train):,} 行 / テスト: {len(X_test):,} 行")

    results = {}
    for name, model in MODELS.items():
        print(f"\n{'─'*55}")
        print(f"  モデル: {name}")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = roc_auc_score(y_test, y_proba)
        report  = classification_report(
            y_test, y_pred,
            target_names=["下落(0)", "上昇(1)"],
            output_dict=True,
        )

        print(f"  ROC-AUC : {auc:.4f}")
        print(classification_report(
            y_test, y_pred, target_names=["下落(0)", "上昇(1)"]
        ))

        results[name] = {
            "model":     model,
            "y_test":    y_test,
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "auc":       auc,
            "report":    report,
            "feat_cols": feat_cols,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. 可視化
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(results: dict) -> None:
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            res["y_test"], res["y_pred"],
            display_labels=["下落(0)", "上昇(1)"],
            ax=ax, colorbar=False,
        )
        ax.set_title(f"{name}\nROC-AUC={res['auc']:.3f}")

    fig.tight_layout()
    out = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n[出力] 混同行列     → {out}")


def plot_roc_curves(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ランダム")

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={res['auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC 曲線  (予測: {HORIZON}日後に{THRESHOLD*100:.0f}%超上昇 / "
        f"全銘柄)"
    )
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / "roc_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] ROC 曲線     → {out}")


def plot_feature_importance(results: dict, top_n: int = 25) -> None:
    rf_res = results.get("RandomForest")
    if rf_res is None:
        return

    clf         = rf_res["model"].named_steps["clf"]
    feat_cols   = rf_res["feat_cols"]
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    labels      = [feat_cols[i] for i in indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 0.4 * top_n + 1.5))
    ax.barh(range(len(labels)), values[::-1], align="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Random Forest — 特徴量重要度 Top {top_n} (全銘柄)")
    fig.tight_layout()
    out = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 特徴量重要度 → {out}")


def plot_price_with_signals(
    fund_df:   pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    results:   dict,
    ticker:    str,
) -> None:
    """指定銘柄の株価チャートにテスト期間の予測シグナルを重ねる。"""
    rf_res = results.get("RandomForest")
    if rf_res is None or ticker not in price_map:
        return

    fund_indexed = fund_df.set_index("ティッカー")
    if ticker not in fund_indexed.index:
        return

    feat_df = build_features(
        price_map[ticker], fund_indexed.loc[ticker], ticker
    )
    if len(feat_df) < MIN_ROWS:
        return

    test_part = feat_df.iloc[int(len(feat_df) * TRAIN_RATIO):]
    feat_cols = rf_res["feat_cols"]

    avail  = [c for c in feat_cols if c in test_part.columns]
    if len(avail) != len(feat_cols):
        return

    X      = test_part[avail].values
    y_pred = rf_res["model"].predict(X)
    proba  = rf_res["model"].predict_proba(X)[:, 1]
    dates  = pd.to_datetime(test_part["date"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(13, 7),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax1.plot(dates, test_part["close"].values, lw=1,
             color="steelblue", label="終値")
    up   = y_pred == 1
    down = y_pred == 0
    ax1.scatter(dates[up],   test_part["close"].values[up],
                marker="^", color="red",  s=35, zorder=5, label="上昇予測")
    ax1.scatter(dates[down], test_part["close"].values[down],
                marker="v", color="blue", s=15, zorder=5,
                alpha=0.4, label="下落予測")

    company = fund_indexed.loc[ticker].get("会社名", ticker) \
        if "会社名" in fund_indexed.columns else ticker
    ax1.set_title(f"{ticker} {company} — テスト期間 予測シグナル")
    ax1.set_ylabel("株価 (円)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, proba, 0.5,
                     where=(proba >= 0.5), alpha=0.5, color="red")
    ax2.fill_between(dates, proba, 0.5,
                     where=(proba < 0.5),  alpha=0.5, color="blue")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("上昇確率")
    ax2.set_xlabel("日付")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / f"signals_{ticker.replace('.', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] シグナル図   → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. 最新シグナル（全銘柄ランキング）
# ─────────────────────────────────────────────────────────────────────────────

def predict_latest(
    fund_df:   pd.DataFrame,
    price_map: dict[str, pd.DataFrame],
    results:   dict,
) -> None:
    """RF モデルで全銘柄の最新データを予測し、上昇確率降順のランキングを CSV 出力する。"""
    rf_res = results.get("RandomForest")
    if rf_res is None:
        return

    model        = rf_res["model"]
    feat_cols    = rf_res["feat_cols"]
    fund_indexed = fund_df.set_index("ティッカー")

    rows = []
    for ticker, price_df in price_map.items():
        if ticker not in fund_indexed.index:
            continue

        fund_row = fund_indexed.loc[ticker]
        feat_df  = build_features(price_df, fund_row, ticker)
        if feat_df.empty:
            continue

        avail = [c for c in feat_cols if c in feat_df.columns]
        if len(avail) != len(feat_cols):
            continue

        last  = feat_df.iloc[[-1]]
        proba = model.predict_proba(last[avail].values)[0][1]
        pred  = "↑上昇" if proba >= 0.5 else "↓下落"
        price = last["close"].values[0]

        company = fund_row.get("会社名", ticker) \
            if "会社名" in fund_row.index else ticker
        sector  = fund_row.get("業種", "") \
            if "業種" in fund_row.index else ""
        score   = fund_row.get("総合点", float("nan")) \
            if "総合点" in fund_row.index else float("nan")

        rows.append({
            "ティッカー": ticker,
            "会社名":     company,
            "業種":       sector,
            "予測":       pred,
            "上昇確率":   round(proba, 4),
            "現在株価":   round(price, 0),
            "総合点":     score,
        })

    if not rows:
        print("[警告] 最新シグナルを生成できませんでした。")
        return

    df_sig = pd.DataFrame(rows).sort_values("上昇確率", ascending=False)
    df_sig["ランク"] = range(1, len(df_sig) + 1)
    df_sig = df_sig[["ランク", "ティッカー", "会社名", "業種",
                      "予測", "上昇確率", "現在株価", "総合点"]]

    # コンソール表示（上位20件）
    print(f"\n{'='*75}")
    print(f"【最新シグナル TOP20】{HORIZON}日後の騰落予測 (Random Forest / 全銘柄)")
    print(f"{'='*75}")
    header = f"{'ランク':>4}  {'ティッカー':10} {'会社名':28} {'予測':6} {'上昇確率':8} {'株価':>8} {'総合点':>6}"
    print(header)
    print("─" * 75)
    for _, row in df_sig.head(20).iterrows():
        print(
            f"{int(row['ランク']):4d}  "
            f"{row['ティッカー']:10} "
            f"{str(row['会社名'])[:26]:28} "
            f"{row['予測']:6} "
            f"{row['上昇確率']:7.1%}  "
            f"{row['現在株価']:>8,.0f}円  "
            f"{row['総合点']:>6.1f}点"
        )

    out = OUTPUT_DIR / "latest_signals.csv"
    df_sig.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[出力] 最新シグナル → {out}  ({len(df_sig)} 銘柄)")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("株価方向 2値分類予測プログラム（全銘柄版）")
    print(f"  予測ホライズン : {HORIZON} 営業日後")
    print(f"  上昇判定閾値  : {THRESHOLD*100:.0f}% 超")
    print(f"  学習期間      : 直近 {LOOKBACK}")
    print("=" * 60)

    # 1. ファンダメンタルズ読み込み
    fund_df = load_fundamentals()
    tickers = fund_df["ティッカー"].tolist()

    # 2. 株価バッチダウンロード
    print(f"\n[株価ダウンロード / キャッシュ確認中... {len(tickers)} 銘柄]")
    price_map = download_prices_batch(tickers)
    print(f"  取得成功: {len(price_map)} 銘柄")

    # 2b. EDINET 大量保有報告書シグナル取得（オプション）
    signals_df = None
    if USE_EDINET and _EDINET_AVAILABLE:
        print("\n[EDINET 大量保有報告書シグナル取得中...]")
        try:
            signals_df = edinet.load_or_fetch_signals(tickers)
            print(f"  取得件数: {len(signals_df)} 件")
        except Exception as e:
            print(f"  [警告] EDINET 取得失敗（スキップ）: {e}")
            signals_df = None

    # 3. データセット構築
    print("\n[特徴量構築中...]")
    dataset, feat_cols = build_dataset(fund_df, price_map, signals_df)

    # 4. 訓練・評価
    results = train_and_evaluate(dataset, feat_cols)

    # 5. 可視化
    print("\n[グラフ生成中...]")
    plot_confusion_matrix(results)
    plot_roc_curves(results)
    plot_feature_importance(results)

    # Final7 の先頭銘柄でシグナル図を生成
    try:
        xl      = pd.ExcelFile(FUNDAMENTALS_FILE)
        f7      = pd.read_excel(FUNDAMENTALS_FILE, sheet_name="Final7",
                                dtype={"ティッカー": str})
        f7_tick = f7["ティッカー"].str.strip().iloc[0]
        plot_price_with_signals(fund_df, price_map, results, f7_tick)
    except Exception:
        pass

    # 6. 最新シグナル（全銘柄）
    print("\n[最新シグナル生成中...]")
    predict_latest(fund_df, price_map, results)

    # 7. サマリー
    print(f"\n{'='*60}")
    print("【精度サマリー】")
    for name, res in results.items():
        rep = res["report"]
        acc = rep["accuracy"]
        f1  = rep.get("上昇(1)", {}).get("f1-score", float("nan"))
        print(f"  {name:25} Accuracy={acc:.3f}  "
              f"F1(上昇)={f1:.3f}  AUC={res['auc']:.3f}")
    print("=" * 60)

    print("\n出力ファイル:")
    for f in sorted(OUTPUT_DIR.glob("*.png")) + sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
