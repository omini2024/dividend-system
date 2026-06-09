#!/usr/bin/env python3
"""
predict_minmax.py — 今後 HORIZON 営業日以内の株価 Min / Max 回帰予測
======================================================================
現在の分類モデルと同じ特徴量を使い、

  y_max = HORIZON日以内の最高値変化率(%) ... 上値余地
  y_min = HORIZON日以内の最低値変化率(%) ... 下値リスク

を RandomForestRegressor で予測する。

出力グラフ:
  minmax_accuracy.png    ... 予測 vs 実際 の散布図・誤差分布
  minmax_chart.png       ... 3076.T / 5970.T の株価チャートに予測レンジを重ねる
  minmax_latest.png      ... 最新予測の見通しサマリー
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import chart_style  # noqa: F401
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import predict_stock as ps

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
HORIZON      = 5        # 予測ホライズン（営業日）
TRAIN_RATIO  = 0.70
MIN_ROWS     = 80
TARGETS      = ["3076.T", "5970.T"]
NAMES        = {"3076.T": "AI HOLDINGS", "5970.T": "G-TEKT"}
COLORS       = {"3076.T": "#e63946",     "5970.T": "#457b9d"}
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── テクニカル特徴量列（predict_stock と同一）────────────────────────────────
TECH_COLS = [
    "r_5_20", "r_5_60", "r_20_60",
    "mom_5", "mom_10", "mom_20", "mom_60",
    "bb_pos", "rsi14", "macd", "vol14",
]


# ─────────────────────────────────────────────────────────────────────────────
# ラベル生成: y_max / y_min (%)
# ─────────────────────────────────────────────────────────────────────────────

def make_minmax_labels(price_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    各時点から horizon 日後まで の最高値・最低値変化率 (%) を計算する。
    返り値: index=price_df.index, 列=[y_max, y_min]
    """
    p = price_df["close"].values
    n = len(p)
    y_max = np.full(n, np.nan)
    y_min = np.full(n, np.nan)
    for i in range(n - horizon):
        future = p[i + 1 : i + horizon + 1]
        y_max[i] = (future.max() / p[i] - 1) * 100
        y_min[i] = (future.min() / p[i] - 1) * 100
    return pd.DataFrame({"y_max": y_max, "y_min": y_min}, index=price_df.index)


# ─────────────────────────────────────────────────────────────────────────────
# 特徴量 DataFrame の構築（predict_stock.add_technical_features を再利用）
# ─────────────────────────────────────────────────────────────────────────────

def build_minmax_features(
    price_df:  pd.DataFrame,
    fund_row:  pd.Series,
    horizon:   int,
) -> pd.DataFrame:
    """テクニカル + ファンダメンタルズ特徴量 + Min/Max ラベルを返す。"""
    df = price_df.copy()
    df = ps.add_technical_features(df)

    labels_df = make_minmax_labels(price_df, horizon)
    df = df.join(labels_df)

    for col in ps.FUND_COLS:
        if col in fund_row.index:
            df[f"fund_{col}"] = fund_row[col]

    df["date"] = df.index
    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# データセット構築（全銘柄）
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(fund_df, price_map, horizon):
    fund_idx = fund_df.set_index("ティッカー")
    frames   = []
    skip     = 0
    for ticker, price_df in price_map.items():
        if ticker not in fund_idx.index:
            skip += 1
            continue
        df = build_minmax_features(price_df, fund_idx.loc[ticker], horizon)
        if len(df) < MIN_ROWS:
            skip += 1
            continue
        df["ticker"] = ticker
        frames.append(df)
    if not frames:
        raise RuntimeError("データが空です。")

    dataset = pd.concat(frames, ignore_index=True)

    # 特徴量列（y_max / y_min / close / date / ticker 以外）
    exclude   = {"y_max", "y_min", "close", "date", "ticker"}
    feat_cols = [
        c for c in dataset.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(dataset[c])
    ]
    print(f"  データセット: {len(dataset):,} 行 / 使用銘柄: {len(frames)} / "
          f"特徴量: {len(feat_cols)} 個")
    return dataset, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# モデル学習・評価
# ─────────────────────────────────────────────────────────────────────────────

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1,
)


def train_regressor(dataset, feat_cols, target_col):
    dataset = dataset.sort_index()
    split   = int(len(dataset) * TRAIN_RATIO)
    X_tr    = dataset.iloc[:split][feat_cols].values
    y_tr    = dataset.iloc[:split][target_col].values
    X_te    = dataset.iloc[split:][feat_cols].values
    y_te    = dataset.iloc[split:][target_col].values

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(**RF_PARAMS)),
    ])
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    # MAPE（ゼロ除算を回避）
    mask = np.abs(y_te) > 0.01
    mape = np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100

    # Hit rate: 予測の符号が実際と一致する割合
    sign_hit = np.mean(np.sign(y_pred) == np.sign(y_te)) * 100

    return model, {
        "y_te": y_te, "y_pred": y_pred,
        "mae": mae, "rmse": rmse, "mape": mape, "sign_hit": sign_hit,
        "feat_cols": feat_cols,
    }


# ─────────────────────────────────────────────────────────────────────────────
# グラフ1: 予測精度の確認（散布図 + 誤差分布）
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy(res_max, res_min, horizon):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Min / Max 予測精度の検証  （HORIZON={horizon}日 / 全1025銘柄 テストセット）",
        fontsize=13,
    )

    for row_idx, (res, label, color) in enumerate([
        (res_max, "Max 変化率（上値）", "crimson"),
        (res_min, "Min 変化率（下値）", "navy"),
    ]):
        y_te   = res["y_te"]
        y_pred = res["y_pred"]

        # ── 左列: 散布図 ────────────────────────────────────────────
        ax_s = axes[row_idx][0]

        # 密度で色付け（ランダムサンプリングで描画を軽量化）
        idx_sample = np.random.choice(len(y_te), size=min(5000, len(y_te)), replace=False)
        ax_s.scatter(
            y_te[idx_sample], y_pred[idx_sample],
            s=4, alpha=0.25, color=color,
        )
        # 45度線（完全予測ライン）
        lo = min(y_te.min(), y_pred.min())
        hi = max(y_te.max(), y_pred.max())
        ax_s.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6, label="完全予測ライン")
        ax_s.axhline(0, color="gray", lw=0.7, ls=":")
        ax_s.axvline(0, color="gray", lw=0.7, ls=":")

        # 統計テキスト
        ax_s.text(
            0.03, 0.97,
            f"MAE  = {res['mae']:.3f}%\n"
            f"RMSE = {res['rmse']:.3f}%\n"
            f"MAPE = {res['mape']:.1f}%\n"
            f"符号一致率 = {res['sign_hit']:.1f}%",
            transform=ax_s.transAxes,
            va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax_s.set_xlabel("実際の変化率 (%)")
        ax_s.set_ylabel("予測変化率 (%)")
        ax_s.set_title(f"{label} — 予測 vs 実際（散布図）")
        ax_s.legend(fontsize=8)
        ax_s.grid(True, alpha=0.2)

        # ── 右列: 予測誤差のヒストグラム ────────────────────────────
        ax_h = axes[row_idx][1]
        errors = y_pred - y_te
        ax_h.hist(errors, bins=80, color=color, alpha=0.7, edgecolor="white")
        ax_h.axvline(0,             color="black", lw=1.5, ls="--", label="誤差ゼロ")
        ax_h.axvline(errors.mean(), color="gold",  lw=1.5, ls="-",
                     label=f"平均誤差 {errors.mean():+.3f}%")
        ax_h.set_xlabel("予測誤差 (予測 − 実際)  %")
        ax_h.set_ylabel("頻度")
        ax_h.set_title(f"{label} — 予測誤差の分布")
        ax_h.legend(fontsize=8)
        ax_h.grid(True, alpha=0.2)

        # 誤差の±1σ範囲を塗る
        mu, sg = errors.mean(), errors.std()
        ax_h.axvspan(mu - sg, mu + sg, alpha=0.12, color=color,
                     label=f"±1σ ({mu-sg:.2f}% ~ {mu+sg:.2f}%)")

    fig.tight_layout()
    out = OUTPUT_DIR / "minmax_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 精度検証グラフ  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# グラフ2: 3076.T / 5970.T の株価チャートに予測レンジを重ねる
# ─────────────────────────────────────────────────────────────────────────────

def plot_minmax_chart(fund_df, price_map, model_max, model_min, res_max, horizon):
    feat_cols = res_max["feat_cols"]
    fund_idx  = fund_df.set_index("ティッカー")

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        f"3076.T / 5970.T  —  株価と {horizon}日後 Min/Max 予測レンジ\n"
        f"（赤帯=予測Max範囲 / 青帯=予測Min範囲 / 直近テスト期間）",
        fontsize=13,
    )

    for ax, ticker in zip(axes, TARGETS):
        if ticker not in price_map or ticker not in fund_idx.index:
            continue

        price_df = price_map[ticker]
        df       = build_minmax_features(price_df, fund_idx.loc[ticker], horizon)
        if df.empty:
            continue

        # 訓練/テスト分割
        split    = int(len(df) * TRAIN_RATIO)
        test_df  = df.iloc[split:]

        avail    = [c for c in feat_cols if c in test_df.columns]
        if len(avail) != len(feat_cols):
            continue

        X_te     = test_df[avail].values
        pred_max = model_max.predict(X_te)   # 変化率(%)
        pred_min = model_min.predict(X_te)

        dates    = pd.to_datetime(test_df["date"])
        prices   = test_df["close"].values

        # 予測 Max/Min の円換算
        price_pred_max = prices * (1 + pred_max / 100)
        price_pred_min = prices * (1 + pred_min / 100)

        # 実際の Max/Min の円換算（正解）
        price_true_max = prices * (1 + test_df["y_max"].values / 100)
        price_true_min = prices * (1 + test_df["y_min"].values / 100)

        color = COLORS[ticker]
        name  = NAMES[ticker]

        # 株価
        ax.plot(dates, prices, lw=1.4, color=color,
                label=f"終値（{ticker} {name}）", zorder=4)

        # 予測レンジ帯
        ax.fill_between(dates, prices, price_pred_max,
                        alpha=0.20, color="crimson", label="予測 Max レンジ")
        ax.fill_between(dates, price_pred_min, prices,
                        alpha=0.20, color="navy",    label="予測 Min レンジ")
        ax.plot(dates, price_pred_max, lw=0.9, color="crimson", ls="--", alpha=0.8,
                label="予測 Max ライン")
        ax.plot(dates, price_pred_min, lw=0.9, color="navy",    ls="--", alpha=0.8,
                label="予測 Min ライン")

        # 実際の Max/Min ライン（正解）
        ax.plot(dates, price_true_max, lw=0.8, color="salmon",  ls=":",  alpha=0.7,
                label="実際 Max（正解）")
        ax.plot(dates, price_true_min, lw=0.8, color="skyblue", ls=":",  alpha=0.7,
                label="実際 Min（正解）")

        # 個別 MAE を計算
        mae_max = mean_absolute_error(test_df["y_max"].values, pred_max)
        mae_min = mean_absolute_error(test_df["y_min"].values, pred_min)

        ax.text(
            0.01, 0.97,
            f"テストセット  MAE  Max={mae_max:.2f}%  /  Min={mae_min:.2f}%",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )

        ax.set_ylabel("株価 (円)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_title(f"{ticker}  {name}", fontsize=11, color=color, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right", ncol=3)
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "minmax_chart.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] Min/Maxチャート → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# グラフ3: 最新予測のサマリー（今後HORIZON日の見通し）
# ─────────────────────────────────────────────────────────────────────────────

def plot_latest_forecast(fund_df, price_map, model_max, model_min, res_max, horizon):
    feat_cols = res_max["feat_cols"]
    fund_idx  = fund_df.set_index("ティッカー")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        f"今後 {horizon} 営業日の株価 Min/Max 予測レンジ（最新データ基準）\n"
        f"【予測日: 2026-06-09】",
        fontsize=13,
    )

    latest_rows = []

    for ax, ticker in zip(axes, TARGETS):
        if ticker not in price_map or ticker not in fund_idx.index:
            continue

        price_df = price_map[ticker]
        df       = build_minmax_features(price_df, fund_idx.loc[ticker], horizon)
        if df.empty:
            continue

        # 直近60日を取得
        recent  = df.tail(60).copy()
        dates   = pd.to_datetime(recent["date"])
        prices  = recent["close"].values

        avail   = [c for c in feat_cols if c in recent.columns]
        if len(avail) != len(feat_cols):
            continue

        pred_max_pct = model_max.predict(recent[avail].values)
        pred_min_pct = model_min.predict(recent[avail].values)

        price_pred_max = prices * (1 + pred_max_pct / 100)
        price_pred_min = prices * (1 + pred_min_pct / 100)

        # 最新の予測値
        now_price = prices[-1]
        now_max   = price_pred_max[-1]
        now_min   = price_pred_min[-1]
        now_max_p = pred_max_pct[-1]
        now_min_p = pred_min_pct[-1]

        latest_rows.append({
            "ticker": ticker, "name": NAMES[ticker],
            "現在値": now_price,
            "予測Max円": now_max, "予測Max%": now_max_p,
            "予測Min円": now_min, "予測Min%": now_min_p,
        })

        color = COLORS[ticker]
        name  = NAMES[ticker]

        # ── 直近60日の株価 + 予測レンジ帯 ──────────────────────────
        ax.plot(dates, prices, lw=1.5, color=color, label="終値", zorder=4)

        ax.fill_between(dates, prices, price_pred_max,
                        alpha=0.25, color="crimson")
        ax.fill_between(dates, price_pred_min, prices,
                        alpha=0.25, color="navy")
        ax.plot(dates, price_pred_max, lw=1.0, color="crimson", ls="--",
                label=f"予測Max: ¥{now_max:,.0f} ({now_max_p:+.1f}%)")
        ax.plot(dates, price_pred_min, lw=1.0, color="navy",    ls="--",
                label=f"予測Min: ¥{now_min:,.0f} ({now_min_p:+.1f}%)")

        # 最新終値を強調
        ax.scatter(dates.iloc[-1], now_price,
                   s=100, color=color, zorder=6)
        ax.annotate(
            f"  現在 ¥{now_price:,.0f}",
            xy=(dates.iloc[-1], now_price),
            fontsize=10, color=color, fontweight="bold", va="center",
        )

        # 予測レンジ幅のテキスト
        range_pct = now_max_p - now_min_p
        ax.text(
            0.02, 0.97,
            f"予測レンジ幅: {range_pct:.1f}%\n"
            f"  上値余地: {now_max_p:+.1f}%  →  ¥{now_max:,.0f}\n"
            f"  下値リスク: {now_min_p:+.1f}%  →  ¥{now_min:,.0f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9),
        )

        ax.set_title(f"{ticker}  {name}", fontsize=11, color=color, fontweight="bold")
        ax.set_ylabel("株価 (円)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.tick_params(axis="x", rotation=35, labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "minmax_latest.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 最新予測サマリー → {out}")
    return latest_rows


# ─────────────────────────────────────────────────────────────────────────────
# 予測精度の理論的限界を説明するグラフ
# ─────────────────────────────────────────────────────────────────────────────

def plot_difficulty(fund_df, price_map, res_max, res_min, horizon):
    """
    予測精度の限界を視覚化:
      - 実際の Max/Min 変化率のヒストグラム（データの散らばり）
      - ベースライン（単純平均予測）vs モデルの誤差比較
    """
    fund_idx = fund_df.set_index("ティッカー")

    all_y_max, all_y_min = [], []
    for ticker in TARGETS:
        if ticker not in price_map or ticker not in fund_idx.index:
            continue
        price_df = price_map[ticker]
        df = build_minmax_features(price_df, fund_idx.loc[ticker], horizon)
        if not df.empty:
            all_y_max.extend(df["y_max"].dropna().tolist())
            all_y_min.extend(df["y_min"].dropna().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Min/Max 予測の精度限界と難易度分析  （HORIZON={horizon}日）",
        fontsize=13,
    )

    for row_idx, (yvals, res, label, color) in enumerate([
        (all_y_max, res_max, "Max 変化率（上値）", "crimson"),
        (all_y_min, res_min, "Min 変化率（下値）", "navy"),
    ]):
        yvals  = np.array(yvals)
        y_te   = res["y_te"]
        y_pred = res["y_pred"]

        # ── 左列: 実際の変化率分布 ─────────────────────────────────
        ax_l = axes[row_idx][0]
        ax_l.hist(yvals, bins=80, color=color, alpha=0.65, edgecolor="white",
                  label="実際の分布（2銘柄）")
        mu, sg = yvals.mean(), yvals.std()
        ax_l.axvline(mu, color="black", lw=1.5, ls="--",
                     label=f"平均 {mu:.2f}%")
        ax_l.axvline(mu + sg, color="gray", lw=1, ls=":",
                     label=f"±1σ = {sg:.2f}%")
        ax_l.axvline(mu - sg, color="gray", lw=1, ls=":")
        ax_l.axvspan(mu - sg, mu + sg, alpha=0.1, color=color)
        ax_l.set_xlabel("変化率 (%)")
        ax_l.set_ylabel("頻度")
        ax_l.set_title(f"{label} — 実際の分布\n"
                       f"（mean={mu:.2f}%  σ={sg:.2f}%）")
        ax_l.legend(fontsize=7.5)
        ax_l.grid(True, alpha=0.2)

        # ── 右列: ベースライン vs モデル の誤差比較 ─────────────────
        ax_r = axes[row_idx][1]

        baseline_pred = np.full_like(y_te, y_te[:int(len(y_te)*0.5)].mean())
        mae_base = mean_absolute_error(y_te, baseline_pred)
        mae_model = res["mae"]

        improvement = (1 - mae_model / mae_base) * 100

        bars = ax_r.bar(
            ["ベースライン\n（訓練平均値）", "RandomForest\nモデル"],
            [mae_base, mae_model],
            color=["lightgray", color],
            edgecolor="white",
            width=0.5,
        )
        ax_r.set_ylabel("MAE (%)")
        ax_r.set_title(f"{label} — ベースライン vs モデル（MAE比較）\n"
                       f"改善率: {improvement:.1f}%")
        ax_r.grid(True, alpha=0.2, axis="y")

        for bar, v in zip(bars, [mae_base, mae_model]):
            ax_r.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.005,
                      f"{v:.3f}%", ha="center", va="bottom", fontsize=10,
                      fontweight="bold")

        # 符号一致率（方向性の精度）
        ax_r.text(
            0.97, 0.95,
            f"符号一致率（方向性）: {res['sign_hit']:.1f}%\n"
            f"MAPE: {res['mape']:.0f}%",
            transform=ax_r.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
        )

    fig.tight_layout()
    out = OUTPUT_DIR / "minmax_difficulty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 難易度分析グラフ → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("株価 Min / Max 回帰予測プログラム")
    print(f"  予測ホライズン: {HORIZON} 営業日後")
    print(f"  対象銘柄      : {', '.join(TARGETS)}")
    print("=" * 65)

    # データ読み込み（Parquet キャッシュ）
    print("\n[Step 1] データ読み込み")
    fund_df   = ps.load_fundamentals()
    tickers   = fund_df["ティッカー"].tolist()

    old_h, old_t   = ps.HORIZON, ps.THRESHOLD
    ps.HORIZON     = HORIZON
    ps.THRESHOLD   = 0.02
    price_map      = ps.download_prices_batch(tickers)
    ps.HORIZON, ps.THRESHOLD = old_h, old_t
    print(f"  {len(price_map)} 銘柄のキャッシュ読み込み完了")

    # データセット構築
    print("\n[Step 2] データセット構築（全1025銘柄）")
    dataset, feat_cols = build_dataset(fund_df, price_map, HORIZON)

    # モデル学習（Max / Min それぞれ）
    print("\n[Step 3] 回帰モデル学習")
    print("  ── Max 変化率モデル ──")
    model_max, res_max = train_regressor(dataset, feat_cols, "y_max")
    print(f"     MAE={res_max['mae']:.3f}%  RMSE={res_max['rmse']:.3f}%  "
          f"MAPE={res_max['mape']:.1f}%  符号一致率={res_max['sign_hit']:.1f}%")

    print("  ── Min 変化率モデル ──")
    model_min, res_min = train_regressor(dataset, feat_cols, "y_min")
    print(f"     MAE={res_min['mae']:.3f}%  RMSE={res_min['rmse']:.3f}%  "
          f"MAPE={res_min['mape']:.1f}%  符号一致率={res_min['sign_hit']:.1f}%")

    # グラフ生成
    print("\n[Step 4] グラフ生成")
    plot_accuracy(res_max, res_min, HORIZON)
    plot_minmax_chart(fund_df, price_map, model_max, model_min, res_max, HORIZON)
    latest = plot_latest_forecast(fund_df, price_map, model_max, model_min, res_max, HORIZON)
    plot_difficulty(fund_df, price_map, res_max, res_min, HORIZON)

    # 最新予測サマリー
    print(f"\n{'='*65}")
    print(f"【最新予測サマリー】今後 {HORIZON} 営業日の株価レンジ予測")
    print("=" * 65)
    for row in latest:
        print(f"\n  {row['ticker']}  {row['name']}")
        print(f"    現在株価    : ¥{row['現在値']:,.0f}")
        print(f"    予測 Max    : ¥{row['予測Max円']:,.0f}  ({row['予測Max%']:+.1f}%)")
        print(f"    予測 Min    : ¥{row['予測Min円']:,.0f}  ({row['予測Min%']:+.1f}%)")
        print(f"    予測レンジ幅: {row['予測Max%'] - row['予測Min%']:.1f}%")

    print(f"\n{'='*65}")
    print("【予測精度の限界について】")
    print(f"  Max MAE : {res_max['mae']:.2f}%  （予測の平均絶対誤差）")
    print(f"  Min MAE : {res_min['mae']:.2f}%")
    print(f"  Max 符号一致率: {res_max['sign_hit']:.1f}%  （方向性の的中率）")
    print(f"  Min 符号一致率: {res_min['sign_hit']:.1f}%")
    print(f"  ※ MAPE が高い（Max:{res_max['mape']:.0f}% / Min:{res_min['mape']:.0f}%）ため")
    print(f"    個別の絶対値予測より「レンジ幅の傾向把握」に適しています。")
    print("=" * 65)

    print("\n出力ファイル:")
    for f in sorted(OUTPUT_DIR.glob("minmax_*.png")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
