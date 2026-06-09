#!/usr/bin/env python3
"""
visualize_forecast.py — 株価予測 可視化スクリプト
=================================================
predict_stock.py のロジックを再利用し、HORIZON=5（5営業日後）で
以下のグラフを生成する。

  Part 1  今後5営業日の予測
    - forecast_5day_ranking.png  : 全銘柄上昇確率 TOP20 横棒グラフ
    - forecast_5day_final7.png   : Final7 銘柄の上昇確率バーチャート

  Part 2  Final7 トップ・最下位の可視化
    - forecast_top_bottom.png    : トップ・最下位の株価チャート + 上昇確率推移

  Part 3  銘柄別予測精度の比較
    - accuracy_comparison.png    : Final7 各銘柄 AUC / Accuracy + 最新確率対比
    - accuracy_radar.png         : Final7 の精度レーダーチャート
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import chart_style  # noqa: F401
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import predict_stock as ps

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
HORIZON_5D   = 5     # 短期予測ホライズン（営業日）
THRESHOLD_5D = 0.02  # 5日後の上昇判定閾値（2% 超）
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Final7 ティッカー
with open(OUTPUT_DIR / "final7.json", encoding="utf-8") as _f:
    _f7 = json.load(_f)
FINAL7_TICKERS = _f7["tickers"]
FINAL7_DATE    = _f7["selected_date"]

# Final7 会社名マップ（後で fund_df から上書き）
F7_NAMES: dict[str, str] = {d["ティッカー"]: d["会社名"] for d in _f7["details"]}

# カラーパレット（Final7 7銘柄固定色）
F7_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a",
             "#f4a261", "#264653", "#a8dadc"]
F7_COLOR_MAP = dict(zip(FINAL7_TICKERS, F7_COLORS))


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def _set_horizon(h: int, t: float) -> tuple[int, float]:
    """ps.HORIZON / ps.THRESHOLD を切り替え、旧値を返す。"""
    old_h, old_t = ps.HORIZON, ps.THRESHOLD
    ps.HORIZON, ps.THRESHOLD = h, t
    return old_h, old_t


def _restore_horizon(old_h: int, old_t: float) -> None:
    ps.HORIZON, ps.THRESHOLD = old_h, old_t


def get_features(
    ticker: str,
    fund_idx: pd.DataFrame,
    price_map: dict,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    """指定 HORIZON で特徴量 DataFrame を返す。"""
    if ticker not in price_map or ticker not in fund_idx.index:
        return pd.DataFrame()
    old = _set_horizon(horizon, threshold)
    feat = ps.build_features(price_map[ticker], fund_idx.loc[ticker], ticker)
    _restore_horizon(*old)
    return feat


def get_proba_series(feat_df: pd.DataFrame, feat_cols: list, model) -> np.ndarray:
    """全行の上昇確率を返す。"""
    avail = [c for c in feat_cols if c in feat_df.columns]
    if len(avail) != len(feat_cols):
        return np.full(len(feat_df), np.nan)
    return model.predict_proba(feat_df[avail].values)[:, 1]


def latest_proba(feat_df: pd.DataFrame, feat_cols: list, model) -> float:
    """最新行の上昇確率を返す。"""
    s = get_proba_series(feat_df, feat_cols, model)
    return float(s[-1]) if len(s) else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Step A: HORIZON=5 のモデル学習
# ─────────────────────────────────────────────────────────────────────────────

def train_5day_model(fund_df, price_map) -> tuple[pd.DataFrame, list, dict]:
    print(f"\n[モデル学習] HORIZON={HORIZON_5D}日 / 閾値={THRESHOLD_5D*100:.0f}%")
    old = _set_horizon(HORIZON_5D, THRESHOLD_5D)
    dataset, feat_cols = ps.build_dataset(fund_df, price_map)
    results            = ps.train_and_evaluate(dataset, feat_cols)
    _restore_horizon(*old)
    return dataset, feat_cols, results


# ─────────────────────────────────────────────────────────────────────────────
# Step B: 全銘柄の最新シグナルを収集
# ─────────────────────────────────────────────────────────────────────────────

def collect_signals(fund_df, price_map, results) -> pd.DataFrame:
    rf_res    = results["RandomForest"]
    model     = rf_res["model"]
    feat_cols = rf_res["feat_cols"]
    fund_idx  = fund_df.set_index("ティッカー")

    rows = []
    old  = _set_horizon(HORIZON_5D, THRESHOLD_5D)
    for ticker, price_df in price_map.items():
        if ticker not in fund_idx.index:
            continue
        feat_df = ps.build_features(price_df, fund_idx.loc[ticker], ticker)
        if feat_df.empty:
            continue
        p = latest_proba(feat_df, feat_cols, model)
        if np.isnan(p):
            continue
        fund_row = fund_idx.loc[ticker]
        rows.append({
            "ティッカー": ticker,
            "会社名":     str(fund_row.get("会社名", ticker))[:20],
            "業種":       str(fund_row.get("業種", "")),
            "上昇確率":   round(p, 4),
            "予測":       "↑上昇" if p >= 0.5 else "↓下落",
            "現在株価":   round(float(feat_df["close"].iloc[-1]), 0),
            "総合点":     float(fund_row.get("総合点", np.nan)),
            "is_final7":  ticker in FINAL7_TICKERS,
        })
    _restore_horizon(*old)

    return pd.DataFrame(rows).sort_values("上昇確率", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1-A: 全銘柄 上昇確率 TOP20 横棒グラフ
# ─────────────────────────────────────────────────────────────────────────────

def plot_ranking_top20(df_sig: pd.DataFrame) -> None:
    top = df_sig.head(20).copy()
    colors = [
        F7_COLOR_MAP.get(t, "steelblue") if t in FINAL7_TICKERS else "steelblue"
        for t in top["ティッカー"]
    ]
    labels = [f"{r['ティッカー']} {r['会社名']}" for _, r in top.iterrows()]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top)), top["上昇確率"].values[::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5)
    ax.axvline(0.5, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel(f"上昇確率  ({HORIZON_5D}営業日後 / 閾値 {THRESHOLD_5D*100:.0f}%超)")
    ax.set_title(
        f"今後 {HORIZON_5D} 営業日の上昇確率 TOP20\n"
        f"（星形マーク = Final7 銘柄）",
        fontsize=12,
    )

    for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        ax.text(row["上昇確率"] + 0.012, i, f"{row['上昇確率']:.1%}",
                va="center", fontsize=8)
        if row["is_final7"]:
            ax.text(0.005, i, "★", color="white", va="center", fontsize=9,
                    fontweight="bold")

    legend = [
        mpatches.Patch(color="steelblue",  label="その他銘柄"),
        mpatches.Patch(color="darkorange", label="Final7 銘柄"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "forecast_5day_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 5日予測ランキング TOP20     → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1-B: Final7 の 5日後予測バーチャート
# ─────────────────────────────────────────────────────────────────────────────

def plot_final7_bar(df_sig: pd.DataFrame) -> None:
    f7 = (
        df_sig[df_sig["ティッカー"].isin(FINAL7_TICKERS)]
        .sort_values("上昇確率")
        .copy()
    )

    labels = [f"{r['ティッカー']}\n{r['会社名']}" for _, r in f7.iterrows()]
    colors = [F7_COLOR_MAP.get(t, "gray") for t in f7["ティッカー"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(range(len(f7)), f7["上昇確率"].values,
                   color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(f7)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.5, color="black", lw=1.5, ls="--")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(f"上昇確率  ({HORIZON_5D}営業日後 / 閾値 {THRESHOLD_5D*100:.0f}%超)")
    ax.set_title(
        f"Final7 銘柄  —  今後 {HORIZON_5D} 営業日の騰落予測\n"
        f"【予測日: {FINAL7_DATE}】",
        fontsize=12,
    )

    for i, (_, row) in enumerate(f7.iterrows()):
        label_txt = f"{row['上昇確率']:.1%}  {'↑上昇' if row['上昇確率'] >= 0.5 else '↓下落'}"
        ax.text(row["上昇確率"] + 0.012, i, label_txt, va="center", fontsize=9)

    # トップ / 最下位 ラベル
    ax.text(1.02, len(f7) - 1, "◀ TOP",    va="center", fontsize=9,
            color="crimson", fontweight="bold")
    ax.text(1.02, 0,          "◀ BOTTOM", va="center", fontsize=9,
            color="navy",   fontweight="bold")

    fig.tight_layout()
    out = OUTPUT_DIR / "forecast_5day_final7.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] Final7 5日予測バーチャート  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: トップ・最下位の株価チャート + 上昇確率推移
# ─────────────────────────────────────────────────────────────────────────────

def plot_top_bottom_detail(
    df_sig:    pd.DataFrame,
    fund_df:   pd.DataFrame,
    price_map: dict,
    results:   dict,
) -> tuple[str, str]:
    """
    Final7 の上昇確率トップと最下位の銘柄について
    (a) 直近 60 日の株価チャート
    (b) 上昇確率の推移
    を 2×2 グリッドで表示する。
    返り値: (top_ticker, bottom_ticker)
    """
    rf_res    = results["RandomForest"]
    model     = rf_res["model"]
    feat_cols = rf_res["feat_cols"]
    fund_idx  = fund_df.set_index("ティッカー")

    f7_sig = (
        df_sig[df_sig["ティッカー"].isin(FINAL7_TICKERS)]
        .sort_values("上昇確率", ascending=False)
        .reset_index(drop=True)
    )
    top_row    = f7_sig.iloc[0]
    bottom_row = f7_sig.iloc[-1]
    top_t      = top_row["ティッカー"]
    bot_t      = bottom_row["ティッカー"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Final7  トップ vs 最下位  —  株価チャートと上昇確率推移\n"
        f"（{HORIZON_5D}営業日後予測 / 直近60日）",
        fontsize=12,
    )

    for col_idx, (ticker, row) in enumerate(
        [(top_t, top_row), (bot_t, bottom_row)]
    ):
        feat_df = get_features(ticker, fund_idx, price_map, HORIZON_5D, THRESHOLD_5D)
        if feat_df.empty or len(feat_df) < 30:
            continue

        recent = feat_df.tail(60).copy()
        dates  = pd.to_datetime(recent["date"])
        prices = recent["close"].values
        proba  = get_proba_series(recent, feat_cols, model)

        color_accent = F7_COLOR_MAP.get(ticker, "steelblue")
        company      = str(row["会社名"])
        tag          = "【TOP】" if col_idx == 0 else "【最下位】"

        # ── 上段: 株価チャート ───────────────────────────────────────
        ax_p = axes[0][col_idx]
        ax_p.plot(dates, prices, lw=1.5, color=color_accent)
        ax_p.scatter(dates.iloc[-1], prices[-1],
                     color="black", s=70, zorder=5)

        # 予測方向を矢印で表示（最終点から右外へ）
        arrow_color = "crimson" if proba[-1] >= 0.5 else "navy"
        arrow_dy    = (prices.max() - prices.min()) * (0.06 if proba[-1] >= 0.5 else -0.06)
        ax_p.annotate(
            f"{'↑ 上昇予測' if proba[-1] >= 0.5 else '↓ 下落予測'}\n{proba[-1]:.1%}",
            xy=(dates.iloc[-1], prices[-1]),
            xytext=(dates.iloc[-1], prices[-1] + arrow_dy),
            fontsize=9, color=arrow_color, fontweight="bold",
            ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
        )

        ax_p.set_title(f"{tag}  {ticker}  {company}\n"
                       f"現在株価: {prices[-1]:,.0f}円  "
                       f"上昇確率: {proba[-1]:.1%}",
                       fontsize=9)
        ax_p.set_ylabel("株価 (円)")
        ax_p.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        ax_p.grid(True, alpha=0.3)
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_p.tick_params(axis="x", rotation=35)

        # ── 下段: 上昇確率推移 ────────────────────────────────────────
        ax_q = axes[1][col_idx]
        ax_q.fill_between(dates, proba, 0.5,
                          where=(proba >= 0.5), alpha=0.45, color="crimson",
                          label="上昇域")
        ax_q.fill_between(dates, proba, 0.5,
                          where=(proba < 0.5),  alpha=0.45, color="navy",
                          label="下落域")
        ax_q.plot(dates, proba, lw=1.2, color="black", alpha=0.7)
        ax_q.axhline(0.5, color="gray", lw=1, ls="--")
        ax_q.scatter(dates.iloc[-1], proba[-1],
                     color=arrow_color, s=70, zorder=5)
        ax_q.set_ylim(0, 1)
        ax_q.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_q.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_q.set_ylabel("上昇確率")
        ax_q.set_xlabel("日付")
        ax_q.grid(True, alpha=0.3)
        ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_q.tick_params(axis="x", rotation=35)
        ax_q.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    out = OUTPUT_DIR / "forecast_top_bottom.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] トップ・最下位チャート       → {out}")
    return top_t, bot_t


# ─────────────────────────────────────────────────────────────────────────────
# Part 3-A: Final7 銘柄別 テストセット精度の計算
# ─────────────────────────────────────────────────────────────────────────────

def calc_per_ticker_metrics(
    fund_df:   pd.DataFrame,
    price_map: dict,
    results:   dict,
) -> pd.DataFrame:
    """Final7 各銘柄のテストセット精度指標を計算して返す。"""
    rf_res    = results["RandomForest"]
    model     = rf_res["model"]
    feat_cols = rf_res["feat_cols"]
    fund_idx  = fund_df.set_index("ティッカー")

    rows = []
    for ticker in FINAL7_TICKERS:
        feat_df = get_features(ticker, fund_idx, price_map, HORIZON_5D, THRESHOLD_5D)
        if len(feat_df) < ps.MIN_ROWS * 2:
            continue

        split   = int(len(feat_df) * ps.TRAIN_RATIO)
        test_df = feat_df.iloc[split:]
        if test_df.empty:
            continue

        avail  = [c for c in feat_cols if c in test_df.columns]
        if len(avail) != len(feat_cols):
            continue

        y_true  = test_df["label"].values
        y_proba = model.predict_proba(test_df[avail].values)[:, 1]
        y_pred  = (y_proba >= 0.5).astype(int)

        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, y_proba)
        acc = accuracy_score(y_true, y_pred)

        # 上昇時の精度（Precision for class=1）
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.0

        rows.append({
            "ティッカー": ticker,
            "会社名":     F7_NAMES.get(ticker, ticker),
            "AUC":        round(auc, 4),
            "Accuracy":   round(acc, 4),
            "Precision":  round(precision, 4),
            "Recall":     round(recall, 4),
            "F1":         round(f1, 4),
            "上昇率(実)": round(y_true.mean(), 4),
            "テスト行数": len(test_df),
        })

    return pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Part 3-B: 精度比較グラフ（横棒 + 最新確率対比）
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(
    df_acc:  pd.DataFrame,
    df_sig:  pd.DataFrame,
    top_t:   str,
    bot_t:   str,
) -> None:
    """
    3段グラフ:
      上段: AUC / Accuracy / F1 の銘柄別比較
      中段: 最新上昇確率 vs 予測精度（AUC）の散布図的バー
      下段: トップ vs 最下位の精度詳細（Precision / Recall / F1）
    """
    if df_acc.empty:
        print("[警告] 精度データが空です。スキップします。")
        return

    # df_sig から最新確率を付与
    sig_idx = df_sig.set_index("ティッカー")
    df_acc  = df_acc.copy()
    df_acc["最新上昇確率"] = df_acc["ティッカー"].map(
        lambda t: sig_idx.loc[t, "上昇確率"] if t in sig_idx.index else np.nan
    )
    df_acc["color"] = df_acc["ティッカー"].map(
        lambda t: F7_COLOR_MAP.get(t, "gray")
    )

    labels = [f"{r['ティッカー']}\n{r['会社名'][:10]}" for _, r in df_acc.iterrows()]
    x      = np.arange(len(df_acc))
    w      = 0.25

    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    fig.suptitle(
        f"Final7  銘柄別 予測精度の比較  ({HORIZON_5D}日後予測)\n"
        f"テスト期間: 直近{int(100 - ps.TRAIN_RATIO*100):.0f}%のデータ",
        fontsize=12,
    )

    # ── 上段: AUC / Accuracy / F1 ────────────────────────────────────
    ax0 = axes[0]
    bars_auc = ax0.bar(x - w,     df_acc["AUC"],      w, label="ROC-AUC",
                       color="steelblue", alpha=0.85, edgecolor="white")
    bars_acc = ax0.bar(x,         df_acc["Accuracy"],  w, label="Accuracy",
                       color="darkorange", alpha=0.85, edgecolor="white")
    bars_f1  = ax0.bar(x + w,     df_acc["F1"],        w, label="F1（上昇クラス）",
                       color="seagreen", alpha=0.85, edgecolor="white")

    ax0.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="ランダム水準")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=8.5)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("スコア")
    ax0.set_title("① 精度指標の銘柄間比較", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.2, axis="y")

    for bars, vals in [(bars_auc, df_acc["AUC"]),
                       (bars_acc, df_acc["Accuracy"]),
                       (bars_f1,  df_acc["F1"])]:
        for bar, v in zip(bars, vals):
            ax0.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)

    # ── 中段: 最新上昇確率 vs AUC ────────────────────────────────────
    ax1 = axes[1]
    bar_p  = ax1.bar(x - w / 2, df_acc["最新上昇確率"], w,
                     color=df_acc["color"].tolist(), alpha=0.9, edgecolor="white",
                     label="最新上昇確率（本日予測）")
    bar_a  = ax1.bar(x + w / 2, df_acc["AUC"],         w,
                     color="goldenrod", alpha=0.75, edgecolor="white",
                     label="ROC-AUC（テスト精度）")

    ax1.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("値")
    ax1.set_title("② 最新予測確率（確信度）vs 予測精度（AUC）", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2, axis="y")

    # 予測確率に矢印アノテーション（TOP・最下位）
    for _, r in df_acc.iterrows():
        t = r["ティッカー"]
        i = df_acc.index[df_acc["ティッカー"] == t][0]
        if t == top_t:
            ax1.annotate("TOP", xy=(i - w / 2, r["最新上昇確率"]),
                         xytext=(i - w / 2, r["最新上昇確率"] + 0.08),
                         fontsize=8, color="crimson", fontweight="bold",
                         ha="center",
                         arrowprops=dict(arrowstyle="->", color="crimson"))
        elif t == bot_t:
            ax1.annotate("最下位", xy=(i - w / 2, r["最新上昇確率"]),
                         xytext=(i - w / 2, r["最新上昇確率"] - 0.10),
                         fontsize=8, color="navy", fontweight="bold",
                         ha="center",
                         arrowprops=dict(arrowstyle="->", color="navy"))

    for bar, v in zip(bar_p, df_acc["最新上昇確率"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{v:.1%}", ha="center", va="bottom", fontsize=7)

    # ── 下段: TOP vs 最下位の Precision / Recall / F1 詳細 ──────────
    ax2 = axes[2]
    pair_df = df_acc[df_acc["ティッカー"].isin([top_t, bot_t])].copy()
    pair_df = pair_df.set_index("ティッカー").reindex([top_t, bot_t])

    metrics    = ["Precision", "Recall", "F1", "AUC", "Accuracy"]
    x2         = np.arange(len(metrics))
    w2         = 0.35
    top_vals   = [pair_df.loc[top_t, m] if top_t in pair_df.index else 0 for m in metrics]
    bot_vals   = [pair_df.loc[bot_t, m] if bot_t in pair_df.index else 0 for m in metrics]
    top_name   = F7_NAMES.get(top_t, top_t)[:12]
    bot_name   = F7_NAMES.get(bot_t, bot_t)[:12]

    b_top = ax2.bar(x2 - w2 / 2, top_vals, w2, label=f"TOP  {top_t} {top_name}",
                    color="crimson", alpha=0.85, edgecolor="white")
    b_bot = ax2.bar(x2 + w2 / 2, bot_vals, w2, label=f"最下位  {bot_t} {bot_name}",
                    color="navy",   alpha=0.75, edgecolor="white")

    ax2.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics, fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("スコア")
    ax2.set_title("③ トップ vs 最下位  —  予測精度の詳細比較", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis="y")

    for bars, vals in [(b_top, top_vals), (b_bot, bot_vals)]:
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.012,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # 差分テキスト
    diffs = [t - b for t, b in zip(top_vals, bot_vals)]
    for xi, (metric, diff) in enumerate(zip(metrics, diffs)):
        color = "crimson" if diff > 0 else "navy"
        ax2.text(xi, max(top_vals[xi], bot_vals[xi]) + 0.07,
                 f"Δ{diff:+.3f}", ha="center", fontsize=7, color=color,
                 fontweight="bold")

    fig.tight_layout()
    out = OUTPUT_DIR / "accuracy_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[出力] 精度比較グラフ               → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 3-C: レーダーチャート（Final7 銘柄別精度プロファイル）
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(df_acc: pd.DataFrame) -> None:
    """Final7 各銘柄の精度 5 軸レーダーチャート。"""
    if df_acc.empty:
        return

    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    N       = len(metrics)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # 閉じる

    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={"polar": True})

    for _, row in df_acc.iterrows():
        ticker = row["ティッカー"]
        values = [float(row[m]) for m in metrics]
        values += values[:1]
        color   = F7_COLOR_MAP.get(ticker, "gray")
        company = F7_NAMES.get(ticker, ticker)[:10]
        ax.plot(angles, values, lw=1.5, color=color,
                label=f"{ticker} {company}")
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7)
    ax.axhline(y=0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"Final7 銘柄別  予測精度プロファイル\n"
                 f"（{HORIZON_5D}日後予測 / テストセット）",
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "accuracy_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[出力] 精度レーダーチャート         → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("株価予測 可視化スクリプト（5営業日予測版）")
    print(f"  Final7 選定日  : {FINAL7_DATE}")
    print(f"  Final7 銘柄    : {', '.join(FINAL7_TICKERS)}")
    print(f"  予測ホライズン : {HORIZON_5D} 営業日後")
    print(f"  上昇判定閾値  : {THRESHOLD_5D*100:.0f}% 超")
    print("=" * 65)

    # ── データ準備（Parquet キャッシュ使用）
    print("\n[Step 1] データ読み込み")
    fund_df = ps.load_fundamentals()
    tickers = fund_df["ティッカー"].tolist()

    print("\n[Step 2] 株価キャッシュ読み込み（ダウンロードなし）")
    old = _set_horizon(HORIZON_5D, THRESHOLD_5D)
    price_map = ps.download_prices_batch(tickers)
    _restore_horizon(*old)
    print(f"  キャッシュから {len(price_map)} 銘柄読み込み完了")

    # ── モデル学習（HORIZON=5）
    print("\n[Step 3] 5日後予測モデルを学習中...")
    _, feat_cols, results = train_5day_model(fund_df, price_map)

    # ── 全銘柄の最新シグナル収集
    print("\n[Step 4] 全銘柄の最新シグナルを収集中...")
    df_sig = collect_signals(fund_df, price_map, results)
    print(f"  {len(df_sig)} 銘柄のシグナルを生成")

    # ─────────────────── Part 1 ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Part 1] 今後5営業日の予測グラフ")
    plot_ranking_top20(df_sig)
    plot_final7_bar(df_sig)

    # ─────────────────── Part 2 ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Part 2] Final7 トップ・最下位の可視化")
    top_t, bot_t = plot_top_bottom_detail(df_sig, fund_df, price_map, results)

    # ─────────────────── Part 3 ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Part 3] Final7 銘柄別予測精度の比較")
    df_acc = calc_per_ticker_metrics(fund_df, price_map, results)

    print("\n  【精度サマリー（Final7）】")
    print(df_acc[["ティッカー", "会社名", "AUC", "Accuracy",
                  "Precision", "Recall", "F1",
                  "上昇率(実)", "テスト行数"]].to_string(index=False))

    plot_accuracy_comparison(df_acc, df_sig, top_t, bot_t)
    plot_radar(df_acc)

    # ── 完了
    print(f"\n{'='*65}")
    print("【完了】出力ファイル一覧")
    for f in sorted(
        list(OUTPUT_DIR.glob("forecast_*.png")) +
        list(OUTPUT_DIR.glob("accuracy_*.png"))
    ):
        print(f"  {f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
