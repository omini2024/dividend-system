#!/usr/bin/env python3
"""
plot_3076_5970.py — 3076.T (AI HOLDINGS) / 5970.T (G-TEKT) 詳細グラフ
=======================================================================
5日後予測モデルを使い、2銘柄の以下を横並びで可視化する。

  行1: 株価チャート + 移動平均（5/20/60日）+ 予測シグナル（↑↓）
  行2: RSI (14日) + 過買い・過売りライン
  行3: MACD ヒストグラム + シグナル線
  行4: 5日後上昇確率の推移（訓練/テスト期間を色分け）

出力: output/detail_3076_5970.png
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import chart_style  # noqa: F401
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import predict_stock as ps

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
TICKERS = ["3076.T", "5970.T"]
NAMES   = {
    "3076.T": "AI HOLDINGS CORPORATION",
    "5970.T": "G-TEKT CORPORATION",
}
COLORS  = {
    "3076.T": "#e63946",   # 赤系
    "5970.T": "#457b9d",   # 青系
}

HORIZON_5D   = 5
THRESHOLD_5D = 0.02
OUTPUT_DIR   = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# テクニカル指標の計算（可視化用に詳細版）
# ─────────────────────────────────────────────────────────────────────────────

def calc_indicators(price_series: pd.Series) -> pd.DataFrame:
    """終値から可視化用テクニカル指標を計算して DataFrame で返す。"""
    p   = price_series.copy()
    df  = pd.DataFrame(index=p.index)
    df["close"] = p

    # 移動平均
    df["ma5"]   = p.rolling(5).mean()
    df["ma20"]  = p.rolling(20).mean()
    df["ma60"]  = p.rolling(60).mean()

    # ボリンジャーバンド (±2σ)
    roll20          = p.rolling(20)
    df["bb_mid"]    = roll20.mean()
    df["bb_upper"]  = roll20.mean() + 2 * roll20.std()
    df["bb_lower"]  = roll20.mean() - 2 * roll20.std()

    # RSI (14日)
    delta           = p.diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]       = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD (12/26/9)
    ema12           = p.ewm(span=12, adjust=False).mean()
    ema26           = p.ewm(span=26, adjust=False).mean()
    macd_line       = ema12 - ema26
    signal_line     = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_line"] = macd_line
    df["macd_sig"]  = signal_line
    df["macd_hist"] = macd_line - signal_line

    return df


# ─────────────────────────────────────────────────────────────────────────────
# モデル学習（HORIZON=5, 全銘柄データ使用）
# ─────────────────────────────────────────────────────────────────────────────

def train_model(fund_df, price_map):
    print(f"[モデル学習] HORIZON={HORIZON_5D}日 / 閾値={THRESHOLD_5D*100:.0f}%...")
    old_h, old_t   = ps.HORIZON, ps.THRESHOLD
    ps.HORIZON     = HORIZON_5D
    ps.THRESHOLD   = THRESHOLD_5D
    dataset, feat_cols = ps.build_dataset(fund_df, price_map)
    results            = ps.train_and_evaluate(dataset, feat_cols)
    ps.HORIZON, ps.THRESHOLD = old_h, old_t
    return results, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 上昇確率の全期間計算
# ─────────────────────────────────────────────────────────────────────────────

def get_full_proba(ticker, fund_df, price_map, model, feat_cols):
    """全期間（訓練+テスト）の上昇確率を返す。"""
    fund_idx = fund_df.set_index("ティッカー")
    if ticker not in fund_idx.index or ticker not in price_map:
        return pd.Series(dtype=float), 0

    old_h, old_t = ps.HORIZON, ps.THRESHOLD
    ps.HORIZON   = HORIZON_5D
    ps.THRESHOLD = THRESHOLD_5D
    feat_df      = ps.build_features(price_map[ticker], fund_idx.loc[ticker], ticker)
    ps.HORIZON, ps.THRESHOLD = old_h, old_t

    if feat_df.empty:
        return pd.Series(dtype=float), 0

    avail = [c for c in feat_cols if c in feat_df.columns]
    if len(avail) != len(feat_cols):
        return pd.Series(dtype=float), 0

    proba  = model.predict_proba(feat_df[avail].values)[:, 1]
    dates  = pd.to_datetime(feat_df["date"])
    split  = int(len(feat_df) * ps.TRAIN_RATIO)

    return pd.Series(proba, index=dates), split


# ─────────────────────────────────────────────────────────────────────────────
# メイングラフ描画
# ─────────────────────────────────────────────────────────────────────────────

def plot_detail(fund_df, price_map, results, feat_cols):
    rf_model = results["RandomForest"]["model"]

    # ── 4行 × 2列レイアウト ─────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 2,
        figsize=(18, 18),
        gridspec_kw={"height_ratios": [3.5, 1.5, 1.5, 1.5]},
    )
    fig.suptitle(
        "3076.T  AI HOLDINGS  vs  5970.T  G-TEKT  —  株価詳細分析\n"
        f"（5営業日後 騰落予測 / 直近2年間）",
        fontsize=14, y=1.01,
    )

    for col, ticker in enumerate(TICKERS):
        name   = NAMES[ticker]
        color  = COLORS[ticker]

        # ── 価格データ取得
        price_df = price_map[ticker]
        p        = price_df["close"]
        dates    = pd.to_datetime(price_df.index)

        # ── テクニカル指標
        ind = calc_indicators(p)

        # ── 上昇確率（全期間）
        proba_s, split_n = get_full_proba(
            ticker, fund_df, price_map, rf_model, feat_cols
        )
        pred_dates = proba_s.index
        proba_vals = proba_s.values

        # 予測シグナル（全期間）
        signals_up   = proba_vals >= 0.5
        signals_down = proba_vals <  0.5

        # 訓練/テスト 境界日
        split_date = pred_dates[split_n] if split_n < len(pred_dates) else pred_dates[-1]

        # ════════════════════════════════════════════════
        # 行1: 株価チャート + 移動平均 + ボリンジャー + シグナル
        # ════════════════════════════════════════════════
        ax_p = axes[0][col]

        # ボリンジャーバンド塗りつぶし
        ax_p.fill_between(
            dates, ind["bb_upper"], ind["bb_lower"],
            alpha=0.07, color=color, label="ボリンジャーバンド(±2σ)",
        )
        ax_p.plot(dates, ind["bb_upper"], lw=0.5, color=color, alpha=0.4, ls="--")
        ax_p.plot(dates, ind["bb_lower"], lw=0.5, color=color, alpha=0.4, ls="--")

        # 株価
        ax_p.plot(dates, p, lw=1.4, color=color, label="終値", zorder=3)

        # 移動平均
        ax_p.plot(dates, ind["ma5"],  lw=1.0, color="orange",    ls="-",  alpha=0.9, label="MA5")
        ax_p.plot(dates, ind["ma20"], lw=1.0, color="limegreen",  ls="-",  alpha=0.9, label="MA20")
        ax_p.plot(dates, ind["ma60"], lw=1.2, color="mediumpurple", ls="--", alpha=0.9, label="MA60")

        # 訓練/テスト 境界線
        ax_p.axvline(split_date, color="gray", lw=1.2, ls=":", alpha=0.8)
        ax_p.text(
            split_date, ind["close"].max() * 0.995,
            " ← 訓練 | テスト →",
            fontsize=7.5, color="gray", va="top",
        )

        # 予測シグナルのオーバーレイ（予測期間のみ）
        if len(pred_dates) == len(proba_vals):
            pred_prices = p.reindex(pred_dates, method="nearest")
            # 上昇シグナル
            ax_p.scatter(
                pred_dates[signals_up], pred_prices.values[signals_up],
                marker="^", color="crimson", s=18, zorder=5, alpha=0.7,
                label="↑上昇予測",
            )
            # 下落シグナル
            ax_p.scatter(
                pred_dates[signals_down], pred_prices.values[signals_down],
                marker="v", color="navy", s=12, zorder=4, alpha=0.4,
                label="↓下落予測",
            )

        # 最新株価テキスト
        ax_p.annotate(
            f"  ¥{p.iloc[-1]:,.0f}",
            xy=(dates[-1], p.iloc[-1]),
            fontsize=9, color=color, fontweight="bold", va="center",
        )

        ax_p.set_title(f"{ticker}  {name}", fontsize=11, color=color, fontweight="bold")
        ax_p.set_ylabel("株価 (円)")
        ax_p.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax_p.legend(fontsize=7, loc="upper left", ncol=3)
        ax_p.grid(True, alpha=0.25)
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
        ax_p.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_p.tick_params(axis="x", rotation=30, labelsize=8)

        # ════════════════════════════════════════════════
        # 行2: RSI
        # ════════════════════════════════════════════════
        ax_r = axes[1][col]
        ax_r.plot(dates, ind["rsi"], lw=1.2, color=color)
        ax_r.axhline(70, color="crimson", lw=0.8, ls="--", alpha=0.7)
        ax_r.axhline(30, color="navy",    lw=0.8, ls="--", alpha=0.7)
        ax_r.axhline(50, color="gray",    lw=0.6, ls=":",  alpha=0.5)
        ax_r.fill_between(dates, ind["rsi"], 70,
                           where=(ind["rsi"] >= 70),
                           alpha=0.25, color="crimson", label="過買い(>70)")
        ax_r.fill_between(dates, ind["rsi"], 30,
                           where=(ind["rsi"] <= 30),
                           alpha=0.25, color="navy",    label="過売り(<30)")
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks([0, 30, 50, 70, 100])
        ax_r.set_ylabel("RSI (14)")
        ax_r.legend(fontsize=7, loc="upper left")
        ax_r.grid(True, alpha=0.25)
        ax_r.axvline(split_date, color="gray", lw=1.0, ls=":", alpha=0.6)
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
        ax_r.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_r.tick_params(axis="x", rotation=30, labelsize=8)

        # 最新RSI値
        rsi_now = ind["rsi"].dropna().iloc[-1]
        rsi_color = "crimson" if rsi_now >= 70 else ("navy" if rsi_now <= 30 else "gray")
        ax_r.text(
            0.98, 0.85, f"RSI={rsi_now:.1f}",
            transform=ax_r.transAxes, ha="right", fontsize=9,
            color=rsi_color, fontweight="bold",
        )

        # ════════════════════════════════════════════════
        # 行3: MACD ヒストグラム
        # ════════════════════════════════════════════════
        ax_m = axes[2][col]
        hist = ind["macd_hist"]
        ax_m.bar(
            dates[hist >= 0], hist[hist >= 0],
            width=1.2, color="crimson", alpha=0.6, label="MACD > 0",
        )
        ax_m.bar(
            dates[hist <  0], hist[hist <  0],
            width=1.2, color="navy",    alpha=0.6, label="MACD < 0",
        )
        ax_m.plot(dates, ind["macd_line"], lw=1.0, color="darkorange",
                  label="MACDライン", alpha=0.9)
        ax_m.plot(dates, ind["macd_sig"],  lw=1.0, color="purple",
                  label="シグナル線",  alpha=0.9)
        ax_m.axhline(0, color="gray", lw=0.7)
        ax_m.set_ylabel("MACD")
        ax_m.legend(fontsize=7, loc="upper left", ncol=2)
        ax_m.grid(True, alpha=0.25)
        ax_m.axvline(split_date, color="gray", lw=1.0, ls=":", alpha=0.6)
        ax_m.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
        ax_m.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_m.tick_params(axis="x", rotation=30, labelsize=8)

        # 最新MACD値
        macd_now = hist.dropna().iloc[-1]
        ax_m.text(
            0.98, 0.85, f"MACD={macd_now:+.2f}",
            transform=ax_m.transAxes, ha="right", fontsize=9,
            color=("crimson" if macd_now >= 0 else "navy"), fontweight="bold",
        )

        # ════════════════════════════════════════════════
        # 行4: 5日後上昇確率の推移
        # ════════════════════════════════════════════════
        ax_q = axes[3][col]

        if len(proba_vals):
            # 訓練期間（薄く）
            train_mask = pred_dates < split_date
            test_mask  = pred_dates >= split_date

            if train_mask.any():
                ax_q.fill_between(
                    pred_dates[train_mask], proba_vals[train_mask], 0.5,
                    where=(proba_vals[train_mask] >= 0.5),
                    alpha=0.15, color="crimson",
                )
                ax_q.fill_between(
                    pred_dates[train_mask], proba_vals[train_mask], 0.5,
                    where=(proba_vals[train_mask] < 0.5),
                    alpha=0.15, color="navy",
                )
                ax_q.plot(pred_dates[train_mask], proba_vals[train_mask],
                          lw=0.8, color="gray", alpha=0.5, label="訓練期間")

            if test_mask.any():
                ax_q.fill_between(
                    pred_dates[test_mask], proba_vals[test_mask], 0.5,
                    where=(proba_vals[test_mask] >= 0.5),
                    alpha=0.4, color="crimson", label="上昇域（テスト）",
                )
                ax_q.fill_between(
                    pred_dates[test_mask], proba_vals[test_mask], 0.5,
                    where=(proba_vals[test_mask] < 0.5),
                    alpha=0.4, color="navy", label="下落域（テスト）",
                )
                ax_q.plot(pred_dates[test_mask], proba_vals[test_mask],
                          lw=1.2, color="black", alpha=0.75, label="テスト期間")

            # 最新確率
            latest_p = proba_vals[-1]
            ax_q.scatter(pred_dates[-1], latest_p,
                         s=80, zorder=6,
                         color="crimson" if latest_p >= 0.5 else "navy")
            ax_q.text(
                0.98, 0.88,
                f"最新: {latest_p:.1%}  {'↑上昇予測' if latest_p >= 0.5 else '↓下落予測'}",
                transform=ax_q.transAxes, ha="right", fontsize=9, fontweight="bold",
                color="crimson" if latest_p >= 0.5 else "navy",
            )

        ax_q.axhline(0.5, color="black", lw=1.2, ls="--")
        ax_q.axvline(split_date, color="gray", lw=1.0, ls=":", alpha=0.6)
        ax_q.set_ylim(0, 1)
        ax_q.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_q.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_q.set_ylabel(f"上昇確率\n({HORIZON_5D}日後)")
        ax_q.set_xlabel("日付")
        ax_q.legend(fontsize=7, loc="upper left", ncol=2)
        ax_q.grid(True, alpha=0.25)
        ax_q.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
        ax_q.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_q.tick_params(axis="x", rotation=30, labelsize=8)

    # ── 共通凡例（縦境界線の説明）
    fig.text(
        0.5, -0.005,
        "点線（縦）: 訓練データ / テストデータの境界  |  "
        "▲赤: 上昇予測  ▼紺: 下落予測  |  "
        f"予測ホライズン: {HORIZON_5D}営業日後 / 閾値: {THRESHOLD_5D*100:.0f}%超",
        ha="center", fontsize=8, color="gray",
    )

    fig.tight_layout(rect=[0, 0.01, 1, 1])
    out = OUTPUT_DIR / "detail_3076_5970.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[出力] 詳細グラフ → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 補足グラフ: 2銘柄の株価推移と上昇確率を重ねた比較グラフ
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(fund_df, price_map, results, feat_cols):
    """2銘柄の株価変化率と上昇確率を1枚に重ねて比較する。"""
    rf_model = results["RandomForest"]["model"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.suptitle(
        "3076.T vs 5970.T  —  株価変化率 & 5日後上昇確率 の比較",
        fontsize=13,
    )

    # 基準日からの変化率を計算
    for ticker in TICKERS:
        price_df = price_map[ticker]
        p        = price_df["close"]
        dates    = pd.to_datetime(price_df.index)
        ret      = (p / p.iloc[0] - 1) * 100   # 累積リターン（%）
        color    = COLORS[ticker]
        name     = NAMES[ticker]

        ax1.plot(dates, ret, lw=1.6, color=color,
                 label=f"{ticker}  {name[:18]}")

    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("基準日比リターン (%)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.tick_params(axis="x", rotation=30, labelsize=8)

    # 上昇確率の推移（テスト期間のみ）
    for ticker in TICKERS:
        proba_s, split_n = get_full_proba(
            ticker, fund_df, price_map, rf_model, feat_cols
        )
        if proba_s.empty:
            continue

        pred_dates = proba_s.index
        proba_vals = proba_s.values
        split_date = pred_dates[split_n] if split_n < len(pred_dates) else pred_dates[-1]

        # テスト期間のみプロット
        test_mask = pred_dates >= split_date
        color     = COLORS[ticker]
        name      = NAMES[ticker]

        ax2.plot(pred_dates[test_mask], proba_vals[test_mask],
                 lw=1.6, color=color,
                 label=f"{ticker}  {name[:18]}")

        # 最新確率をマーク
        ax2.scatter(pred_dates[-1], proba_vals[-1],
                    s=80, color=color, zorder=5)

    ax2.axhline(0.5, color="black", lw=1.2, ls="--")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_ylabel(f"上昇確率 ({HORIZON_5D}日後)")
    ax2.set_xlabel("日付（テスト期間のみ）")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.tick_params(axis="x", rotation=30, labelsize=8)

    fig.tight_layout()
    out = OUTPUT_DIR / "comparison_3076_5970.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[出力] 比較グラフ   → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("3076.T / 5970.T  詳細グラフ生成")
    print(f"  予測ホライズン: {HORIZON_5D} 営業日後")
    print("=" * 65)

    # データ読み込み（キャッシュ使用）
    print("\n[Step 1] データ読み込み（Parquet キャッシュ）")
    fund_df   = ps.load_fundamentals()
    tickers   = fund_df["ティッカー"].tolist()

    old_h, old_t   = ps.HORIZON, ps.THRESHOLD
    ps.HORIZON     = HORIZON_5D
    ps.THRESHOLD   = THRESHOLD_5D
    price_map      = ps.download_prices_batch(tickers)
    ps.HORIZON, ps.THRESHOLD = old_h, old_t
    print(f"  {len(price_map)} 銘柄のキャッシュ読み込み完了")

    # 対象銘柄の確認
    for t in TICKERS:
        if t not in price_map:
            print(f"  [エラー] {t} のデータが見つかりません")
            return
        df = price_map[t]
        print(f"  {t}: {len(df)} 行  "
              f"{df.index[0].date()} ～ {df.index[-1].date()}  "
              f"最新終値 ¥{df['close'].iloc[-1]:,.0f}")

    # モデル学習
    print("\n[Step 2] モデル学習中（全1025銘柄 × HORIZON=5日）...")
    results, feat_cols = train_model(fund_df, price_map)

    # グラフ生成
    print("\n[Step 3] 詳細グラフ生成中...")
    plot_detail(fund_df, price_map, results, feat_cols)

    print("\n[Step 4] 比較グラフ生成中...")
    plot_comparison(fund_df, price_map, results, feat_cols)

    print(f"\n{'='*65}")
    print("【完了】")
    for f in sorted(OUTPUT_DIR.glob("*3076*")) + sorted(OUTPUT_DIR.glob("*5970*")):
        print(f"  {f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
