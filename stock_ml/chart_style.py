#!/usr/bin/env python3
"""
chart_style.py — 全スクリプト共通の matplotlib スタイル設定
=============================================================
印刷時の紙歪みを防ぐため、濃い塗りつぶしを避けたシンプル設計。
各スクリプトの matplotlib.use("Agg") 直後に以下を記述する：

    import chart_style  # noqa: F401
"""

import matplotlib

matplotlib.rcParams.update({
    # フォント
    "font.family":         "Hiragino Sans",
    "axes.unicode_minus":  False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,

    # 背景・枠（塗りつぶしなし）
    "figure.facecolor":    "#ffffff",
    "axes.facecolor":      "#ffffff",
    "axes.edgecolor":      "#333333",
    "axes.linewidth":      0.8,

    # グリッド（薄い線のみ）
    "grid.color":          "#cccccc",
    "grid.linewidth":      0.5,
    "grid.alpha":          0.6,
    "axes.grid":           True,

    # タイトル・ラベル
    "axes.titlesize":      11,
    "axes.labelsize":      9,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,

    # 線・点
    "lines.linewidth":     1.2,
    "lines.markersize":    4,

    # 凡例
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "legend.fontsize":     8,

    # デフォルトカラーサイクル（薄色・印刷に映える）
    "axes.prop_cycle": matplotlib.cycler(color=[
        "#2563a8",  # 青
        "#c0392b",  # 赤
        "#27ae60",  # 緑
        "#7f8c8d",  # グレー
        "#8e44ad",  # 紫
        "#d35400",  # オレンジ
        "#16a085",  # ティール
        "#2c3e50",  # 濃紺
    ]),
})
