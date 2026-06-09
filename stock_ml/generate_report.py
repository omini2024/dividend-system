#!/usr/bin/env python3
"""
generate_report.py — 株価予測手法に関する調査・検討報告書の生成
"""

import matplotlib
matplotlib.use("Agg")
import chart_style  # noqa: F401

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from datetime import date

OUTPUT_PATH = Path(__file__).parent.parent / "output" / "stock_forecast_report.pdf"

A4 = (8.27, 11.69)   # A4インチ
MARGIN = 0.08         # ページ余白（figure座標）

C_NAVY   = "#1a3a5c"
C_BLUE   = "#2563a8"
C_LIGHT  = "#e8f0fb"
C_GRAY   = "#6b7280"
C_RED    = "#c0392b"
C_GREEN  = "#1a7a4a"
C_GOLD   = "#b8860b"
C_WHITE  = "#ffffff"
C_LINE   = "#d1d5db"


def draw_header(ax, title, subtitle=None, page_num=None, total=None):
    """各ページ共通ヘッダー"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # ヘッダーバー
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.72), 1.0, 0.28, boxstyle="square,pad=0",
        facecolor=C_NAVY, edgecolor="none", transform=ax.transAxes, clip_on=False))
    ax.text(0.03, 0.92, title, transform=ax.transAxes,
            fontsize=15, color=C_WHITE, fontweight="bold", va="top")
    if subtitle:
        ax.text(0.03, 0.78, subtitle, transform=ax.transAxes,
                fontsize=9, color="#a0b8d8", va="top")
    if page_num and total:
        ax.text(0.97, 0.78, f"{page_num} / {total}", transform=ax.transAxes,
                fontsize=8, color="#a0b8d8", va="top", ha="right")
    # 細線
    ax.axhline(0.70, color=C_BLUE, lw=2, xmin=0, xmax=1)


def hline(ax, y, **kw):
    ax.axhline(y, color=C_LINE, lw=0.8, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# ページ1: 表紙
# ─────────────────────────────────────────────────────────────────────────────

def page_cover(pdf):
    fig = plt.figure(figsize=A4)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 背景上部帯
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.55), 1.0, 0.45, boxstyle="square,pad=0",
        facecolor=C_NAVY, edgecolor="none"))

    # タイトル
    ax.text(0.5, 0.90, "株価予測手法に関する", fontsize=26,
            color=C_WHITE, fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.82, "調 査・検 討 報 告 書", fontsize=28,
            color=C_WHITE, fontweight="bold", ha="center", va="center")

    # サブタイトル
    ax.text(0.5, 0.73,
            "― 統計手法の限界とオルタナティブアプローチの可能性 ―",
            fontsize=11, color="#a0b8d8", ha="center", va="center")

    # 区切りライン
    ax.plot([0.15, 0.85], [0.68, 0.68], color=C_GOLD, lw=2)

    # 日付・作成情報
    today = date.today().strftime("%Y年%m月%d日")
    ax.text(0.5, 0.62, f"作成日：{today}", fontsize=11,
            color="#a0b8d8", ha="center", va="center")

    # 下部コンテンツ概要ボックス
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, 0.12), 0.84, 0.40,
        boxstyle="round,pad=0.01",
        facecolor=C_LIGHT, edgecolor=C_BLUE, lw=1.2))

    ax.text(0.5, 0.49, "本報告書の構成", fontsize=12,
            color=C_NAVY, fontweight="bold", ha="center", va="center")
    ax.plot([0.15, 0.85], [0.47, 0.47], color=C_BLUE, lw=0.8)

    items = [
        ("第1章", "株価予測の本質的な難しさ"),
        ("第2章", "現行予測システムの評価結果"),
        ("第3章", "主要統計手法の比較と限界"),
        ("第4章", "オルタナティブアプローチの可能性"),
        ("第5章", "総合考察と実用的な活用指針"),
    ]
    for i, (num, text) in enumerate(items):
        y = 0.43 - i * 0.056
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.12, y - 0.018), 0.10, 0.032,
            boxstyle="round,pad=0.002",
            facecolor=C_BLUE, edgecolor="none"))
        ax.text(0.17, y - 0.002, num, fontsize=8.5,
                color=C_WHITE, ha="center", va="center", fontweight="bold")
        ax.text(0.25, y - 0.002, text, fontsize=9.5,
                color=C_NAVY, va="center")

    # フッター
    ax.text(0.5, 0.05, "本報告書は自動株価予測システムの開発・検討過程における知見をまとめたものです",
            fontsize=8, color=C_GRAY, ha="center", va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ページ2: 第1章 株価予測の本質的な難しさ
# ─────────────────────────────────────────────────────────────────────────────

def page_ch1(pdf):
    fig = plt.figure(figsize=A4)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.06)

    # ヘッダー用サブプロット
    ax_h = fig.add_axes([0, 0.88, 1, 0.12])
    draw_header(ax_h,
                "第1章　株価予測の本質的な難しさ",
                "なぜ統計モデルだけでは限界があるのか", 2, 6)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 1-1 効率的市場仮説 ─────────────────────────────────────
    ax.text(0.0, 0.97, "1-1　効率的市場仮説（EMH）が示す根本的限界",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.943)

    body1 = (
        "金融市場では、現時点で入手可能なあらゆる情報はほぼ瞬時に株価へ反映されるとされています（効率的\n"
        "市場仮説：Efficient Market Hypothesis）。この仮説が正しければ、公開情報のみを用いた予測モデルで\n"
        "継続的に市場平均を上回ることは原理的に不可能です。\n\n"
        "ただし、現実の市場は「完全には効率的でない」ことも確認されており、短期的・局所的なアノマリー（規\n"
        "則的な歪み）が存在します。これが定量的手法に「わずかな余地」を与えています。"
    )
    ax.text(0.02, 0.925, body1, fontsize=9.5, color="#222222",
            va="top", linespacing=1.7)

    # EMH 図解
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.72), 0.96, 0.10,
        boxstyle="round,pad=0.008",
        facecolor="#fff8e1", edgecolor=C_GOLD, lw=1.2))
    ax.text(0.5, 0.775,
            "市場参加者全員が同じ情報で予測 → 「予測できる情報」は即座に価格へ織り込まれる"
            " → 次の瞬間の価格変動はランダム",
            fontsize=9, color=C_NAVY, ha="center", va="center",
            style="italic")

    # ── 1-2 ランダムウォークとの対比 ─────────────────────────
    ax.text(0.0, 0.700, "1-2　ランダムウォーク仮説との対比",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.668)

    body2 = (
        "株価変動がランダムウォーク（コインを投げるように完全にランダム）であるならば、予測の AUC（受信者\n"
        "動作特性）は 0.50 となります。現実には市場の非効率性により、テクニカル・ファンダメンタルズ分析\n"
        "を組み合わせると AUC 0.55〜0.65 程度が得られることがあります。"
    )
    ax.text(0.02, 0.653, body2, fontsize=9.5, color="#222222",
            va="top", linespacing=1.7)

    # AUC帯グラフ（横棒）
    labels = ["完全ランダム", "単純移動平均のみ", "機械学習（本システム）",
              "ヘッジファンドの現実的上限", "完全予測（理論値）"]
    values = [0.500, 0.520, 0.585, 0.650, 1.000]
    colors_bar = [C_GRAY, "#aab4be", C_BLUE, C_GREEN, "#9b59b6"]

    y_base = 0.38
    bar_h  = 0.040
    gap    = 0.055
    ax.text(0.02, y_base + len(labels) * gap + 0.02,
            "予測精度（AUC）の目安",
            fontsize=10, color=C_NAVY, fontweight="bold", va="bottom")

    for i, (lbl, val, col) in enumerate(zip(labels, values, colors_bar)):
        y = y_base + i * gap
        # 背景バー（最大幅）
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.30, y), 0.65, bar_h - 0.005,
            boxstyle="square,pad=0",
            facecolor="#e9ecef", edgecolor="none"))
        # 実測バー（0.5〜1.0の範囲を0.30〜0.95にマッピング）
        bar_w = (val - 0.45) / 0.55 * 0.65
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.30, y), max(bar_w, 0.005), bar_h - 0.005,
            boxstyle="square,pad=0",
            facecolor=col, edgecolor="none", alpha=0.85))
        ax.text(0.28, y + bar_h / 2 - 0.002, lbl,
                fontsize=8, color="#333333", ha="right", va="center")
        ax.text(0.31 + max(bar_w, 0.005), y + bar_h / 2 - 0.002,
                f"  AUC = {val:.3f}",
                fontsize=8, color="#333333", va="center")

    # 0.5ラインの縦線
    ax.plot([0.30, 0.30], [y_base - 0.01, y_base + len(labels) * gap + 0.01],
            color=C_RED, lw=1.2, ls="--", alpha=0.6)
    ax.text(0.30, y_base - 0.018, "0.50\n(ランダム)", fontsize=7.5,
            color=C_RED, ha="center", va="top")

    # ── 1-3 Min/Max予測の検証結果 ─────────────────────────────
    ax.text(0.0, 0.330, "1-3　Min / Max 絶対値予測の検証結果（本システム実測）",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.298)

    # 表
    table_data = [
        ["指標",               "Max 変化率予測",   "Min 変化率予測"],
        ["MAE（平均絶対誤差）", "2.30 %",          "2.00 %"],
        ["RMSE",               "3.61 %",          "3.00 %"],
        ["MAPE",               "297 %",            "245 %"],
        ["符号一致率（方向性）", "78.2 %",          "71.2 %"],
        ["実際の平均変動幅",   "±1.6〜2.3 %",    "±1.3〜3.0 %"],
    ]
    col_x = [0.03, 0.38, 0.68]
    row_y = [0.285, 0.252, 0.222, 0.192, 0.162, 0.132]
    col_w = [0.34, 0.29, 0.27]

    # ヘッダー行
    for j, (cx, cw, txt) in enumerate(zip(col_x, col_w, table_data[0])):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx - 0.005, row_y[0] - 0.008), cw, 0.032,
            boxstyle="square,pad=0", facecolor=C_NAVY, edgecolor="none"))
        ax.text(cx + cw / 2 - 0.005, row_y[0] + 0.006, txt,
                fontsize=8.5, color=C_WHITE, ha="center", va="center",
                fontweight="bold")

    for i, row in enumerate(table_data[1:], 1):
        bg = C_LIGHT if i % 2 == 0 else C_WHITE
        for j, (cx, cw, txt) in enumerate(zip(col_x, col_w, row)):
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx - 0.005, row_y[i] - 0.008), cw, 0.030,
                boxstyle="square,pad=0", facecolor=bg, edgecolor=C_LINE, lw=0.5))
            fc = C_RED if txt in ("297 %", "245 %") else (C_GREEN if "78" in txt or "71" in txt else "#222222")
            ax.text(cx + cw / 2 - 0.005, row_y[i] + 0.005, txt,
                    fontsize=8.5, color=fc, ha="center", va="center")

    # 注釈
    ax.text(0.02, 0.095,
            "【解釈】MAPE（平均絶対パーセント誤差）が 200〜300% であることは、予測誤差が予測対象値の 2〜3 倍に\n"
            "　　　　達することを意味します。絶対価格水準の予測は現実的ではなく、「方向性（符号一致率 70〜78%）」\n"
            "　　　　と「レンジ幅の傾向把握」に限定して活用するのが適切です。",
            fontsize=8.5, color=C_NAVY, va="top", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8e1",
                      edgecolor=C_GOLD, lw=1.0))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ページ3: 第2章 主要統計手法の比較
# ─────────────────────────────────────────────────────────────────────────────

def page_ch2(pdf):
    fig = plt.figure(figsize=A4)
    ax_h = fig.add_axes([0, 0.88, 1, 0.12])
    draw_header(ax_h,
                "第2章　主要統計手法の比較と限界",
                "現存する代表的な量的手法の特性・適用範囲・現実的な精度", 3, 6)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 大きな比較表
    headers = ["手法", "主な用途", "AUC目安", "強み", "限界"]
    col_x   = [0.00, 0.17, 0.36, 0.47, 0.72]
    col_w   = [0.17, 0.19, 0.11, 0.25, 0.28]

    rows = [
        ["ARIMA\nSARIMA",    "短期トレンド\n予測",     "0.50〜0.52", "解釈が容易\n季節性対応",      "非線形関係\n対応不可"],
        ["GARCH",            "ボラティリティ\n予測",   "高い※",      "リスク管理に\n最適",           "価格方向の\n予測は不可"],
        ["ロジスティック\n回帰", "分類（基準線）",    "0.52〜0.56", "解釈しやすい\n過学習が少ない", "非線形パターン\n捕捉が弱い"],
        ["ランダム\nフォレスト", "分類・回帰",        "0.55〜0.65", "特徴量重要度\n過学習に強い",   "未来の構造変化\nに弱い"],
        ["XGBoost\nLightGBM",  "分類・回帰",        "0.57〜0.65", "高精度・高速\nKaggle定番",     "ハイパーパラメータ\n調整が複雑"],
        ["LSTM\n（深層学習）", "時系列\nパターン学習", "0.55〜0.62", "長期依存関係\nの学習が可能",  "大量データ必要\n過学習リスク大"],
        ["Transformer\n系",    "注意機構による\n長距離依存", "0.57〜0.63", "文脈理解が\n強力",    "計算コスト大\n解釈困難"],
        ["ベイズ構造\n時系列", "トレンド分解\n予測区間", "0.53〜0.58", "不確実性の\n定量化が強み", "計算コスト高\n実装が複雑"],
        ["アンサンブル\n（スタッキング）", "複数モデル統合", "0.60〜0.68", "安定した精度\n汎化性能高い", "実装コスト高\n解釈困難"],
    ]

    row_h  = 0.074
    y_top  = 0.96

    # ヘッダー
    for cx, cw, h in zip(col_x, col_w, headers):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx, y_top - 0.028), cw - 0.004, 0.030,
            boxstyle="square,pad=0", facecolor=C_NAVY, edgecolor="none"))
        ax.text(cx + (cw - 0.004) / 2, y_top - 0.013, h,
                fontsize=8.5, color=C_WHITE, ha="center", va="center",
                fontweight="bold")

    for i, row in enumerate(rows):
        y = y_top - 0.030 - (i + 1) * row_h
        bg = C_LIGHT if i % 2 == 0 else C_WHITE
        is_current = (i == 3)  # ランダムフォレスト（現在実装）

        for j, (cx, cw, txt) in enumerate(zip(col_x, col_w, row)):
            ec = C_BLUE if is_current else C_LINE
            lw = 1.5 if is_current else 0.5
            ax.add_patch(mpatches.FancyBboxPatch(
                (cx, y), cw - 0.004, row_h - 0.006,
                boxstyle="square,pad=0",
                facecolor="#e8f4fd" if is_current else bg,
                edgecolor=ec, lw=lw))
            # AUC列は色付け
            if j == 2:
                auc_val = float(txt.split("〜")[0].replace("高い※", "0.7")) if txt != "高い※" else 0.7
                fc = C_GREEN if auc_val >= 0.60 else (C_BLUE if auc_val >= 0.55 else C_GRAY)
            else:
                fc = C_NAVY if is_current else "#333333"
            ax.text(cx + (cw - 0.004) / 2, y + row_h / 2 - 0.003, txt,
                    fontsize=7.8, color=fc, ha="center", va="center",
                    linespacing=1.4)

        if is_current:
            ax.text(0.97, y + row_h / 2 - 0.003, "← 現在実装",
                    fontsize=7.5, color=C_BLUE, ha="right", va="center",
                    fontweight="bold")

    # 注釈
    note_y = y_top - 0.030 - len(rows) * row_h - 0.02
    ax.text(0.0, note_y,
            "※ GARCHの「AUC高い」は価格方向ではなくボラティリティ予測に限定した評価です。",
            fontsize=8, color=C_GRAY, va="top")

    # 結論ボックス
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.01), 1.0, 0.12,
        boxstyle="round,pad=0.01",
        facecolor=C_LIGHT, edgecolor=C_BLUE, lw=1.2))
    ax.text(0.5, 0.125, "本章の結論", fontsize=10, color=C_NAVY,
            fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.095,
            "どの手法も「方向性予測の AUC 0.68 を安定的に超えた」という学術報告は極めて少ない。\n"
            "アンサンブル + オルタナティブデータの組み合わせが現実的な最高水準であるが、\n"
            "取得・実装コストと精度向上幅のトレードオフを十分に考慮する必要がある。",
            fontsize=9, color="#222222", ha="center", va="top", linespacing=1.7)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ページ4: 第3章 オルタナティブアプローチ（前半）
# ─────────────────────────────────────────────────────────────────────────────

def page_ch3a(pdf):
    fig = plt.figure(figsize=A4)
    ax_h = fig.add_axes([0, 0.88, 1, 0.12])
    draw_header(ax_h,
                "第3章　オルタナティブアプローチの可能性（前半）",
                "統計モデル以外の角度から市場を読む手法", 4, 6)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.0, 0.97,
            "統計的手法の限界を補うべく、機関投資家・ヘッジファンドは以下のような「非伝統的」アプローチを\n"
            "積極的に採用しています。これらは「いつ・いくらになるか」という点予測ではなく、「どの銘柄が\n"
            "相対的に有利か」を判断するための補助情報として活用されています。",
            fontsize=9.5, color="#222222", va="top", linespacing=1.7)

    sections = [
        {
            "title": "① オルタナティブデータ（非伝統的データ）",
            "color": C_NAVY,
            "items": [
                ("衛星画像データ",
                 "工場の稼働ランプ・駐車場の混雑・農作物の育成状況を衛星画像で定量化。\n"
                 "決算発表前に業績を高精度で推定。米大手ヘッジファンドが2010年代から実用化。\n"
                 "【信頼性】高 ／【コスト】非常に高い（年間数百万〜数千万円）"),
                ("クレジットカード取引データ",
                 "特定企業への消費者支出を月次でリアルタイム追跡。小売・外食・EC企業の\n"
                 "月次売上を決算前に高精度で予測可能。\n"
                 "【信頼性】高 ／【コスト】高い（データベンダー契約）"),
                ("求人情報・人材動向",
                 "Indeed・LinkedInの求人数増減で企業の拡大/縮小フェーズを先読み。\n"
                 "技術系採用が増加 → IT投資増 → 数四半期後の業績改善を示唆。\n"
                 "【信頼性】中〜高 ／【コスト】低〜中（スクレイピング or API）"),
                ("Googleトレンド・検索量",
                 "製品名・企業名の検索ボリュームは需要の先行指標。個人投資家の注目度\n"
                 "も反映。無料で取得可能だが、ノイズが多く解釈に経験が必要。\n"
                 "【信頼性】中 ／【コスト】無料"),
            ]
        },
        {
            "title": "② センチメント分析 / 自然言語処理（NLP）",
            "color": C_BLUE,
            "items": [
                ("決算説明会テキスト分析",
                 "経営者の発言語調（慎重→積極への変化）は株価に3〜6ヶ月先行することが\n"
                 "学術的に実証済み（Loughran & McDonald 2011等）。LLMを使えば低コストで実装可能。\n"
                 "【信頼性】中〜高 ／【コスト】低（LLM API使用）"),
                ("ニュースフロー分析",
                 "報道量の急増は大きな価格変動の先行指標（方向は不明）。ネガティブ報道と\n"
                 "ポジティブ報道の比率変化がシグナルとなる。\n"
                 "【信頼性】中 ／【コスト】低〜中"),
            ]
        },
    ]

    y = 0.76
    for sec in sections:
        # セクションタイトル
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.002), 1.0, 0.026,
            boxstyle="square,pad=0",
            facecolor=sec["color"], edgecolor="none"))
        ax.text(0.02, y + 0.010, sec["title"],
                fontsize=10.5, color=C_WHITE, fontweight="bold", va="center")
        y -= 0.015

        for item_title, item_body in sec["items"]:
            y -= 0.005
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - 0.060), 0.98, 0.062,
                boxstyle="round,pad=0.005",
                facecolor=C_LIGHT, edgecolor=C_LINE, lw=0.8))
            ax.text(0.03, y - 0.001, f"→ {item_title}",
                    fontsize=9, color=sec["color"], fontweight="bold", va="top")
            ax.text(0.03, y - 0.018, item_body,
                    fontsize=8.2, color="#333333", va="top", linespacing=1.55)
            y -= 0.075

        y -= 0.010

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ページ5: 第3章 オルタナティブアプローチ（後半）
# ─────────────────────────────────────────────────────────────────────────────

def page_ch3b(pdf):
    fig = plt.figure(figsize=A4)
    ax_h = fig.add_axes([0, 0.88, 1, 0.12])
    draw_header(ax_h,
                "第3章　オルタナティブアプローチの可能性（後半）",
                "イベントドリブン・オプション情報・産業構造分析", 5, 6)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sections = [
        {
            "title": "③ イベントドリブン戦略（事象起点の予測）",
            "color": C_GREEN,
            "items": [
                ("決算サプライズ後のドリフト",
                 "アナリスト予想を上回った決算発表の翌日から1〜3ヶ月間、超過リターンが\n"
                 "継続する傾向が多数の学術論文で実証されています（PEAD：Post-Earnings\n"
                 "Announcement Drift）。最も再現性の高いアノマリーの一つ。\n"
                 "【信頼性】高 ／【実装難度】中（決算予想データの取得が必要）"),
                ("指数採用・除外",
                 "日経225・TOPIXへの銘柄追加が発表されると、インデックスファンドの\n"
                 "買い需要が確実に発生し採用実施日まで買い圧力が継続する。\n"
                 "【信頼性】高 ／【実装難度】低（東証発表を監視するだけ）"),
                ("自社株買い・大量保有報告書",
                 "EDINETで無料取得可能。自社株買い発表後は需給改善効果があり、短期的に\n"
                 "株価を下支えする。大量保有報告書は買収・アクティビスト参入の先行シグナル。\n"
                 "【信頼性】中 ／【実装難度】低（EDINET API対応済み）"),
            ]
        },
        {
            "title": "④ オプション市場から「集合知」を読む",
            "color": "#8e44ad",
            "items": [
                ("インプライドボラティリティ（IV）",
                 "オプション価格に織り込まれた市場参加者全体の期待ボラティリティ。IV が\n"
                 "実績ボラティリティを大幅に上回る時は「何か大きな動きが近い」と市場が\n"
                 "警戒していることを示す。決算・政策発表前に頻発。\n"
                 "【信頼性】高 ／【コスト】中（オプションデータの購読）"),
                ("プットコールレシオ（PCR）",
                 "Put（売り）オプション需要 / Call（買い）オプション需要の比率。PCR が\n"
                 "極端に高い（1.5以上）場合は過剰悲観を示し、逆張りの買いシグナルになる\n"
                 "ことが多い。【信頼性】中〜高 ／【コスト】中"),
            ]
        },
        {
            "title": "⑤ サプライチェーン・産業連関分析",
            "color": C_GOLD,
            "items": [
                ("上流企業から下流企業を予測",
                 "半導体製造装置メーカーの受注増 → 3〜6ヶ月後に半導体メーカーの生産増\n"
                 "→ さらに後にスマホ・PC完成品メーカーの売上増、という構造的連鎖を活用。\n"
                 "日本では自動車部品メーカーの業績がメーカー本体の数四半期後を先行。\n"
                 "【信頼性】中〜高 ／【実装難度】中（産業連関の知識が必要）"),
            ]
        },
    ]

    y = 0.96
    for sec in sections:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.002), 1.0, 0.026,
            boxstyle="square,pad=0",
            facecolor=sec["color"], edgecolor="none"))
        ax.text(0.02, y + 0.010, sec["title"],
                fontsize=10.5, color=C_WHITE, fontweight="bold", va="center")
        y -= 0.015

        for item_title, item_body in sec["items"]:
            y -= 0.005
            lines = item_body.count("\n") + 1
            box_h = 0.018 + lines * 0.020
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.01, y - box_h - 0.004), 0.98, box_h + 0.008,
                boxstyle="round,pad=0.005",
                facecolor=C_LIGHT, edgecolor=C_LINE, lw=0.8))
            ax.text(0.03, y - 0.001, f"→ {item_title}",
                    fontsize=9, color=sec["color"], fontweight="bold", va="top")
            ax.text(0.03, y - 0.018, item_body,
                    fontsize=8.2, color="#333333", va="top", linespacing=1.55)
            y -= box_h + 0.018

        y -= 0.012

    # 信頼性マップ（簡易）
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.01), 1.0, 0.085,
        boxstyle="round,pad=0.008",
        facecolor="#f0f4f8", edgecolor=C_NAVY, lw=1.2))
    ax.text(0.5, 0.088, "手法別・信頼性と実装コストの総合評価",
            fontsize=9.5, color=C_NAVY, fontweight="bold", ha="center", va="top")

    items_eval = [
        ("決算サプライズ/指数採用",  "★★★★★", "低〜中"),
        ("オプションIV/PCR",        "★★★★☆", "中"),
        ("衛星/カード等オルタナ",   "★★★★☆", "非常に高"),
        ("センチメント/NLP",         "★★★☆☆", "低〜中"),
        ("産業連関分析",            "★★★☆☆", "中"),
    ]
    xe = [0.01, 0.38, 0.60, 0.80]
    ax.text(xe[0], 0.072, "手法", fontsize=7.5, color=C_NAVY, fontweight="bold")
    ax.text(xe[1], 0.072, "信頼性", fontsize=7.5, color=C_NAVY, fontweight="bold")
    ax.text(xe[2], 0.072, "コスト", fontsize=7.5, color=C_NAVY, fontweight="bold")

    for i, (name, stars, cost) in enumerate(items_eval):
        y_e = 0.060 - i * 0.012
        ax.text(xe[0], y_e, name,  fontsize=7.5, color="#222222")
        ax.text(xe[1], y_e, stars, fontsize=7.5, color=C_GOLD)
        ax.text(xe[2], y_e, cost,  fontsize=7.5, color="#222222")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# ページ6: 第4章 総合考察と実用的な活用指針
# ─────────────────────────────────────────────────────────────────────────────

def page_ch4(pdf):
    fig = plt.figure(figsize=A4)
    ax_h = fig.add_axes([0, 0.88, 1, 0.12])
    draw_header(ax_h,
                "第4章　総合考察と実用的な活用指針",
                "現行システムの位置付けと今後の拡張方向", 6, 6)

    ax = fig.add_axes([0.07, 0.06, 0.86, 0.80])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 4-1 何が予測しやすいか ────────────────────────────────
    ax.text(0.0, 0.97, "4-1　予測難易度スペクトラム",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.938)

    # スペクトラムバー
    spectrum = [
        ("ボラティリティ\n（GARCH）",   0.85, C_GREEN),
        ("相対ランキング\n（銘柄選別）", 0.68, "#3498db"),
        ("方向性\n（上/下）",           0.58, C_GOLD),
        ("価格水準\n（Min/Max）",       0.30, C_RED),
    ]
    ax.text(0.02, 0.920, "予測しやすい", fontsize=8, color=C_GREEN, va="top")
    ax.text(0.82, 0.920, "予測しにくい", fontsize=8, color=C_RED, va="top", ha="right")
    ax.annotate("", xy=(0.88, 0.905), xytext=(0.02, 0.905),
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2))

    for i, (lbl, score, col) in enumerate(spectrum):
        x = 0.03 + i * 0.235
        bar_fill = score
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.836), 0.20, 0.060,
            boxstyle="round,pad=0.005",
            facecolor=col, edgecolor="none", alpha=0.85))
        ax.text(x + 0.10, 0.866, lbl,
                fontsize=8, color=C_WHITE, ha="center", va="center",
                fontweight="bold", linespacing=1.4)
        ax.text(x + 0.10, 0.828, f"有効性スコア: {score:.0%}",
                fontsize=7.5, color=col, ha="center", va="top")

    # ── 4-2 現行システムの評価 ────────────────────────────────
    ax.text(0.0, 0.800, "4-2　現行システム（annual_select.py + predict_stock.py）の位置付け",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.768)

    body2 = (
        "現行システムは「ファンダメンタルズスコアリング × 機械学習分類」の組み合わせであり、\n"
        "1,025銘柄をスクリーニングする用途において、以下の点で統計的な有効性が認められます。"
    )
    ax.text(0.02, 0.753, body2, fontsize=9.5, color="#222222", va="top", linespacing=1.7)

    strengths = [
        ("特徴量重要度 No.1 が「総合点」",
         "annual_select.py が独自算出するファンダメンタルズスコアが機械学習モデルで\n最重要特徴量となっており、スコアリングの有効性が裏付けられています。"),
        ("1,025銘柄規模での AUC 0.585",
         "7銘柄のみの場合（AUC≈0.537）と比べ大幅に向上。サンプル数の確保が\n予測精度に直結することが実証されました。"),
        ("分類予測の方向性（符号一致率 78%）",
         "絶対価格水準の予測は困難ですが、「上がりやすい銘柄群」と「下がりやすい銘柄群」\nの傾向把握には十分な精度が確認されています。"),
    ]
    y = 0.700
    for title, body in strengths:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.068), 0.05, 0.058,
            boxstyle="square,pad=0", facecolor=C_BLUE, edgecolor="none"))
        ax.text(0.035, y - 0.039, "✓", fontsize=14, color=C_WHITE,
                ha="center", va="center")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.07, y - 0.068), 0.92, 0.058,
            boxstyle="round,pad=0.005",
            facecolor=C_LIGHT, edgecolor=C_LINE, lw=0.6))
        ax.text(0.09, y - 0.016, title, fontsize=9, color=C_NAVY,
                fontweight="bold", va="top")
        ax.text(0.09, y - 0.034, body, fontsize=8.2, color="#333333",
                va="top", linespacing=1.5)
        y -= 0.080

    # ── 4-3 推奨される拡張 ────────────────────────────────────
    ax.text(0.0, 0.455, "4-3　推奨される機能拡張（優先度順）",
            fontsize=12, color=C_NAVY, fontweight="bold", va="top")
    hline(ax, 0.423)

    extensions = [
        ("高", C_GREEN, "決算発表カレンダーとの連動",
         "決算前後にシグナル強度を調整。PEADアノマリーを意図的に活用できる。"),
        ("高", C_GREEN, "EDINET APIによる自社株買い・大量保有情報の取込",
         "無料かつ公開情報。需給イベントをシグナルに追加することで精度向上が期待。"),
        ("中", C_GOLD,  "決算説明会テキストのセンチメント分析",
         "LLM APIを使えば低コストで実装可能。定性情報の定量化。"),
        ("低", C_GRAY,  "オルタナティブデータ（衛星・カード等）の導入",
         "精度向上効果は期待できるが、データ取得コストが高く個人利用の費用対効果は低い。"),
    ]
    y = 0.408
    for priority, p_col, title, body in extensions:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.01, y - 0.045), 0.08, 0.038,
            boxstyle="round,pad=0.003",
            facecolor=p_col, edgecolor="none", alpha=0.85))
        ax.text(0.05, y - 0.026, f"優先度\n{priority}",
                fontsize=7.5, color=C_WHITE, ha="center", va="center",
                fontweight="bold", linespacing=1.3)
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.10, y - 0.045), 0.89, 0.038,
            boxstyle="round,pad=0.003",
            facecolor="#f8f9fa", edgecolor=C_LINE, lw=0.6))
        ax.text(0.12, y - 0.012, title, fontsize=9, color=C_NAVY,
                fontweight="bold", va="top")
        ax.text(0.12, y - 0.028, body, fontsize=8, color="#444444", va="top")
        y -= 0.055

    # ── 最終結論 ──────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.0, 0.01), 1.0, 0.118,
        boxstyle="round,pad=0.010",
        facecolor=C_NAVY, edgecolor="none"))
    ax.text(0.5, 0.122, "総合結論", fontsize=11, color=C_WHITE,
            fontweight="bold", ha="center", va="top")
    ax.text(0.5, 0.098,
            "株価の絶対値予測（精確な価格水準）は、市場の効率性と情報の非対称性により、\n"
            "どの統計手法・AIモデルを用いても現実的な精度での実現は困難です。\n"
            "一方、「相対的な銘柄選別」「方向性の傾向把握」「リスク管理のためのボラティリティ予測」\n"
            "においては、ファンダメンタルズ × 機械学習 × イベント情報の組み合わせが最も実用的です。\n"
            "本システムはその方向性において正しい設計思想を持っており、今後の拡張余地も十分にあります。",
            fontsize=9, color="#d4e6f1", ha="center", va="top", linespacing=1.75)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("報告書を生成しています...")
    with PdfPages(OUTPUT_PATH) as pdf:
        # PDFメタデータ
        from datetime import datetime
        d = pdf.infodict()
        d['Title']   = '株価予測手法に関する調査・検討報告書'
        d['Author']  = '株価予測システム'
        d['Subject'] = '統計手法の限界とオルタナティブアプローチの可能性'
        d['CreationDate'] = datetime.today()

        page_cover(pdf)
        page_ch1(pdf)
        page_ch2(pdf)
        page_ch3a(pdf)
        page_ch3b(pdf)
        page_ch4(pdf)

    print(f"完了: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
