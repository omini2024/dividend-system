# ==============================
# 【売買タイミング判定】Final7変更検出 & テクニカル分析モジュール
# monthly_monitor.py から import して使用する
#
# final7_prev.json 管理ルール:
#   毎月の月次実行末尾で final7.json の内容を final7_prev.json に上書き保存する。
#   これにより「年次選定直後の月次実行」でのみ差分が検出される。
#   （年次選定で Final7 が更新 → 次の月次で prev≠current → タイミング分析実行
#     → prev を current で更新 → 翌月以降は変更なし）
#
# 使用テクニカル指標:
#   MA25 / MA75 / MA200（単純移動平均）
#   RSI14（相対力指数）
#   MA25/MA75 クロス状態
#
# 判定記号:
#   ◎好機  ○検討  △待機  ─様子見
# ==============================

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

OUTPUT_DIR  = "output"
FINAL7_JSON = os.path.join(OUTPUT_DIR, "final7.json")
FINAL7_PREV = os.path.join(OUTPUT_DIR, "final7_prev.json")


# ============================================================
# テクニカル指標計算
# ============================================================

def _calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    rsi   = 100 - (100 / (1 + rs))
    valid = rsi.dropna()
    return round(float(valid.iloc[-1]), 1) if not valid.empty else 50.0


def _get_indicators(ticker: str) -> dict | None:
    """
    yfinance から直近1年の日足を取得し MA・RSI を計算して返す。
    データ不足や取得失敗時は None を返す。
    """
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < 30:
            return None
        close = hist["Close"]
        n     = len(close)
        price = float(close.iloc[-1])

        ma25  = float(close.rolling(25).mean().iloc[-1])  if n >= 25  else None
        ma75  = float(close.rolling(75).mean().iloc[-1])  if n >= 75  else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
        rsi   = _calc_rsi(close)

        # MA25/MA75 クロス状態（直近10日で判定）
        cross = "データ不足(MA75算出に日数不足)"
        if n >= 75:
            s25 = close.rolling(25).mean().iloc[-10:]
            s75 = close.rolling(75).mean().iloc[-10:]
            if s25.iloc[-1] > s75.iloc[-1] and s25.iloc[0] <= s75.iloc[0]:
                cross = "★ゴールデンクロス（直近10日）"
            elif s25.iloc[-1] < s75.iloc[-1] and s25.iloc[0] >= s75.iloc[0]:
                cross = "★デッドクロス（直近10日）"
            elif s25.iloc[-1] > s75.iloc[-1]:
                cross = "MA25>MA75（上昇傾向）"
            else:
                cross = "MA25<MA75（下降傾向）"

        return {
            "price": round(price, 0),
            "ma25":  round(ma25,  0) if ma25  is not None else None,
            "ma75":  round(ma75,  0) if ma75  is not None else None,
            "ma200": round(ma200, 0) if ma200 is not None else None,
            "rsi":   rsi,
            "cross": cross,
        }
    except Exception:
        return None


# ============================================================
# 売り/買いタイミング判定
# ============================================================

def _judge_sell(ind: dict) -> tuple[str, str]:
    """除外銘柄（売り候補）のタイミング判定"""
    rsi   = ind["rsi"]
    price = ind["price"]
    ma25  = ind["ma25"]
    ma200 = ind["ma200"]
    cross = ind["cross"]
    notes = []

    if rsi >= 70:
        verdict = "◎売り好機"
        notes.append(f"RSI={rsi}（買われすぎ水準・高値圏）")
    elif rsi >= 55:
        verdict = "○売り検討"
        notes.append(f"RSI={rsi}（やや高水準・上昇モメンタム）")
    elif rsi <= 35:
        verdict = "△反発待ち"
        notes.append(f"RSI={rsi}（売られすぎ水準・反発後または損切り検討）")
    else:
        verdict = "─様子見"
        notes.append(f"RSI={rsi}（中立水準）")

    if ma25 is not None:
        pos = "MA上位圏" if price > ma25 else "MA下位圏（戻り待ち）"
        notes.append(f"株価{price:.0f} vs MA25({ma25:.0f}) → {pos}")

    if ma200 is not None:
        if price < ma200:
            notes.append(f"⚠️株価{price:.0f}<MA200({ma200:.0f})（長期下降トレンド）")

    if "デッドクロス" in cross:
        notes.append(f"⚠️{cross}")
    elif "ゴールデンクロス" in cross:
        notes.append(f"✅{cross}")
    else:
        notes.append(cross)

    return verdict, " / ".join(notes)


def _judge_buy(ind: dict) -> tuple[str, str]:
    """新規追加銘柄（買い候補）のタイミング判定"""
    rsi   = ind["rsi"]
    price = ind["price"]
    ma25  = ind["ma25"]
    ma200 = ind["ma200"]
    cross = ind["cross"]
    notes = []

    if rsi <= 35:
        verdict = "◎買い好機"
        notes.append(f"RSI={rsi}（売られすぎ水準・押し目）")
    elif rsi <= 50:
        verdict = "○買い検討"
        notes.append(f"RSI={rsi}（中立〜やや低水準）")
    elif rsi >= 70:
        verdict = "△押し目待ち"
        notes.append(f"RSI={rsi}（買われすぎ水準・高値追い回避）")
    else:
        verdict = "─様子見"
        notes.append(f"RSI={rsi}（中立水準）")

    if ma200 is not None:
        if price > ma200:
            notes.append(f"✅株価{price:.0f}>MA200({ma200:.0f})（長期上昇トレンド）")
        else:
            notes.append(f"⚠️株価{price:.0f}<MA200({ma200:.0f})（長期下降トレンド・慎重に）")

    if ma25 is not None:
        pos = "押し目圏（MA付近 or 以下）" if price <= ma25 * 1.02 else "短期上位圏"
        notes.append(f"株価{price:.0f} vs MA25({ma25:.0f}) → {pos}")

    if "ゴールデンクロス" in cross:
        notes.append(f"✅{cross}")
    elif "デッドクロス" in cross:
        notes.append(f"⚠️{cross}")
    else:
        notes.append(cross)

    return verdict, " / ".join(notes)


# ============================================================
# 連続異常検知銘柄の売りタイミング分析（monthly_monitor から呼び出す）
# exclusion_candidate が確定した月に即時実行する
# ============================================================

def analyze_exclusion_timing(tickers: list, details: dict) -> str:
    """
    2回連続異常検知で exclusion_candidate となった銘柄の売りタイミングを分析する。
    年次選定の変更を待たずに、フラグ確定と同月に売り判断を提供する。

    Args:
        tickers : exclusion_candidate となったティッカーのリスト
        details : final7.json の details を {ティッカー: dict} に変換したもの

    Returns:
        report_text (str): メール本文に追記するレポート文字列
    """
    if not tickers:
        return ""

    today_str = datetime.today().strftime("%Y年%m月%d日")
    lines = [
        "",
        "=" * 60,
        f"【緊急】売りタイミング判定 — 連続異常検知銘柄 ({today_str})",
        "  2ヶ月連続で統計的異常が検出されました。",
        "  年次選定を待たず、売り時期の検討を推奨します。",
        "=" * 60,
    ]

    for t in tickers:
        d    = details.get(t, {})
        name = d.get("会社名", t)
        lines.append("")
        lines.append(f"  [{t}] {name}  ← 次回年次選定で除外候補")
        lines.append(
            f"    選定時: 利回り={d.get('利回り%(3年平均)', '-')}% / "
            f"総合点={d.get('総合点', '-')} / DOE={d.get('DOE%', '-')}%"
        )
        excl_reason = d.get("exclusion_reason", "")
        if excl_reason:
            lines.append(f"    異常検知: {excl_reason}")

        ind = _get_indicators(t)
        if ind:
            verdict, reason = _judge_sell(ind)
            lines.append(f"    テクニカル売り判定: {verdict}")
            lines.append(f"    根拠: {reason}")
        else:
            lines.append("    ⚠️ テクニカルデータ取得失敗")

    lines += [
        "",
        "─" * 40,
        "【判定基準（売り）】",
        "  ◎売り好機 : RSI≥70（買われすぎ水準・高値圏）",
        "  ○売り検討 : RSI≥55（やや高水準）",
        "  △反発待ち : RSI≤35（売られすぎ・損切りか反発後に売り）",
        "  ─様子見   : RSI中立（急がず観察）",
        "─" * 40,
    ]

    return "\n".join(lines)


# ============================================================
# メイン: タイミング分析（monthly_monitor から呼び出す）
# ============================================================

def analyze_trade_timing() -> tuple[bool, str]:
    """
    Final7 の変更を検出し、売り/買いタイミング分析を実施する。

    Returns:
        has_changes (bool): 変更があった場合 True
        report_text (str):  メール本文に追記するレポート文字列

    副作用:
        実行後に final7.json の内容で final7_prev.json を上書き保存する。
    """
    if not os.path.exists(FINAL7_JSON):
        return False, "⚠️ 売買タイミング判定: final7.json が見つかりません。"

    with open(FINAL7_JSON, "r", encoding="utf-8") as f:
        current = json.load(f)

    # 初回実行: ベースライン保存のみ
    if not os.path.exists(FINAL7_PREV):
        with open(FINAL7_PREV, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
        return False, (
            "ℹ️ 売買タイミング判定: 初回実行のためベースライン(final7_prev.json)を保存しました。\n"
            "   次回年次選定後の月次実行から変更検出が有効になります。"
        )

    with open(FINAL7_PREV, "r", encoding="utf-8") as f:
        prev = json.load(f)

    prev_tickers    = set(prev.get("tickers", []))
    current_tickers = set(current.get("tickers", []))
    removed = prev_tickers - current_tickers   # 除外銘柄 → 売り検討
    added   = current_tickers - prev_tickers   # 新規銘柄 → 買い検討

    # final7_prev.json を今回の内容で更新（次回月次では変更なし判定になる）
    with open(FINAL7_PREV, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)

    if not removed and not added:
        return False, "ℹ️ 売買タイミング判定: Final7に変更なし。"

    prev_details    = {d["ティッカー"]: d for d in prev.get("details",    [])}
    current_details = {d["ティッカー"]: d for d in current.get("details", [])}
    today_str = datetime.today().strftime("%Y年%m月%d日")

    lines = [
        "",
        "=" * 60,
        f"【売買タイミング判定】{today_str}",
        f"前回選定: {prev.get('selected_date', '不明')}  →  今回選定: {current.get('selected_date', '不明')}",
        f"除外（売り候補）: {len(removed)} 社  /  新規（買い候補）: {len(added)} 社",
        "=" * 60,
    ]

    # ── 売り候補 ──────────────────────────────
    if removed:
        lines.append("")
        lines.append("■ 売り候補（Final7から除外された銘柄）")
        lines.append("  ファンダメンタルの悪化が検出されました。売り時期を検討してください。")
        for t in sorted(removed):
            d    = prev_details.get(t, {})
            name = d.get("会社名", t)
            lines.append("")
            lines.append(f"  [{t}] {name}")
            lines.append(
                f"    選定時: 利回り={d.get('利回り%(3年平均)', '-')}% / "
                f"総合点={d.get('総合点', '-')} / DOE={d.get('DOE%', '-')}%"
            )
            ind = _get_indicators(t)
            if ind:
                verdict, reason = _judge_sell(ind)
                lines.append(f"    タイミング判定: {verdict}")
                lines.append(f"    根拠: {reason}")
            else:
                lines.append("    ⚠️ テクニカルデータ取得失敗（上場廃止またはネット障害）")

    # ── 買い候補 ──────────────────────────────
    if added:
        lines.append("")
        lines.append("■ 買い候補（Final7に新規追加された銘柄）")
        lines.append("  ファンダメンタルが高評価されました。買い時期を検討してください。")
        for t in sorted(added):
            d    = current_details.get(t, {})
            name = d.get("会社名", t)
            lines.append("")
            lines.append(f"  [{t}] {name}")
            lines.append(
                f"    今回: 利回り={d.get('利回り%(3年平均)', '-')}% / "
                f"総合点={d.get('総合点', '-')} / DOE={d.get('DOE%', '-')}%"
            )
            ind = _get_indicators(t)
            if ind:
                verdict, reason = _judge_buy(ind)
                lines.append(f"    タイミング判定: {verdict}")
                lines.append(f"    根拠: {reason}")
            else:
                lines.append("    ⚠️ テクニカルデータ取得失敗（上場廃止またはネット障害）")

    lines += [
        "",
        "─" * 40,
        "【判定基準】",
        "  ◎好機 : RSIが極端水準（売り≥70 / 買い≤35）",
        "  ○検討 : RSIが方向性を示す水準",
        "  △待機 : RSIが逆方向の水準（タイミングを待つ）",
        "  ─様子見: RSI中立水準（急がず観察）",
        "  MA200上位 = 長期上昇トレンド確認（買いの安心感）",
        "  MA25/75クロス = 短中期トレンド転換のサイン",
        "─" * 40,
    ]

    return True, "\n".join(lines)
