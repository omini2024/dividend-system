"""
sell_review.py
──────────────
長期安定配当株スクリーニング ── 売り時判定エンジン

設計方針（design memo 準拠）:
  - 自動売却は行わない。sell_candidate までは自動、sell_confirmed は人間確認後のみ。
  - 数値判定・状態遷移はルールベースで固定。
  - テキスト根拠の生成のみ Gemma3:12b（llm_reviewer.py）に委ねる。

状態:
  hold → watch → review_required → sell_candidate → sell_confirmed

実行方法:
    cd /Users/kagetatoshiyuki/dividend-system
    python3 sell_review.py
"""

import json
import logging
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

from llm_reviewer import generate_llm_evidence

try:
    from email_config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL
    MAIL_ENABLED = True
except ImportError:
    MAIL_ENABLED = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sell_review")

# ── パス ──────────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path(__file__).parent / "output"
FINAL7_JSON   = OUTPUT_DIR / "final7.json"
MONITOR_LOG   = OUTPUT_DIR / "monitor_log.xlsx"
OUTPUT_JSON   = OUTPUT_DIR / "sell_review_candidates.json"
OUTPUT_HTML   = OUTPUT_DIR / "sell_review_report.html"

# ── 判定閾値 ──────────────────────────────────────────────────────────────
DIV_CUT_THRESHOLD        = -20.0   # 配当変化率がこれ以下 → sell_candidate（A分類）
PAYOUT_WARN_THRESHOLD    = 70.0    # 配当性向がこれ以上 → review_required（B分類）
CONSECUTIVE_ANOMALY_MIN  = 2       # 連続異常がこれ以上 → review_required（B分類）
ANOMALY_SCORE_THRESHOLD  = -0.50   # monthly_anomaly_scoreがこれ以下を「異常」とみなす
DEBT_RATIO_WARN          = 30.0    # 負債比率がこれ以上 → watch（C分類）
HOLD_SCORE_RANK_WARN     = 5       # hold_score順位がこれ以下 → watch（D分類）

# ── hold_score ウェイト（設計メモ準拠）──────────────────────────────────
HOLD_SCORE_WEIGHTS = {
    "dividend_safety"      : 0.35,
    "financial_stability"  : 0.25,
    "earnings_quality"     : 0.20,
    "valuation"            : 0.10,
    "sector_relative"      : 0.10,
}

# 状態の優先順位（大きいほど深刻）
STATE_PRIORITY = {
    "hold"           : 0,
    "watch"          : 1,
    "review_required": 2,
    "sell_candidate" : 3,
    "sell_confirmed" : 4,
}


# ===========================================================================
# メイン
# ===========================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("sell_review 開始")
    logger.info("=" * 60)

    # ── データ読み込み ──────────────────────────────────────────────────
    final7   = _load_final7()
    mon_df   = _load_monitor_log()
    details  = final7["details"]
    logger.info(f"対象銘柄: {len(details)}件")

    # ── hold_score 計算 ─────────────────────────────────────────────────
    for det in details:
        det["hold_score"] = _calc_hold_score(det)
    scores = [d["hold_score"] for d in details]
    rank_map = _rank_map(scores)   # hold_score → 順位（1=最高）

    # ── ルールベース判定 ────────────────────────────────────────────────
    candidates = []
    for det in details:
        ticker  = det["ティッカー"]
        company = det["会社名"]
        sector  = det["業種"]

        # 月次履歴の取得
        rows       = mon_df[mon_df["ティッカー"] == ticker].sort_values("確認日")
        latest_row = rows.iloc[-1] if len(rows) else None
        div_change = float(latest_row["配当変化率%"]) if latest_row is not None else 0.0
        latest_anomaly_score = (
            float(latest_row["monthly_anomaly_score"]) if latest_row is not None else 0.0
        )
        consecutive = _count_consecutive_anomaly(rows)

        # evidence_metrics（ルール判定・LLM両方に渡す）
        evidence_metrics = {
            "配当性向%"              : det.get("配当性向%(複数年平均)", 0),
            "負債比率"               : det.get("負債比率", 0),
            "ROE%"                   : det.get("ROE%(複数年平均)", 0),
            "売上成長率%"            : det.get("売上成長率%", 0),
            "配当変化率%"            : div_change,
            "latest_anomaly_score"   : latest_anomaly_score,
            "consecutive_anomaly_count": consecutive,
            "hold_score"             : det["hold_score"],
            "hold_score_rank"        : rank_map[det["hold_score"]],
        }

        # 状態と理由コードを決定
        state, reason_codes = _determine_state(det, evidence_metrics)
        evidence_metrics["state"] = state

        logger.info(
            f"[{ticker}] {company[:20]} "
            f"hold_score={det['hold_score']:.1f} "
            f"rank={rank_map[det['hold_score']]} "
            f"state={state} "
            f"reasons={reason_codes}"
        )

        # hold は LLM を呼ばない（コスト節約）
        if state == "hold":
            llm_evidence = {"risk_summary": "", "review_comment": "", "backend": "skipped"}
        else:
            logger.info(f"  → LLMレビュー生成中...")
            llm_evidence = generate_llm_evidence(
                ticker, company, sector, evidence_metrics, reason_codes
            )

        candidates.append({
            "ticker"             : ticker,
            "company"            : company,
            "sector"             : sector,
            "state"              : state,
            "reason_codes"       : reason_codes,
            "evidence_metrics"   : evidence_metrics,
            "llm_evidence"       : llm_evidence,
            "manual_check_required": state in ("sell_candidate", "review_required"),
            "review_date"        : date.today().isoformat(),
        })

    # ── ソート: 深刻な順 ────────────────────────────────────────────────
    candidates.sort(key=lambda x: STATE_PRIORITY.get(x["state"], 0), reverse=True)

    # ── JSON 出力 ────────────────────────────────────────────────────────
    OUTPUT_JSON.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"JSON出力完了: {OUTPUT_JSON}")

    # ── HTMLレポート生成 ──────────────────────────────────────────────
    html = _generate_html_report(candidates)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    logger.info(f"HTMLレポート出力完了: {OUTPUT_HTML}")

    # ── メール通知 ────────────────────────────────────────────────────
    alert_targets = [c for c in candidates
                     if STATE_PRIORITY.get(c["state"], 0) >= STATE_PRIORITY["review_required"]]
    if alert_targets:
        _send_alert_email(candidates, alert_targets, html)
    else:
        logger.info("メール通知対象なし（review_required以上の銘柄なし）")

    # ── サマリ表示 ────────────────────────────────────────────────────
    logger.info("")
    logger.info("【判定サマリ】")
    for c in candidates:
        flag = "⚠" if c["manual_check_required"] else "✅"
        logger.info(
            f"  {flag} [{c['state']:16s}] {c['ticker']} {c['company'][:18]}"
            f"  hold={c['evidence_metrics']['hold_score']:.1f}"
            f"  reasons={c['reason_codes']}"
        )


# ===========================================================================
# データ読み込み
# ===========================================================================
def _load_final7() -> dict:
    if not FINAL7_JSON.exists():
        raise FileNotFoundError(f"final7.json が見つかりません: {FINAL7_JSON}")
    return json.loads(FINAL7_JSON.read_text(encoding="utf-8"))


def _load_monitor_log() -> pd.DataFrame:
    if not MONITOR_LOG.exists():
        logger.warning(f"monitor_log.xlsx が見つかりません: {MONITOR_LOG}")
        return pd.DataFrame(columns=["ティッカー", "確認日", "配当変化率%",
                                      "monthly_anomaly_score", "monthly_anomaly_flag"])
    return pd.read_excel(MONITOR_LOG)


# ===========================================================================
# hold_score 計算（0〜100点）
# ===========================================================================
def _calc_hold_score(det: dict) -> float:
    """
    既存の採点項目（安定性点・財務点・ROE点など）を正規化して hold_score を算出する。
    各項目は元スコアの最大値（10点）を基準に 0〜100 へスケーリング。
    """
    # dividend_safety: 安定性点（0〜10）→ 配当の安定度
    div_safety = min(det.get("安定性点", 5) / 10, 1.0) * 100

    # financial_stability: 財務点（0〜10）
    fin_stab = min(det.get("財務点", 5) / 10, 1.0) * 100

    # earnings_quality: ROE点 + 成長点（各0〜10、計0〜20）
    earn_quality = min((det.get("ROE点", 5) + det.get("成長点", 5)) / 20, 1.0) * 100

    # valuation: 配当性向点（0〜10）→ 適切な性向ほど高い
    valuation = min(det.get("配当性向点", 5) / 10, 1.0) * 100

    # sector_relative: 総合点（0〜100）をそのまま使用
    sector_rel = det.get("総合点", 50)

    w = HOLD_SCORE_WEIGHTS
    score = (
        div_safety   * w["dividend_safety"]
        + fin_stab   * w["financial_stability"]
        + earn_quality * w["earnings_quality"]
        + valuation  * w["valuation"]
        + sector_rel * w["sector_relative"]
    )
    return round(score, 1)


def _rank_map(scores: list[float]) -> dict[float, int]:
    """スコアリストを受け取り、{スコア: 順位} の辞書を返す（同点同順位）。"""
    sorted_unique = sorted(set(scores), reverse=True)
    return {s: sorted_unique.index(s) + 1 for s in scores}


# ===========================================================================
# 月次連続異常カウント
# ===========================================================================
def _count_consecutive_anomaly(rows: pd.DataFrame) -> int:
    """直近から遡って monthly_anomaly_score が閾値以下の連続回数を返す。"""
    if rows.empty:
        return 0
    flags = rows.sort_values("確認日", ascending=False)["monthly_anomaly_score"].tolist()
    count = 0
    for v in flags:
        if float(v) <= ANOMALY_SCORE_THRESHOLD:
            count += 1
        else:
            break
    return count


# ===========================================================================
# 状態遷移ルール（設計メモの4分類）
# ===========================================================================
def _determine_state(
    det: dict, em: dict
) -> tuple[str, list[str]]:
    """
    evidence_metrics と final7 詳細データから状態と理由コードを返す。
    state は最も深刻な判定を採用する。
    """
    state        = "hold"
    reason_codes : list[str] = []

    payout_ratio = em["配当性向%"]
    debt_ratio   = em["負債比率"]
    div_change   = em["配当変化率%"]
    consecutive  = em["consecutive_anomaly_count"]
    hold_rank    = em["hold_score_rank"]

    # ── A. 配当方針の毀損（最重要）──────────────────────────────────────
    if div_change <= DIV_CUT_THRESHOLD:
        state = _escalate(state, "sell_candidate")
        reason_codes.append("dividend_cut")

    # ── B. 配当原資の毀損 ────────────────────────────────────────────────
    if payout_ratio >= PAYOUT_WARN_THRESHOLD:
        state = _escalate(state, "review_required")
        reason_codes.append("high_payout_ratio")

    if consecutive >= CONSECUTIVE_ANOMALY_MIN:
        state = _escalate(state, "review_required")
        reason_codes.append(f"consecutive_anomaly_x{consecutive}")

    # ── C. 財務安全性の毀損 ──────────────────────────────────────────────
    if debt_ratio >= DEBT_RATIO_WARN:
        state = _escalate(state, "watch")
        reason_codes.append("high_debt_ratio")

    # ── D. 入れ替え余地 ──────────────────────────────────────────────────
    if hold_rank >= HOLD_SCORE_RANK_WARN:
        state = _escalate(state, "watch")
        reason_codes.append("low_hold_score_rank")

    return state, reason_codes


def _escalate(current: str, new: str) -> str:
    """現在の状態と新しい状態のうち、より深刻な方を返す。"""
    return new if STATE_PRIORITY.get(new, 0) > STATE_PRIORITY.get(current, 0) else current


# ===========================================================================
# HTMLレポート生成
# ===========================================================================
_STATE_STYLE = {
    "sell_confirmed" : ("border: 2px solid #c00; background: #fff;",       "■ SELL CONFIRMED",  "#c00"),
    "sell_candidate" : ("border: 2px solid #c00; background: #fff;",       "▲ SELL CANDIDATE",  "#c00"),
    "review_required": ("border: 2px solid #888; background: #fff;",       "● REVIEW REQUIRED", "#555"),
    "watch"          : ("border: 1px solid #aaa; background: #fafafa;",    "◎ WATCH",           "#666"),
    "hold"           : ("border: 1px solid #ccc; background: #fafafa;",    "○ HOLD",            "#999"),
}


def _generate_html_report(candidates: list[dict]) -> str:
    today    = date.today().isoformat()
    cards_html = ""

    for c in candidates:
        state  = c["state"]
        em     = c["evidence_metrics"]
        llm    = c["llm_evidence"]
        style, label, color = _STATE_STYLE.get(state, ("border:1px solid #ccc;", state, "#333"))

        reasons_html = (
            " / ".join(f'<code>{r}</code>' for r in c["reason_codes"])
            if c["reason_codes"] else '<span style="color:#999">なし</span>'
        )

        risk_html    = llm.get("risk_summary", "").replace("\n", "<br>") or "—"
        comment_html = llm.get("review_comment", "") or "—"
        backend      = llm.get("backend", "")

        manual_badge = (
            '<span style="border:1px solid #c00;color:#c00;padding:1px 6px;font-size:11px;border-radius:3px">要目視確認</span>'
            if c["manual_check_required"] else ""
        )

        cards_html += f"""
<div style="margin-bottom:18px; padding:14px 18px; {style} border-radius:4px;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
    <span style="font-weight:bold; font-size:15px; color:{color}">{label}</span>
    {manual_badge}
    <span style="font-size:12px; color:#888">{c['ticker']} / {c['sector']}</span>
  </div>
  <div style="font-size:16px; font-weight:bold; margin-bottom:10px">{c['company']}</div>

  <table style="width:100%; border-collapse:collapse; font-size:12px; margin-bottom:10px;">
    <tr style="background:#f0f0f0;">
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">hold_score</th>
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">順位</th>
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">配当性向%</th>
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">負債比率</th>
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">配当変化率%</th>
      <th style="border:0.8px solid #ccc; padding:4px 8px; text-align:left">連続異常</th>
    </tr>
    <tr>
      <td style="border:0.8px solid #ccc; padding:4px 8px"><strong>{em['hold_score']:.1f}</strong></td>
      <td style="border:0.8px solid #ccc; padding:4px 8px">{em['hold_score_rank']}位 / 7</td>
      <td style="border:0.8px solid #ccc; padding:4px 8px">{em['配当性向%']:.1f}%</td>
      <td style="border:0.8px solid #ccc; padding:4px 8px">{em['負債比率']:.2f}</td>
      <td style="border:0.8px solid #ccc; padding:4px 8px">{em['配当変化率%']:+.1f}%</td>
      <td style="border:0.8px solid #ccc; padding:4px 8px">{em['consecutive_anomaly_count']}回</td>
    </tr>
  </table>

  <div style="margin-bottom:6px; font-size:12px">
    <strong>判定理由:</strong> {reasons_html}
  </div>
  <div style="margin-bottom:6px; font-size:12px">
    <strong>懸念点 ({backend}):</strong><br>
    <div style="padding:6px 10px; background:#f8f8f8; border-left:3px solid #ccc; margin-top:4px; font-size:12px">{risk_html}</div>
  </div>
  <div style="font-size:12px">
    <strong>コメント:</strong><br>
    <div style="padding:6px 10px; background:#f8f8f8; border-left:3px solid #ccc; margin-top:4px; font-size:12px">{comment_html}</div>
  </div>
</div>"""

    alert_count = sum(1 for c in candidates if STATE_PRIORITY.get(c["state"], 0) >= STATE_PRIORITY["review_required"])
    banner = ""
    if alert_count:
        banner = f'<div style="border:1.5px solid #c00; padding:8px 14px; margin-bottom:16px; font-size:13px; color:#c00">⚠ review_required 以上の銘柄が {alert_count} 件あります。目視確認が必要です。</div>'

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>売り時判定レポート {today}</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size:13px; color:#222;
          max-width:860px; margin:0 auto; padding:20px; }}
  h1   {{ font-size:16px; border-bottom:1.5px solid #333; padding-bottom:6px; margin-bottom:4px; }}
  h2   {{ font-size:13px; color:#555; margin:0 0 16px; font-weight:normal; }}
  code {{ background:#f0f0f0; padding:1px 4px; border-radius:2px; font-size:11px; }}
</style>
</head>
<body>
<h1>長期安定配当株 売り時判定レポート</h1>
<h2>作成日: {today} &nbsp;|&nbsp; 対象銘柄: {len(candidates)}件 &nbsp;|&nbsp; 判定エンジン: sell_review.py + Gemma3:12b</h2>
{banner}
{cards_html}
<hr style="border:none; border-top:0.8px solid #ccc; margin-top:24px;">
<p style="font-size:11px; color:#999">
  このレポートは投資助言ではありません。自動処理は sell_candidate までに止め、sell_confirmed は人間の確認後のみ付与してください。
</p>
</body>
</html>"""


# ===========================================================================
# メール通知
# ===========================================================================
def _send_alert_email(
    all_candidates: list[dict],
    alert_targets: list[dict],
    html_body: str,
) -> None:
    today = date.today().isoformat()
    subject = f"【売り時判定】要確認 {len(alert_targets)}件 ({today})"

    if not MAIL_ENABLED:
        logger.warning(f"メール設定未検出のためスキップ: {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = TO_EMAIL
    msg["Subject"] = subject

    # テキスト版（フォールバック）
    text_lines = [f"売り時判定レポート {today}", ""]
    for c in alert_targets:
        em = c["evidence_metrics"]
        text_lines.append(
            f"[{c['state'].upper():16s}] {c['ticker']} {c['company']}"
            f"  hold={em['hold_score']:.1f}  reasons={c['reason_codes']}"
        )
    text_lines += ["", "詳細は添付HTMLレポートを参照してください。"]
    msg.attach(MIMEText("\n".join(text_lines), "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info(f"メール送信完了 → {TO_EMAIL}  件名: {subject}")
    except Exception as e:
        logger.error(f"メール送信失敗: {e}")


# ===========================================================================
if __name__ == "__main__":
    main()
