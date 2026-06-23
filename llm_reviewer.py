"""
llm_reviewer.py
───────────────
Gemma3:12b（ollama）を使った定性レビュー生成モジュール。
数値判定・状態遷移はルールベースの sell_review.py が行う。
このモジュールは「根拠文の生成」だけを担う。

呼び出し例:
    from llm_reviewer import generate_llm_evidence
    evidence = generate_llm_evidence(ticker, company, metrics, reason_codes)
"""

import json
import logging
import time
import urllib.request
from typing import Any

logger = logging.getLogger("sell_review.llm")

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:12b"

# LLMの出力は揺れやすいため低温度で固定
_DEFAULT_OPTIONS = {"temperature": 0.1, "num_predict": 512}


# ---------------------------------------------------------------------------
# 内部: ollama 呼び出し
# ---------------------------------------------------------------------------
def _call_ollama(prompt: str) -> str:
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": _DEFAULT_OPTIONS,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.loads(resp.read()).get("response", "").strip()
    logger.debug(f"ollama 応答: {time.time()-t0:.1f}秒 / {len(text)}文字")
    return text


# ---------------------------------------------------------------------------
# 公開インターフェース
# ---------------------------------------------------------------------------
def generate_llm_evidence(
    ticker: str,
    company: str,
    sector: str,
    metrics: dict[str, Any],
    reason_codes: list[str],
) -> dict[str, str]:
    """
    銘柄の定量指標と判定理由コードを受け取り、
    保有継続レビュー用の根拠文をGemma3:12bで生成する。

    Returns:
        {
            "risk_summary": "懸念点の箇条書き（LLM生成）",
            "review_comment": "保有継続/売却検討コメント（LLM生成）",
            "backend": "Gemma3:12b"
        }
    """
    try:
        risk_summary  = _generate_risk_summary(ticker, company, sector, metrics, reason_codes)
        review_comment = _generate_review_comment(ticker, company, metrics, reason_codes, risk_summary)
        return {
            "risk_summary"  : risk_summary,
            "review_comment": review_comment,
            "backend"       : OLLAMA_MODEL,
        }
    except Exception as e:
        logger.error(f"[{ticker}] LLMレビュー生成失敗: {e}")
        return {
            "risk_summary"  : "(生成失敗)",
            "review_comment": "(生成失敗)",
            "backend"       : OLLAMA_MODEL,
        }


# ---------------------------------------------------------------------------
# 内部: リスクサマリ生成
# ---------------------------------------------------------------------------
_RISK_PROMPT = """
あなたは長期配当株の保有継続審査を行うアナリストです。
以下の銘柄情報を読み、配当継続性・財務安全性の観点から懸念点を日本語で3点以内で箇条書きにしてください。
余分な説明は不要です。箇条書きのみ出力してください。

【銘柄】{ticker} {company}（{sector}）

【主要指標】
- 配当性向（複数年平均）: {payout_ratio}%
- 負債比率: {debt_ratio}
- ROE（複数年平均）: {roe}%
- 売上成長率: {sales_growth}%
- 月次異常スコア: {anomaly_score}（直近）
- 月次連続異常回数: {consecutive_anomaly}回

【判定理由コード】
{reason_codes}

【懸念点（箇条書き、3点以内）】
""".strip()


def _generate_risk_summary(
    ticker: str, company: str, sector: str,
    metrics: dict[str, Any], reason_codes: list[str],
) -> str:
    prompt = _RISK_PROMPT.format(
        ticker            = ticker,
        company           = company,
        sector            = sector,
        payout_ratio      = metrics.get("配当性向%", "不明"),
        debt_ratio        = metrics.get("負債比率", "不明"),
        roe               = metrics.get("ROE%", "不明"),
        sales_growth      = metrics.get("売上成長率%", "不明"),
        anomaly_score     = metrics.get("latest_anomaly_score", "不明"),
        consecutive_anomaly = metrics.get("consecutive_anomaly_count", 0),
        reason_codes      = "\n".join(f"- {r}" for r in reason_codes) if reason_codes else "- なし（正常）",
    )
    return _call_ollama(prompt)


# ---------------------------------------------------------------------------
# 内部: 保有継続コメント生成
# ---------------------------------------------------------------------------
_COMMENT_PROMPT = """
あなたは長期配当株の保有継続審査を行うアナリストです。
以下の情報をもとに、この銘柄の保有継続可否について100〜150字の日本語コメントを1つ書いてください。
余分な前置きは不要です。コメント本文のみ出力してください。

【銘柄】{ticker} {company}
【保有継続スコア】{hold_score}点（100点満点）
【状態判定】{state}
【判定理由】{reason_codes}
【懸念点】
{risk_summary}

【コメント】
""".strip()


def _generate_review_comment(
    ticker: str, company: str,
    metrics: dict[str, Any], reason_codes: list[str],
    risk_summary: str,
) -> str:
    prompt = _COMMENT_PROMPT.format(
        ticker       = ticker,
        company      = company,
        hold_score   = metrics.get("hold_score", "不明"),
        state        = metrics.get("state", "不明"),
        reason_codes = "、".join(reason_codes) if reason_codes else "なし",
        risk_summary = risk_summary,
    )
    return _call_ollama(prompt)
