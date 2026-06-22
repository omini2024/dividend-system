#!/usr/bin/env python3
"""
edinet_signals.py — EDINET APIによる大量保有報告書シグナル取得
================================================================
以下の情報をEDINET API（無料・APIキー不要）から取得し、
predict_stock.py の特徴量として追加できる形に整形する。

  ・大量保有報告書（新規）: ordinanceCode=04 / formCode=030000
  ・大量保有報告書（変更）: ordinanceCode=04 / formCode=030001

生成される特徴量:
  days_since_large_holder  : 直近の大量保有報告書からの経過日数
  large_holder_flag_30d    : 直近30日以内に提出があれば 1
  large_holder_flag_90d    : 直近90日以内に提出があれば 1

使い方:
  from edinet_signals import add_large_holder_features
  price_df = add_large_holder_features(price_df, ticker)
"""

import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
EDINET_BASE   = "https://disclosure.edinet-api.go.jp/api/v2"
LOOKBACK_DAYS = 90          # 何日分遡るか
API_SLEEP     = 0.5         # API呼び出し間隔（秒）
CACHE_DIR     = Path(__file__).parent.parent / "output" / "edinet_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LARGE_HOLDER_FORMS = {"030000", "030001"}   # 大量保有（新規・変更）


# ─────────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def ticker_to_seccode(ticker: str) -> str:
    """
    ティッカー → EDINET の secCode（5桁）に変換
    例: '3076.T' → '30760', '72700.T' → '72700'
    """
    code = ticker.split(".")[0]   # '.T' を除去
    if len(code) == 4:
        return code + "0"         # 4桁 → 末尾に '0' を追加
    return code.zfill(5)


# ─────────────────────────────────────────────────────────────────────────────
# EDINET API 呼び出し
# ─────────────────────────────────────────────────────────────────────────────

def fetch_documents_one_day(date_str: str) -> list:
    """
    指定日の提出書類一覧を取得する。
    失敗した場合は空リストを返す（証券取引所の休業日は件数0になる）。
    """
    url    = f"{EDINET_BASE}/documents.json"
    params = {"date": date_str, "type": 2}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  [EDINET] {date_str} 取得失敗: {e}")
        return []


def fetch_large_holder_signals(
    tickers:      list[str],
    lookback_days: int = LOOKBACK_DAYS,
    use_cache:    bool = True,
) -> pd.DataFrame:
    """
    直近 lookback_days 日間の大量保有報告書を全銘柄分取得してまとめる。

    Returns
    -------
    DataFrame: columns = [date, ticker, form_type, filer_name, doc_id]
    """
    cache_file = CACHE_DIR / f"large_holder_{date.today().strftime('%Y%m%d')}.parquet"

    # キャッシュが当日分あればそれを使う
    if use_cache and cache_file.exists():
        print("[EDINET] キャッシュから大量保有情報を読み込み")
        return pd.read_parquet(cache_file)

    # ティッカー → secCode のマッピング辞書
    sec_map = {ticker_to_seccode(t): t for t in tickers}

    end_date   = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    biz_days   = pd.bdate_range(start_date, end_date)   # 営業日のみ

    print(f"[EDINET] 大量保有報告書を取得中 ({start_date} 〜 {end_date}、{len(biz_days)} 営業日)")

    records = []
    for i, bday in enumerate(biz_days):
        ds   = bday.strftime("%Y-%m-%d")
        docs = fetch_documents_one_day(ds)

        for doc in docs:
            if doc.get("formCode") not in LARGE_HOLDER_FORMS:
                continue
            sec = doc.get("secCode", "")
            if sec not in sec_map:
                continue
            records.append({
                "date":       pd.Timestamp(bday),
                "ticker":     sec_map[sec],
                "form_type":  "新規" if doc["formCode"] == "030000" else "変更",
                "filer_name": doc.get("filerName", ""),
                "doc_id":     doc.get("docID", ""),
            })

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(biz_days)} 日処理済み（取得件数: {len(records)}）")
        time.sleep(API_SLEEP)

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["date", "ticker", "form_type", "filer_name", "doc_id"]
    )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.to_parquet(cache_file, index=False)
        print(f"[EDINET] 取得完了: {len(df)} 件 → {cache_file}")
    else:
        print("[EDINET] 対象銘柄の大量保有報告書なし")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 特徴量の生成
# ─────────────────────────────────────────────────────────────────────────────

def add_large_holder_features(
    price_df:   pd.DataFrame,
    ticker:     str,
    signals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    price_df（終値 DataFrame）に大量保有報告書の特徴量を追加する。

    追加列:
      days_since_large_holder : 直近の提出からの経過営業日数（提出がなければ 999）
      large_holder_flag_30d   : 直近 30 日以内に提出があれば 1
      large_holder_flag_90d   : 直近 90 日以内に提出があれば 1
    """
    df = price_df.copy()

    # この銘柄の提出日一覧
    tk_signals = signals_df[signals_df["ticker"] == ticker].copy()

    if tk_signals.empty:
        df["days_since_large_holder"] = 999
        df["large_holder_flag_30d"]   = 0
        df["large_holder_flag_90d"]   = 0
        return df

    submit_dates = sorted(tk_signals["date"].dt.normalize().unique())

    days_since  = []
    flag_30d    = []
    flag_90d    = []

    for idx_date in df.index:
        d = pd.Timestamp(idx_date).normalize()
        past = [s for s in submit_dates if s <= d]
        if past:
            delta = (d - past[-1]).days
        else:
            delta = 999
        days_since.append(delta)
        flag_30d.append(1 if delta <= 30  else 0)
        flag_90d.append(1 if delta <= 90  else 0)

    df["days_since_large_holder"] = days_since
    df["large_holder_flag_30d"]   = flag_30d
    df["large_holder_flag_90d"]   = flag_90d
    return df


# ─────────────────────────────────────────────────────────────────────────────
# predict_stock.py への組み込み用ラッパー
# ─────────────────────────────────────────────────────────────────────────────

def load_or_fetch_signals(tickers: list[str]) -> pd.DataFrame:
    """
    predict_stock.py から呼び出す想定のラッパー。
    当日キャッシュがあれば使い、なければ API から取得する。
    """
    return fetch_large_holder_signals(tickers, lookback_days=LOOKBACK_DAYS)


# ─────────────────────────────────────────────────────────────────────────────
# 単独実行時: シグナル一覧を表示
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import predict_stock as ps

    print("=" * 60)
    print("EDINET 大量保有報告書シグナル取得")
    print("=" * 60)

    fund_df = ps.load_fundamentals()
    tickers = fund_df["ティッカー"].tolist()

    signals = fetch_large_holder_signals(tickers, lookback_days=90)

    if signals.empty:
        print("直近90日間に対象銘柄の大量保有報告書は見つかりませんでした。")
    else:
        print(f"\n直近90日間の大量保有報告書 ({len(signals)} 件):\n")
        print(signals.to_string(index=False))

        print("\n銘柄別集計:")
        summary = (
            signals.groupby(["ticker", "form_type"])
            .size()
            .reset_index(name="件数")
            .sort_values("件数", ascending=False)
        )
        # 会社名を付加
        name_map = fund_df.set_index("ティッカー")["会社名"].to_dict() if "会社名" in fund_df.columns else {}
        summary["会社名"] = summary["ticker"].map(name_map).fillna("")
        print(summary[["ticker", "会社名", "form_type", "件数"]].to_string(index=False))
