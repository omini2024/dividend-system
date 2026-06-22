#!/usr/bin/env python3
"""
jquants_signals.py — J-Quants API によるイベント・センチメントシグナル
======================================================================
predict_stock.py の特徴量として追加する以下の情報を J-Quants API から取得する。

【1】決算発表カレンダー（get_eq_earnings_cal）
    ・days_to_earnings     : 次回決算発表まで何営業日か（PEADアノマリーに有効）
    ・days_since_earnings  : 前回決算発表から何営業日経過したか
    ・is_earnings_week     : 決算発表の前後5営業日以内なら 1

【2】投資家別売買動向（get_eq_investor_types）
    ・frgn_net_ratio       : 外国人投資家の純買い比率（買い÷(買い+売り) - 0.5）
    ・invtrust_net_ratio   : 投資信託の純買い比率
    ・bank_net_ratio       : 銀行・生保の純買い比率（大量保有に近い主体）

使い方:
    from jquants_signals import load_jquants_signals
    earnings_map, sentiment_df = load_jquants_signals(tickers, api_key)
    # → build_features() に渡す
"""

import warnings
from pathlib import Path

import jquantsapi
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 設定 ────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "output" / "jquants_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EARNINGS_WINDOW = 5    # 決算発表の前後何営業日以内を「決算週」とするか


# ─────────────────────────────────────────────────────────────────────────────
# J-Quants クライアント初期化
# ─────────────────────────────────────────────────────────────────────────────

def get_client(api_key: str) -> jquantsapi.ClientV2:
    return jquantsapi.ClientV2(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# 【1】決算発表カレンダー
# ─────────────────────────────────────────────────────────────────────────────

def fetch_earnings_calendar(client: jquantsapi.ClientV2) -> pd.DataFrame:
    """
    全銘柄の決算発表予定を取得する。
    Returns: Code(str), Date(datetime)
    """
    from datetime import date
    cache = CACHE_DIR / f"earnings_cal_{date.today().strftime('%Y%m%d')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print("[J-Quants] 決算発表カレンダー取得中...")
    df = client.get_eq_earnings_cal()
    df["Code"] = df["Code"].astype(str).str.zfill(4)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(cache, index=False)
    print(f"  取得完了: {len(df)} 件")
    return df


def build_earnings_map(earnings_df: pd.DataFrame) -> dict[str, list]:
    """
    Code → 決算発表日リスト のマッピング辞書を生成する。
    """
    result = {}
    for code, grp in earnings_df.groupby("Code"):
        result[code] = sorted(grp["Date"].tolist())
    return result


def add_earnings_features(
    price_df:    pd.DataFrame,
    ticker:      str,
    earnings_map: dict[str, list],
) -> pd.DataFrame:
    """
    決算発表日に基づく特徴量を price_df に追加する。

    追加列:
      days_to_earnings    : 次回決算まで何日か（なければ 999）
      days_since_earnings : 前回決算から何日か（なければ 999）
      is_earnings_week    : 決算前後 EARNINGS_WINDOW 日以内なら 1
    """
    df = price_df.copy()

    # ティッカー → 4桁コード変換（3076.T → 3076）
    code = ticker.split(".")[0].zfill(4)
    dates = earnings_map.get(code, [])

    to_next   = []
    from_prev = []
    is_week   = []

    for idx_date in df.index:
        d       = pd.Timestamp(idx_date).normalize()
        future  = [x for x in dates if x > d]
        past    = [x for x in dates if x <= d]

        d_to   = (future[0] - d).days if future else 999
        d_from = (d - past[-1]).days  if past   else 999
        week   = 1 if (d_to <= EARNINGS_WINDOW or d_from <= EARNINGS_WINDOW) else 0

        to_next.append(d_to)
        from_prev.append(d_from)
        is_week.append(week)

    df["days_to_earnings"]    = to_next
    df["days_since_earnings"] = from_prev
    df["is_earnings_week"]    = is_week
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 【2】投資家別売買動向（市場センチメント）
# ─────────────────────────────────────────────────────────────────────────────

def fetch_investor_sentiment(
    client: jquantsapi.ClientV2,
    from_yyyymmdd: str = "20240101",
    to_yyyymmdd:   str = "",
) -> pd.DataFrame:
    """
    投資家別売買動向（週次）から市場全体のセンチメント指標を生成する。

    生成列:
      frgn_net_ratio    : 外国人の純買い比率
      invtrust_net_ratio: 投資信託の純買い比率
      bank_net_ratio    : 銀行等の純買い比率（大口機関の動向）
    """
    from datetime import date
    cache = CACHE_DIR / f"investor_sentiment_{date.today().strftime('%Y%m%d')}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print("[J-Quants] 投資家別売買動向取得中...")
    if not to_yyyymmdd:
        to_yyyymmdd = date.today().strftime("%Y%m%d")

    df = client.get_eq_investor_types(
        from_yyyymmdd=from_yyyymmdd,
        to_yyyymmdd=to_yyyymmdd,
    )

    # TSEPrime のみ使用（最も市場を代表）
    df = df[df["Section"] == "TSEPrime"].copy()
    df["PubDate"] = pd.to_datetime(df["PubDate"])

    def net_ratio(buy, sell):
        total = buy + sell
        return np.where(total > 0, (buy - sell) / total, 0)

    df["frgn_net_ratio"]     = net_ratio(df["FrgnBuy"],   df["FrgnSell"])
    df["invtrust_net_ratio"] = net_ratio(df["InvTrBuy"],  df["InvTrSell"])
    df["bank_net_ratio"]     = net_ratio(
        df["BankBuy"] + df["InsCoBuy"],
        df["BankSell"] + df["InsCoSell"],
    )

    result = df[["PubDate", "frgn_net_ratio", "invtrust_net_ratio", "bank_net_ratio"]].copy()
    result.sort_values("PubDate", inplace=True)
    result.reset_index(drop=True, inplace=True)
    result.to_parquet(cache, index=False)
    print(f"  取得完了: {len(result)} 週分")
    return result


def add_sentiment_features(
    price_df:     pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    投資家別売買動向を price_df の各日付に対応させて追加する。
    週次データのため、各価格行には直近の週のデータを対応させる。
    """
    df = price_df.copy()

    # 週次 → 日次に前方補完（asof merge）
    sentiment_sorted = sentiment_df.sort_values("PubDate")

    cols = ["frgn_net_ratio", "invtrust_net_ratio", "bank_net_ratio"]
    for col in cols:
        df[col] = np.nan

    price_dates = pd.to_datetime(df.index).normalize()

    for col in cols:
        merged = pd.merge_asof(
            pd.DataFrame({"date": price_dates}),
            sentiment_sorted[["PubDate", col]].rename(columns={"PubDate": "date"}),
            on="date",
            direction="backward",
        )
        df[col] = merged[col].values

    df[cols] = df[cols].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# まとめて取得するラッパー
# ─────────────────────────────────────────────────────────────────────────────

def load_jquants_signals(
    tickers: list[str],
    api_key: str,
) -> tuple[dict, pd.DataFrame]:
    """
    predict_stock.py から呼び出す統合ラッパー。

    Returns
    -------
    earnings_map  : dict  {code: [決算発表日リスト]}
    sentiment_df  : DataFrame  投資家別売買動向の週次データ
    """
    client       = get_client(api_key)
    earnings_df  = fetch_earnings_calendar(client)
    earnings_map = build_earnings_map(earnings_df)
    sentiment_df = fetch_investor_sentiment(client, from_yyyymmdd="20240101")
    return earnings_map, sentiment_df


# ─────────────────────────────────────────────────────────────────────────────
# predict_stock.build_features から呼ぶ統合関数
# ─────────────────────────────────────────────────────────────────────────────

def add_jquants_features(
    price_df:     pd.DataFrame,
    ticker:       str,
    earnings_map: dict,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """決算 + センチメントの特徴量を一括追加する。"""
    df = add_earnings_features(price_df, ticker, earnings_map)
    df = add_sentiment_features(df, sentiment_df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 単独実行時: シグナル一覧を確認
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import predict_stock as ps

    API_KEY = ps._ROOT.parent / "dividend-system" / "annual_select.py"
    # annual_select.py からAPIキーを読み取る
    import re
    src = (Path(__file__).parent.parent / "annual_select.py").read_text()
    m   = re.search(r'JQUANTS_API_KEY\s*=\s*["\']([^"\']+)["\']', src)
    if not m:
        print("annual_select.py から J-Quants APIキーが見つかりません。")
        sys.exit(1)
    api_key = m.group(1)
    print(f"APIキー: {api_key[:8]}...")

    fund_df = ps.load_fundamentals()
    tickers = fund_df["ティッカー"].tolist()

    print("\n" + "=" * 60)
    print("J-Quants シグナル取得")
    print("=" * 60)

    earnings_map, sentiment_df = load_jquants_signals(tickers, api_key)

    print("\n=== 直近の決算発表予定（Final7銘柄） ===")
    final7 = ["3076.T", "7270.T", "2168.T", "4611.T", "2307.T", "4776.T", "5970.T"]
    from datetime import date
    today = pd.Timestamp(date.today())
    for t in final7:
        code  = t.split(".")[0].zfill(4)
        dates = earnings_map.get(code, [])
        future = [d for d in dates if d > today]
        past   = [d for d in dates if d <= today]
        nxt    = future[0].strftime("%Y-%m-%d") if future else "不明"
        prv    = past[-1].strftime("%Y-%m-%d")  if past   else "不明"
        name   = fund_df[fund_df["ティッカー"] == t]["会社名"].values
        name   = name[0] if len(name) else ""
        print(f"  {t:8} {str(name)[:20]:22} 前回={prv}  次回={nxt}")

    print("\n=== 最新の投資家別センチメント ===")
    latest = sentiment_df.iloc[-1]
    print(f"  集計日        : {latest['PubDate'].strftime('%Y-%m-%d')}")
    print(f"  外国人純買い比率: {latest['frgn_net_ratio']:+.3f}")
    print(f"  投信純買い比率  : {latest['invtrust_net_ratio']:+.3f}")
    print(f"  銀行・生保比率  : {latest['bank_net_ratio']:+.3f}")
