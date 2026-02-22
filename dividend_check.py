# ==============================
# 長期安定配当株 自動抽出システム（最終版）
# ==============================

import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

# ========= 設定 =========
YEARS = 15
JPX_FILE = "data_j.xlsx"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_result.xlsx")

# ========= フォルダ作成 =========
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= JPXデータ読み込み =========
jpx = pd.read_excel(JPX_FILE)

prime = jpx[jpx["市場・商品区分"].str.contains("プライム", na=False)]
tickers = (prime["コード"].astype(str) + ".T").tolist()

today = datetime.today()
start_year = today.year - YEARS

rows = []

print("Checking Prime stocks...")

# ==============================
# メイン処理
# ==============================

for t in tqdm(tickers):
    try:
        stock = yf.Ticker(t)
        div = stock.dividends
        info = stock.info

        if div.empty:
            continue

        df = div.to_frame(name="dividend")
        df["year"] = df.index.year
        yearly = df.groupby("year").sum()

        # --- 15年連続配当チェック ---
        ok = True
        for y in range(start_year, today.year):
            if y not in yearly.index or yearly.loc[y, "dividend"] <= 0:
                ok = False
                break
        if not ok:
            continue

        price = info.get("currentPrice") or 0
        dividend_yield = (info.get("dividendYield") or 0) * 100
        payout = (info.get("payoutRatio") or 0) * 100
        roe = (info.get("returnOnEquity") or 0) * 100
        growth = (info.get("revenueGrowth") or 0) * 100
        debt = info.get("debtToEquity") or 0

        # --- スコアリング ---
        yield_score = 2 if dividend_yield >= 4 else (1 if dividend_yield >= 3 else 0)
        payout_score = 2 if payout <= 60 else (1 if payout <= 100 else 0)
        roe_score = 2 if roe >= 10 else (1 if roe >= 5 else 0)
        growth_score = 2 if growth >= 5 else (1 if growth >= 0 else 0)
        debt_score = 2 if debt <= 100 else (1 if debt <= 200 else 0)

        total_score = (
            yield_score +
            payout_score +
            roe_score +
            growth_score +
            debt_score
        )

        rows.append({
            "ティッカー": t,
            "会社名": info.get("shortName", ""),
            "業種": info.get("sector", ""),
            "株価": price,
            "利回り%": round(dividend_yield, 2),
            "配当性向%": round(payout, 2),
            "ROE%": round(roe, 2),
            "売上成長率%": round(growth, 2),
            "負債比率": round(debt, 2),
            "利回り点": yield_score,
            "配当性向点": payout_score,
            "ROE点": roe_score,
            "成長点": growth_score,
            "財務点": debt_score,
            "総合点": total_score,
            "15年連続配当": "YES"
        })

    except Exception:
        pass

# ==============================
# DataFrame化
# ==============================

df_all = pd.DataFrame(rows)

# ==============================
# Fモデル：業種最大2社ルール
# ==============================

df_sorted = df_all.sort_values("総合点", ascending=False)

selected = []
sector_count = {}

for _, row in df_sorted.iterrows():
    sector = row["業種"]

    if sector_count.get(sector, 0) < 2:
        selected.append(row)
        sector_count[sector] = sector_count.get(sector, 0) + 1

    if len(selected) == 7:
        break

df_final = pd.DataFrame(selected)

# ==============================
# Excel出力
# ==============================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_all.to_excel(writer, index=False, sheet_name="AllCandidates")
    df_final.to_excel(writer, index=False, sheet_name="Final7")

print("Excel created:", OUTPUT_FILE)
print("AllCandidates:", len(df_all))
print("Final7:", len(df_final))