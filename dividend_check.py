import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

# ==============================
# 設定
# ==============================
YEARS = 15
JPX_FILE = "data_j.xlsx"
OUTPUT_DIR = "output"
OUTPUT_FILE = "output/final_result.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 配当CAGR計算
# ==============================
def calc_dividend_cagr(div_series, years=5):
    if div_series.empty:
        return 0

    df = div_series.to_frame(name="dividend")
    df["year"] = df.index.year
    yearly = df.groupby("year").sum()

    if len(yearly) < years + 1:
        return 0

    latest_year = yearly.index.max()
    past_year = latest_year - years

    if past_year not in yearly.index:
        return 0

    start = yearly.loc[past_year, "dividend"]
    end = yearly.loc[latest_year, "dividend"]

    if start <= 0 or end <= 0:
        return 0

    cagr = (end / start) ** (1 / years) - 1
    return round(cagr * 100, 2)

# ==============================
# 減配回数カウント
# ==============================
def count_dividend_cuts(yearly):
    values = yearly["dividend"].values
    cuts = 0
    for i in range(1, len(values)):
        if values[i] < values[i-1]:
            cuts += 1
    return cuts

# ==============================
# JPX読み込み
# ==============================
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

        # 15年連続配当チェック
        ok = True
        for y in range(start_year, today.year):
            if y not in yearly.index or yearly.loc[y, "dividend"] <= 0:
                ok = False
                break
        if not ok:
            continue

        # 指標取得
        yield_rate = (info.get("dividendYield") or 0) * 100
        payout = (info.get("payoutRatio") or 0) * 100
        roe = (info.get("returnOnEquity") or 0) * 100
        growth = (info.get("revenueGrowth") or 0) * 100
        debt = info.get("debtToEquity") or 0

        div_cagr = calc_dividend_cagr(div)
        cuts = count_dividend_cuts(yearly)

        # ======================
        # スコアリング
        # ======================
        yield_score = 2 if yield_rate >= 4 else 1 if yield_rate >= 3 else 0
        payout_score = 2 if payout <= 40 else 1 if payout <= 60 else 0
        roe_score = 2 if roe >= 10 else 1 if roe >= 7 else 0
        growth_score = 1 if growth > 0 else 0
        debt_score = 2 if debt <= 50 else 1 if debt <= 70 else 0

        if div_cagr >= 5:
            div_growth_score = 3
        elif div_cagr >= 3:
            div_growth_score = 2
        elif div_cagr >= 1:
            div_growth_score = 1
        else:
            div_growth_score = 0

        if cuts == 0:
            stability_score = 3
        elif cuts == 1:
            stability_score = 2
        elif cuts == 2:
            stability_score = 1
        else:
            stability_score = 0

        total_score = (
            yield_score +
            payout_score +
            roe_score +
            growth_score +
            debt_score +
            div_growth_score +
            stability_score
        )

        rows.append({
            "ティッカー": t,
            "会社名": info.get("shortName", ""),
            "業種": info.get("sector", ""),
            "株価": info.get("currentPrice", 0),
            "利回り%": round(yield_rate,2),
            "配当性向%": round(payout,2),
            "ROE%": round(roe,2),
            "売上成長率%": round(growth,2),
            "負債比率": round(debt,2),
            "配当成長率%": div_cagr,
            "減配回数": cuts,
            "総合点": total_score
        })

    except Exception:
        pass

# ==============================
# DataFrame化
# ==============================
df_out = pd.DataFrame(rows)
df_out = df_out.sort_values("総合点", ascending=False)

# ==============================
# Fモデル（業種最大2社）
# ==============================
selected = []
sector_count = {}

for _, row in df_out.iterrows():
    sector = row["業種"]
    if sector_count.get(sector, 0) < 2:
        selected.append(row)
        sector_count[sector] = sector_count.get(sector, 0) + 1
    if len(selected) == 7:
        break

final_df = pd.DataFrame(selected)

# ==============================
# Excel出力
# ==============================
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df_out.to_excel(writer, index=False, sheet_name="AllCandidates")
    final_df.to_excel(writer, index=False, sheet_name="Final7")

print("Excel created:", OUTPUT_FILE)
print("AllCandidates:", len(df_out))
print("Final7:", len(final_df))