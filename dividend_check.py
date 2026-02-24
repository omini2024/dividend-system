# ==============================
# 長期安定配当株 自動抽出システム（最終版）
# ==============================

import yfinance as yf
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
# ===== メール通知 =====
import smtplib
from email.mime.text import MIMEText
from email_config import EMAIL_ADDRESS, EMAIL_PASSWORD

def send_alert(subject, body):
    print("send_alert called")

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    from email_config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    print("SMTP connect...")
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()

    print("SMTP login...")
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

    print("SMTP send...")
    server.send_message(msg)
    server.quit()

    print("SMTP send OK")

# ========= 設定 =========
YEARS = 15
JPX_FILE = "/home/kageta/projects/dividend_project/dividend-system/data_j.xlsx"
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
send_alert(
    "【配当システム】本日の監視結果",
    f"Final7 が {len(df_final)} 銘柄抽出されました。\n\n{df_final[['ティッカー','会社名']].to_string(index=False)}"
)
def one_year_return(ticker, year):
    try:
        start = f"{year}-01-01"
        end = f"{year+1}-01-01"

        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            return None

        # MultiIndex対策
        if isinstance(df.columns, tuple) or hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)

        start_price = df["Close"].iloc[0]
        end_price = df["Close"].iloc[-1]

        return (end_price / start_price) - 1

    except:
        return None
    
def select_final7(df):
    df = df.sort_values("総合点", ascending=False)

    selected = []
    sector_count = {}

    for _, row in df.iterrows():
        sector = row["業種"]

        if sector_count.get(sector, 0) < 2:
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1

        if len(selected) == 7:
            break

    return pd.DataFrame(selected)

def run_backtest(df_out, start_year=2014, end_year=2023):

    results = []

    for year in range(start_year, end_year + 1):
        print(f"Backtesting {year}...")

        final7 = select_final7(df_out)

        yearly_returns = []

        for t in final7["ティッカー"]:
            r = one_year_return(t, year)
            if r is not None:
                yearly_returns.append(r)

        if len(yearly_returns) > 0:
            avg_return = sum(yearly_returns) / len(yearly_returns)
        else:
            avg_return = None

        results.append({
            "Year": year,
            "Average Return": avg_return
        })

    return pd.DataFrame(results)

print("Running Backtest...")

bt_df = run_backtest(df_all)

cagr = ((1 + bt_df["Average Return"]).prod()) ** (1 / len(bt_df)) - 1

print("strategy CAGR:", round(cagr * 100, 2), "%")

cum = (1 + bt_df["Average Return"]).cumprod()
drawdown = cum / cum.cummax() - 1

print("\n=== Backtest Result ===")
print("CAGR:", round(cagr * 100, 2), "%")
print("Max Drawdown:", round(drawdown.min() * 100, 2), "%")

import matplotlib.pyplot as plt

bt_df.plot(x="Year", y="Average Return", kind="bar", title="Yearly Average Return")
plt.tight_layout()
plt.savefig("output/backtest_bar.png")
plt.close()

print("=== BEFORE SEND ALERT ===")
send_alert("テスト送信", "dividend_check.py からのテストです")
print("=== AFTER SEND ALERT ===")
print("### MAIL TEST START ###")
raise SystemExit("STOP HERE")