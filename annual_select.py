# ==============================
# 【年1回実行】長期安定配当株 メイン選定スクリプト（Cモデル・本格版）
# 実行タイミング：年1回（毎年4月1日 ※休日の場合は翌営業日）
# 役割：全プライム上場銘柄をスキャンしてFinal7を決定・保存
#
# 【Cモデルの改良点】
# 改良1: 株価を履歴データから自前取得（info.get()の不安定さを排除）
# 改良2: 配当利回りを直近3年平均配当で計算（単年異常値の影響を平滑化）
# 改良3: ROE・配当性向を複数年平均で計算（単年決算異常値の影響を平滑化）
# 改良4: 異常値はcontinueで除外せずフラグを立てて残す（目視確認可能）
# ==============================

import yfinance as yf
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import os
import json
import numpy as np

try:
    from email_config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL
    MAIL_ENABLED = True
except ImportError:
    MAIL_ENABLED = False
    print("⚠️  email_config が見つかりません。メール通知はスキップします。")

# ========= 設定 =========
YEARS      = 15
AVG_YEARS  = 3        # 利回り・ROE・配当性向の平均計算に使う年数
JPX_FILE   = "data_j.xlsx"
OUTPUT_DIR = "output"
ANNUAL_RESULT_FILE = os.path.join(OUTPUT_DIR, "annual_result.xlsx")
FINAL7_JSON        = os.path.join(OUTPUT_DIR, "final7.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

today = datetime.today()

# 4月1日起点の会計年度ベース
if today.month >= 4:
    fiscal_year = today.year
else:
    fiscal_year = today.year - 1

start_year = fiscal_year - YEARS


# ========= メール通知 =========
def send_alert(subject, body):
    if not MAIL_ENABLED:
        print(f"[メール省略] {subject}\n{body}")
        return
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_ADDRESS
    msg["To"]      = TO_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
    server.send_message(msg)
    server.quit()


# ============================================================
# 【改良1】株価を履歴データから自前取得
# 【旧コード】info.get("currentPrice") → 取得ミスが多く異常値の原因
# 【新コード】history()の直近終値を使用、失敗時はinfoにフォールバック
# ============================================================
def get_current_price(stock):
    try:
        hist = stock.history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None


# ============================================================
# 【改良3】ROE・配当性向を複数年平均で計算
# 【旧コード】info.get("returnOnEquity") / info.get("payoutRatio") の単年値
# 【新コード】financials/balance_sheet/cashflowから複数年平均を計算
#             取得できない場合はNoneを返しinfoの単年値にフォールバック
# ============================================================
def get_multi_year_financials(stock, avg_years):
    try:
        financials = stock.financials
        balance    = stock.balance_sheet
        cashflow   = stock.cashflow

        if financials.empty or balance.empty:
            return None, None

        roe_list    = []
        payout_list = []
        cols = financials.columns[:avg_years]

        for col in cols:
            try:
                # 純利益
                net_income = None
                for label in ["Net Income", "Net Income Common Stockholders"]:
                    if label in financials.index:
                        net_income = financials.loc[label, col]
                        break

                # 自己資本
                equity = None
                for label in ["Stockholders Equity", "Total Stockholders Equity",
                              "Common Stock Equity"]:
                    if label in balance.index:
                        equity = balance.loc[label, col]
                        break

                # 配当総額
                div_paid = None
                if not cashflow.empty and col in cashflow.columns:
                    for label in ["Cash Dividends Paid", "Common Stock Dividend Paid"]:
                        if label in cashflow.index:
                            div_paid = abs(cashflow.loc[label, col])
                            break

                if net_income and equity and equity != 0:
                    roe_list.append(net_income / equity * 100)

                if div_paid and net_income and net_income > 0:
                    payout_list.append(div_paid / net_income * 100)

            except Exception:
                continue

        avg_roe    = float(np.mean(roe_list))    if roe_list    else None
        avg_payout = float(np.mean(payout_list)) if payout_list else None
        return avg_roe, avg_payout

    except Exception:
        return None, None


# ========= 配当安定性スコア（増配・減配耐性） =========
def calc_dividend_stability(yearly_div, start_year, end_year):
    years = list(range(start_year, end_year))
    divs  = [yearly_div.get(y, 0) for y in years]

    increase = decrease = maintain = 0
    for i in range(1, len(divs)):
        prev, curr = divs[i-1], divs[i]
        if prev <= 0:
            continue
        rate = (curr - prev) / prev
        if rate > 0.01:
            increase += 1
        elif rate < -0.01:
            decrease += 1
        else:
            maintain += 1

    total = increase + decrease + maintain
    if total == 0:
        return 5.0
    score = (increase * 1.0 + maintain * 0.5 - decrease * 2.0) / total
    return round(max(0.0, min(10.0, score * 10)), 2)


# ========= 連続値スコア（0〜10点） =========
def calc_score(value, low, high, reverse=False):
    if value is None:
        return 0.0
    if reverse:
        value, low, high = -value, -high, -low
    score = (value - low) / (high - low) * 10
    return round(max(0.0, min(10.0, score)), 2)


# ========= Final7選定 =========
def select_final7(df):
    """
    データ品質が「正常」の銘柄のみを対象に、
    総合点→安定性点→利回り点の順でソートし、
    同一業種最大2社の制限を適用して上位7社を返す。
    """
    df_valid  = df[df["データ品質"] == "正常"].copy()
    df_sorted = df_valid.sort_values(
        ["総合点", "安定性点", "利回り点"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    selected     = []
    sector_count = {}

    for _, row in df_sorted.iterrows():
        sector = row["業種"]
        if sector_count.get(sector, 0) < 2:
            selected.append(row)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) == 7:
            break

    return pd.DataFrame(selected).reset_index(drop=True)


# ========= JPXデータ読み込み =========
print("JPXデータ読み込み中...")
jpx     = pd.read_excel(JPX_FILE)
prime   = jpx[jpx["市場・商品区分"].str.contains("プライム", na=False)]
tickers = (prime["コード"].astype(str) + ".T").tolist()
print(f"対象銘柄数: {len(tickers)}")

rows = []

# ==============================
# メイン処理（全銘柄スキャン）
# ==============================
for t in tqdm(tickers, desc="スキャン中"):
    try:
        stock = yf.Ticker(t)
        info  = stock.info
        div   = stock.dividends

        if div.empty:
            continue

        df_div     = div.to_frame(name="dividend")
        df_div["year"] = df_div.index.year
        yearly_div = df_div.groupby("year")["dividend"].sum().to_dict()

        # 15年連続配当チェック（会計年度ベース）
        if not all(yearly_div.get(y, 0) > 0 for y in range(start_year, fiscal_year)):
            continue

        # --- 【改良1】株価を履歴データから自前取得 ---
        current_price = get_current_price(stock)
        if current_price is None:
            current_price = info.get("currentPrice") or 0

        # --- 【改良2】配当利回りを直近3年平均配当で計算 ---
        avg_div_years  = [fiscal_year - 1 - i for i in range(AVG_YEARS)]
        avg_div_values = [yearly_div[y] for y in avg_div_years if yearly_div.get(y, 0) > 0]
        last_year_div  = yearly_div.get(fiscal_year - 1, 0)  # 前年確定値（表示用）

        if avg_div_values and current_price > 0:
            avg_div        = sum(avg_div_values) / len(avg_div_values)
            dividend_yield = (avg_div / current_price) * 100
        elif current_price > 0 and last_year_div > 0:
            dividend_yield = (last_year_div / current_price) * 100
        else:
            dividend_yield = (info.get("dividendYield") or 0) * 100

        # --- 【改良4】異常値フラグ（continueで除外しない） ---
        data_quality  = "正常"
        quality_notes = []

        if dividend_yield <= 0 or dividend_yield > 15.0:
            data_quality = "要確認"
            quality_notes.append(f"利回り異常値({dividend_yield:.2f}%)")
            dividend_yield = 0.0

        if current_price <= 0:
            data_quality = "要確認"
            quality_notes.append("株価取得失敗")

        # --- 【改良3】ROE・配当性向を複数年平均で計算 ---
        avg_roe, avg_payout = get_multi_year_financials(stock, AVG_YEARS)

        roe    = avg_roe    if avg_roe    is not None else (info.get("returnOnEquity") or 0) * 100
        payout = avg_payout if avg_payout is not None else (info.get("payoutRatio")   or 0) * 100

        if roe > 100 or roe < -100:
            quality_notes.append(f"ROE異常値({roe:.1f}%)")
            roe = 0.0
        if payout > 200 or payout < 0:
            quality_notes.append(f"配当性向異常値({payout:.1f}%)")
            payout = 0.0

        growth = (info.get("revenueGrowth") or 0) * 100
        debt   =  info.get("debtToEquity")  or 0

        # スコアリング（各指標0〜10点）
        yield_score     = calc_score(dividend_yield, 1.0,   6.0)
        payout_score    = calc_score(payout,        20.0,  80.0, reverse=True)
        roe_score       = calc_score(roe,            3.0,  20.0)
        growth_score    = calc_score(growth,        -5.0,  15.0)
        debt_score      = calc_score(debt,           0.0, 300.0, reverse=True)
        stability_score = calc_dividend_stability(yearly_div, start_year, fiscal_year)

        # 総合点（最大90点）
        total_score = round(
            yield_score     * 2.0 +
            stability_score * 2.0 +
            payout_score    * 1.5 +
            roe_score       * 1.5 +
            growth_score    * 1.0 +
            debt_score      * 1.0,
            4
        )

        rows.append({
            "ティッカー":              t,
            "会社名":                  info.get("shortName", ""),
            "業種":                    info.get("sector", ""),
            "データ品質":              data_quality,
            "品質メモ":                " / ".join(quality_notes) if quality_notes else "",
            "株価":                    round(current_price, 0),
            "確定配当(前年)":          round(last_year_div, 2),
            "平均配当(3年)":           round(sum(avg_div_values)/len(avg_div_values), 2) if avg_div_values else 0,
            "利回り%(3年平均)":        round(dividend_yield, 2),
            "配当性向%(複数年平均)":   round(payout, 2),
            "ROE%(複数年平均)":        round(roe, 2),
            "売上成長率%":             round(growth, 2),
            "負債比率":                round(debt, 2),
            "利回り点":                yield_score,
            "安定性点":                stability_score,
            "配当性向点":              payout_score,
            "ROE点":                   roe_score,
            "成長点":                  growth_score,
            "財務点":                  debt_score,
            "総合点":                  total_score,
            "15年連続配当":            "YES",
            "選定年":                  fiscal_year,
        })

    except Exception:
        pass

# ==============================
# DataFrame化・選定
# ==============================
df_all = pd.DataFrame(rows)

if df_all.empty:
    print("⚠️  候補銘柄が0件です。JPXファイルやネット接続を確認してください。")
    exit()

df_final = select_final7(df_all)
df_check = df_all[df_all["データ品質"] == "要確認"].sort_values("総合点", ascending=False)

# ==============================
# Excel出力（3シート構成）
# ==============================
with pd.ExcelWriter(ANNUAL_RESULT_FILE, engine="openpyxl") as writer:
    df_final.to_excel(
        writer, index=False, sheet_name="Final7"
    )
    df_all[df_all["データ品質"] == "正常"].sort_values(
        "総合点", ascending=False
    ).to_excel(writer, index=False, sheet_name="AllCandidates")
    if not df_check.empty:
        df_check.to_excel(writer, index=False, sheet_name="DataCheck_要確認")

# ==============================
# Final7をJSONに保存
# ==============================
final7_data = {
    "selected_date": today.strftime("%Y-%m-%d"),
    "tickers": df_final["ティッカー"].tolist(),
    "details": df_final.to_dict(orient="records")
}
with open(FINAL7_JSON, "w", encoding="utf-8") as f:
    json.dump(final7_data, f, ensure_ascii=False, indent=2)

# ==============================
# 結果表示・メール通知
# ==============================
print("\n" + "=" * 60)
print(f"【年次選定完了】{today.strftime('%Y年%m月%d日')}")
print(f"全候補: {len(df_all)} 社  /  Final7: {len(df_final)} 社  /  要確認: {len(df_check)} 社")
print("=" * 60)
print(df_final[["ティッカー", "会社名", "利回り%(3年平均)", "安定性点", "総合点"]].to_string(index=False))

if not df_check.empty:
    print(f"\n⚠️  データ要確認銘柄: {len(df_check)} 社")
    print("→ annual_result.xlsx の「DataCheck_要確認」シートを確認してください")

print(f"\n結果保存: {ANNUAL_RESULT_FILE}")
print(f"監視用JSON: {FINAL7_JSON}")

body = (
    f"【年次選定完了】{today.strftime('%Y年%m月%d日')}\n\n"
    f"全候補: {len(df_all)} 社 / Final7: {len(df_final)} 社 / 要確認: {len(df_check)} 社\n\n"
    f"{df_final[['ティッカー','会社名','利回り%(3年平均)','総合点']].to_string(index=False)}\n\n"
    + (f"⚠️ データ要確認銘柄あり: {len(df_check)} 社\n"
       f"{df_check[['ティッカー','会社名','品質メモ']].to_string(index=False)}"
       if not df_check.empty else "")
)
send_alert(f"【配当システム】{fiscal_year}年度 年次選定結果", body)
print("\n完了。")


