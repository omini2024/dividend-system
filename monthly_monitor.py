# ==============================
# 【月末実行】長期安定配当株 月次監視スクリプト
# 実行タイミング：毎月末（例：月末最終営業日）
# 役割：Final7の7社のみをチェックし、異常があればメール通知
#        通常時は何もしない（ランキング再計算は行わない）
# ==============================

import yfinance as yf
from datetime import datetime
import pandas as pd
import json
import os

try:
    from email_config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL
    MAIL_ENABLED = True
except ImportError:
    MAIL_ENABLED = False
    print("⚠️  email_config が見つかりません。メール通知はスキップします。")

# ========= 設定 =========
OUTPUT_DIR     = "output"
FINAL7_JSON    = os.path.join(OUTPUT_DIR, "final7.json")    # 年次選定スクリプトが生成
MONITOR_LOG    = os.path.join(OUTPUT_DIR, "monitor_log.xlsx")

# ========= アラート閾値 =========
DIV_DROP_THRESHOLD  = -0.20   # 配当が前年比20%以上減でアラート
PRICE_DROP_THRESHOLD = -0.30  # 株価が選定時から30%以上下落でアラート


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


# ========= Final7 JSONを読み込み =========
if not os.path.exists(FINAL7_JSON):
    print("❌ final7.json が見つかりません。先に annual_select.py を実行してください。")
    exit()

with open(FINAL7_JSON, "r", encoding="utf-8") as f:
    final7_data = json.load(f)

selected_date = final7_data["selected_date"]
tickers       = final7_data["tickers"]
details       = {d["ティッカー"]: d for d in final7_data["details"]}

today = datetime.today()
print(f"\n{'='*60}")
print(f"【月次監視】{today.strftime('%Y年%m月%d日')}")
print(f"選定日: {selected_date}  /  監視銘柄: {len(tickers)} 社")
print(f"{'='*60}")


# ==============================
# 各銘柄チェック
# ==============================
alerts   = []   # アクションが必要な銘柄
ok_list  = []   # 問題なしの銘柄
log_rows = []   # ログ用

for t in tickers:
    base = details[t]
    issues = []

    try:
        stock = yf.Ticker(t)
        info  = stock.info

        # ----- 上場廃止チェック -----
        if not info or info.get("regularMarketPrice") is None:
            issues.append("⚠️ 上場廃止または取引停止の可能性")

        # ----- 配当停止チェック -----
        div = stock.dividends
        current_year_div = 0
        last_year_div_now = 0

        if not div.empty:
            df_div = div.to_frame(name="dividend")
            df_div["year"] = df_div.index.year
            yearly_div = df_div.groupby("year")["dividend"].sum().to_dict()
            current_year_div  = yearly_div.get(today.year, 0)
            last_year_div_now = yearly_div.get(today.year - 1, 0)
        else:
            issues.append("⚠️ 配当データなし（配当停止の可能性）")

        # ----- 配当大幅減チェック -----
        base_div = base.get("確定配当(前年)", 0)
        if base_div > 0 and last_year_div_now > 0:
            div_change = (last_year_div_now - base_div) / base_div
            if div_change <= DIV_DROP_THRESHOLD:
                issues.append(
                    f"⚠️ 配当大幅減: {base_div:.2f}→{last_year_div_now:.2f}"
                    f"（{div_change*100:.1f}%）"
                )

        # ----- 株価大幅下落チェック -----
        current_price = info.get("currentPrice") or 0
        base_price    = base.get("株価", 0)
        price_change  = 0.0
        if base_price > 0 and current_price > 0:
            price_change = (current_price - base_price) / base_price
            if price_change <= PRICE_DROP_THRESHOLD:
                issues.append(
                    f"⚠️ 株価大幅下落: {base_price:.0f}→{current_price:.0f}"
                    f"（{price_change*100:.1f}%）"
                )

        # ----- 結果まとめ -----
        status = "✅ 異常なし" if not issues else "🔴 要確認"
        entry = {
            "ティッカー":     t,
            "会社名":         base.get("会社名", ""),
            "選定時株価":     base_price,
            "現在株価":       round(current_price, 0),
            "株価変化%":      round(price_change * 100, 1),
            "選定時配当":     base_div,
            "直近確定配当":   round(last_year_div_now, 2),
            "ステータス":     status,
            "問題内容":       " / ".join(issues) if issues else "",
            "確認日":         today.strftime("%Y-%m-%d"),
        }
        log_rows.append(entry)

        if issues:
            alerts.append((t, base.get("会社名", ""), issues))
            print(f"🔴 {t} {base.get('会社名','')} → {' / '.join(issues)}")
        else:
            ok_list.append(t)
            print(f"✅ {t} {base.get('会社名','')} → 異常なし")

    except Exception as e:
        msg = f"❌ データ取得エラー: {e}"
        print(f"{t}: {msg}")
        alerts.append((t, base.get("会社名", ""), [msg]))
        log_rows.append({
            "ティッカー": t, "会社名": base.get("会社名",""),
            "ステータス": "❌ エラー", "問題内容": msg,
            "確認日": today.strftime("%Y-%m-%d"),
        })

# ==============================
# ログ保存（Excelに追記）
# ==============================
df_log = pd.DataFrame(log_rows)

if os.path.exists(MONITOR_LOG):
    df_existing = pd.read_excel(MONITOR_LOG)
    df_log = pd.concat([df_existing, df_log], ignore_index=True)

df_log.to_excel(MONITOR_LOG, index=False)

# ==============================
# メール通知（異常がある時のみ送信）
# ==============================
print(f"\n{'='*60}")
if alerts:
    print(f"🔴 要確認: {len(alerts)} 社  /  ✅ 異常なし: {len(ok_list)} 社")
    lines = [f"【月次監視】{today.strftime('%Y年%m月%d日')} - 要確認あり\n"]
    for t, name, issues in alerts:
        lines.append(f"■ {t} {name}")
        for iss in issues:
            lines.append(f"  {iss}")
        lines.append("")
    lines.append(f"監視ログ: {MONITOR_LOG}")
    body = "\n".join(lines)
    send_alert(f"【配当システム】{today.strftime('%Y年%m月')} 要確認銘柄あり", body)
    print("⚡ アラートメール送信しました。")
else:
    print(f"✅ 全{len(ok_list)}社 異常なし。メール通知はスキップします。")

print(f"ログ保存: {MONITOR_LOG}")
print("完了。")
