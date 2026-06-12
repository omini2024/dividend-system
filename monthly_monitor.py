# ==============================
# 【月末実行】長期安定配当株 月次監視スクリプト
# 実行タイミング：毎月末（例：月末最終営業日）
# 役割：Final7の7社のみをチェックし、異常があればメール通知
#        通常時は何もしない（ランキング再計算は行わない）
#
# 【改良】
# 追加1: Isolation Forestによる月次異常スコアリング
#         annualと同じ特徴量 + 株価変化率% + 配当変化率% を使用
# 追加2: 2回連続anomaly_score≤-0.5でexclusion_candidate=trueをfinal7.jsonに付与
# ==============================

import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import json
import os

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import trade_timing
    TRADE_TIMING_ENABLED = True
except ImportError:
    TRADE_TIMING_ENABLED = False
    print("⚠️  trade_timing が見つかりません。売買タイミング判定はスキップします。")

try:
    from email_config import SMTP_SERVER, SMTP_PORT, EMAIL_ADDRESS, EMAIL_PASSWORD, TO_EMAIL
    MAIL_ENABLED = True
except ImportError:
    MAIL_ENABLED = False
    print("⚠️  email_config が見つかりません。メール通知はスキップします。")

# ========= 設定 =========
OUTPUT_DIR     = "output"
FINAL7_JSON    = os.path.join(OUTPUT_DIR, "final7.json")
MONITOR_LOG    = os.path.join(OUTPUT_DIR, "monitor_log.xlsx")

# ========= アラート閾値 =========
DIV_DROP_THRESHOLD    = -0.20   # 配当が前年比20%以上減でアラート
PRICE_DROP_THRESHOLD  = -0.30   # 株価が選定時から30%以上下落でアラート
ANOMALY_SCORE_THRESH  = -0.50   # 2回連続でこの値以下 → exclusion_candidate
ANOMALY_FEATURES_BASE = [
    "利回り%(3年平均)", "配当性向%(複数年平均)", "ROE%(複数年平均)",
    "DOE%", "PBR", "理論利回り%", "実質PBR倍率",
    "売上成長率%", "負債比率",
]
ANOMALY_FEATURES_MONTHLY = ANOMALY_FEATURES_BASE + ["株価変化率%", "配当変化率%"]


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
alerts   = []
ok_list  = []
log_rows = []

for t in tickers:
    base   = details[t]
    issues = []

    try:
        stock = yf.Ticker(t)
        info  = stock.info

        # ----- 上場廃止チェック -----
        if not info or info.get("regularMarketPrice") is None:
            issues.append("⚠️ 上場廃止または取引停止の可能性")

        # ----- 配当データ取得 -----
        div = stock.dividends
        current_year_div  = 0
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
        div_change_pct = 0.0
        if base_div > 0 and last_year_div_now > 0:
            div_change_pct = (last_year_div_now - base_div) / base_div
            if div_change_pct <= DIV_DROP_THRESHOLD:
                issues.append(
                    f"⚠️ 配当大幅減: {base_div:.2f}→{last_year_div_now:.2f}"
                    f"（{div_change_pct*100:.1f}%）"
                )

        # ----- 株価大幅下落チェック -----
        current_price = info.get("currentPrice") or 0
        base_price    = base.get("株価", 0)
        price_change_pct = 0.0
        if base_price > 0 and current_price > 0:
            price_change_pct = (current_price - base_price) / base_price
            if price_change_pct <= PRICE_DROP_THRESHOLD:
                issues.append(
                    f"⚠️ 株価大幅下落: {base_price:.0f}→{current_price:.0f}"
                    f"（{price_change_pct*100:.1f}%）"
                )

        # ----- 月次Isolation Forest用の特徴量行を構築 -----
        monthly_row = {}
        for col in ANOMALY_FEATURES_BASE:
            monthly_row[col] = base.get(col, 0) or 0
        monthly_row["株価変化率%"]  = round(price_change_pct * 100, 2)
        monthly_row["配当変化率%"]  = round(div_change_pct  * 100, 2)

        # ----- 結果まとめ -----
        status = "✅ 異常なし" if not issues else "🔴 要確認"
        entry = {
            "ティッカー":     t,
            "会社名":         base.get("会社名", ""),
            "選定時株価":     base_price,
            "現在株価":       round(current_price, 0),
            "株価変化率%":    round(price_change_pct * 100, 1),
            "選定時配当":     base_div,
            "直近確定配当":   round(last_year_div_now, 2),
            "配当変化率%":    round(div_change_pct * 100, 1),
            "ステータス":     status,
            "問題内容":       " / ".join(issues) if issues else "",
            "確認日":         today.strftime("%Y-%m-%d"),
            # Isolation Forest用（後で一括計算して上書き）
            "monthly_anomaly_score": 0.0,
            "monthly_anomaly_flag":  0,
        }
        # 特徴量も一時保存
        entry["_monthly_row"] = monthly_row

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
            "ティッカー": t, "会社名": base.get("会社名", ""),
            "ステータス": "❌ エラー", "問題内容": msg,
            "確認日": today.strftime("%Y-%m-%d"),
            "monthly_anomaly_score": 0.0,
            "monthly_anomaly_flag":  0,
            "_monthly_row": {},
        })


# ==============================
# 【追加1】月次 Isolation Forest スコアリング
# 7社は少なすぎるため、annualのAllCandidatesデータと結合してスコア計算
# AllCandidatesがない場合は7社のみで計算（精度は低下）
# ==============================
print("\n月次異常スコアを計算中...")

ANNUAL_RESULT_FILE = os.path.join(OUTPUT_DIR, "annual_result.xlsx")
background_df = pd.DataFrame()
if os.path.exists(ANNUAL_RESULT_FILE):
    try:
        background_df = pd.read_excel(ANNUAL_RESULT_FILE, sheet_name="AllCandidates")
    except Exception:
        pass

# 月次行をDataFrameに
monthly_feature_rows = [r["_monthly_row"] for r in log_rows if r.get("_monthly_row")]
monthly_tickers      = [r["ティッカー"]    for r in log_rows if r.get("_monthly_row")]

if monthly_feature_rows:
    df_monthly = pd.DataFrame(monthly_feature_rows, index=monthly_tickers)

    # 背景データと結合（月次固有列は背景データには存在しないため0埋め）
    if not background_df.empty:
        use_base = [c for c in ANOMALY_FEATURES_BASE if c in background_df.columns]
        bg = background_df[use_base].copy().fillna(0)
        bg["株価変化率%"] = 0.0
        bg["配当変化率%"] = 0.0
        combined = pd.concat([bg, df_monthly[ANOMALY_FEATURES_MONTHLY].fillna(0)], ignore_index=False)
    else:
        combined = df_monthly[ANOMALY_FEATURES_MONTHLY].fillna(0)

    use_cols = [c for c in ANOMALY_FEATURES_MONTHLY if c in combined.columns]
    X = combined[use_cols]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=300, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)

    # 月次7銘柄のみスコアを取得（末尾len(df_monthly)行）
    n = len(df_monthly)
    scores_monthly = iso.score_samples(X_scaled[-n:])
    flags_monthly  = iso.predict(X_scaled[-n:])

    for i, t in enumerate(monthly_tickers):
        for r in log_rows:
            if r["ティッカー"] == t:
                r["monthly_anomaly_score"] = round(float(scores_monthly[i]), 4)
                r["monthly_anomaly_flag"]  = int(flags_monthly[i])
                break

    print(f"✅  月次異常スコア計算完了")


# ==============================
# 【追加2】連続異常チェック → exclusion_candidate付与
# 前回ログのmonthly_anomaly_scoreと比較して2回連続≤閾値ならフラグ
# ==============================
exclusion_candidates = []

if os.path.exists(MONITOR_LOG):
    try:
        df_prev_log = pd.read_excel(MONITOR_LOG)
        # 各ティッカーの直近1回分のスコアを取得
        if "monthly_anomaly_score" in df_prev_log.columns and "確認日" in df_prev_log.columns:
            df_prev_log["確認日"] = pd.to_datetime(df_prev_log["確認日"])
            prev_latest = (
                df_prev_log.sort_values("確認日")
                .groupby("ティッカー")
                .last()["monthly_anomaly_score"]
                .to_dict()
            )
            for r in log_rows:
                t = r["ティッカー"]
                prev_score = prev_latest.get(t, 0.0)
                curr_score = r.get("monthly_anomaly_score", 0.0)
                if prev_score <= ANOMALY_SCORE_THRESH and curr_score <= ANOMALY_SCORE_THRESH:
                    exclusion_candidates.append(t)
                    r["問題内容"] = (r.get("問題内容") or "") + \
                        f" / ⚠️連続異常検知(score:{curr_score:.4f})"
                    r["ステータス"] = "🔴 要確認"
    except Exception as e:
        print(f"⚠️  前回ログ読み込みエラー: {e}")

# final7.jsonにexclusion_candidateフラグを付与
if exclusion_candidates:
    print(f"\n🚨 連続異常検知 → exclusion_candidate付与: {exclusion_candidates}")
    for d in final7_data["details"]:
        if d["ティッカー"] in exclusion_candidates:
            d["exclusion_candidate"] = True
            d["exclusion_reason"] = f"monthly連続異常 ({today.strftime('%Y-%m-%d')})"
        else:
            d.pop("exclusion_candidate", None)
            d.pop("exclusion_reason", None)
    with open(FINAL7_JSON, "w", encoding="utf-8") as f:
        json.dump(final7_data, f, ensure_ascii=False, indent=2)
    print(f"✅  final7.json更新: exclusion_candidate={exclusion_candidates}")
else:
    # 前回のフラグをリセット（連続でなくなった場合）
    changed = False
    for d in final7_data["details"]:
        if "exclusion_candidate" in d:
            d.pop("exclusion_candidate", None)
            d.pop("exclusion_reason", None)
            changed = True
    if changed:
        with open(FINAL7_JSON, "w", encoding="utf-8") as f:
            json.dump(final7_data, f, ensure_ascii=False, indent=2)


# ==============================
# ログ保存（_monthly_row列を除いてExcelに追記）
# ==============================
for r in log_rows:
    r.pop("_monthly_row", None)

df_log = pd.DataFrame(log_rows)

if os.path.exists(MONITOR_LOG):
    df_existing = pd.read_excel(MONITOR_LOG)
    df_log = pd.concat([df_existing, df_log], ignore_index=True)

df_log.to_excel(MONITOR_LOG, index=False)


# ==============================
# メール通知
# ==============================
print(f"\n{'='*60}")

# ==============================
# 売買タイミング判定
# ==============================
timing_changed  = False
timing_report   = ""
exclusion_timing_report = ""
if TRADE_TIMING_ENABLED:
    # 連続異常検知銘柄の緊急売りタイミング（exclusion_candidate 確定月に即時実行）
    if exclusion_candidates:
        print("\n連続異常検知銘柄の売りタイミングを判定中...")
        exclusion_timing_report = trade_timing.analyze_exclusion_timing(
            exclusion_candidates, details
        )
        print(exclusion_timing_report)

    # Final7 入れ替わり時の売り買いタイミング（年次選定変更を検出）
    print("\n売買タイミング判定を実行中...")
    timing_changed, timing_report = trade_timing.analyze_trade_timing()
    print(timing_report)

# anomaly scoreサマリー
score_lines = ["■ 月次異常スコア（低いほど異常度高）"]
for r in log_rows:
    flag_str = "⚠️異常" if r.get("monthly_anomaly_flag") == -1 else "正常"
    score_lines.append(
        f"  {r['ティッカー']} {r.get('会社名',''):<25} "
        f"score:{r.get('monthly_anomaly_score', 0.0):>8.4f}  [{flag_str}]"
    )
score_summary = "\n".join(score_lines)
print(score_summary)

if exclusion_candidates:
    excl_names = [details[t].get("会社名", t) for t in exclusion_candidates]
    print(f"\n🚨 次回年次選定除外候補: {', '.join(excl_names)}")

if alerts or exclusion_candidates:
    print(f"\n🔴 要確認: {len(alerts)} 社  /  ✅ 異常なし: {len(ok_list)} 社")
    lines = [f"【月次監視】{today.strftime('%Y年%m月%d日')} - 要確認あり\n"]
    for t, name, issues in alerts:
        lines.append(f"■ {t} {name}")
        for iss in issues:
            lines.append(f"  {iss}")
        lines.append("")
    if exclusion_candidates:
        lines.append("=" * 40)
        lines.append("🚨 次回年次選定 除外候補（連続異常検知）")
        for t in exclusion_candidates:
            lines.append(f"  {t} {details[t].get('会社名', '')}")
        lines.append("")
    lines.append(score_summary)
    if exclusion_timing_report:
        lines.append(exclusion_timing_report)
    if timing_changed:
        lines.append(timing_report)
    lines.append(f"\n監視ログ: {MONITOR_LOG}")
    body = "\n".join(lines)
    send_alert(f"【配当システム】{today.strftime('%Y年%m月')} 要確認銘柄あり", body)
    print("⚡ アラートメール送信しました。")
else:
    print(f"✅ 全{len(ok_list)}社 異常なし。")
    # 異常なしでもスコアサマリーはメール送信
    body = (
        f"【月次監視】{today.strftime('%Y年%m月%d日')} - 全社異常なし\n\n"
        + score_summary
        + (("\n" + exclusion_timing_report) if exclusion_timing_report else "")
        + (("\n" + timing_report) if timing_changed else "")
        + f"\n\n監視ログ: {MONITOR_LOG}"
    )
    send_alert(f"【配当システム】{today.strftime('%Y年%m月')} 月次監視完了", body)

print(f"ログ保存: {MONITOR_LOG}")
print("完了。")
