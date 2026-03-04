# 長期安定配当株 自動抽出システム
## セットアップガイド

---

## 📁 ファイル構成

```
dividend-system/
├── annual_select.py      # 【年1回】全銘柄スキャン・Final7選定
├── monthly_monitor.py    # 【月末】7社監視・異常アラート
├── email_config.py       # メール設定（自分で作成）
├── data_j.xlsx           # JPX上場銘柄ファイル（JPXサイトからDL）
└── output/               # 結果ファイル（自動生成）
    ├── annual_result.xlsx # 年次選定結果
    ├── final7.json        # 月末監視用・選定結果
    └── monitor_log.xlsx   # 月次監視ログ（累積）
```

---

## 🖥️ 動作環境

| 項目 | 推奨 |
|------|------|
| OS | Windows 10/11 または macOS |
| Python | 3.10 以上 |
| メモリ | 4GB以上（全銘柄スキャン時） |
| ネット | 必須（yfinanceがYahoo! Financeにアクセス） |

---

## ⚙️ セットアップ手順

### 1. Pythonのインストール確認
```bash
python --version
# Python 3.10.x などと表示されればOK
```
※ 入っていない場合は https://www.python.org からインストール

---

### 2. 必要ライブラリのインストール
```bash
pip install yfinance pandas openpyxl tqdm
```

| ライブラリ | 用途 |
|-----------|------|
| yfinance | Yahoo! Financeから株価・配当データ取得 |
| pandas | データ集計・DataFrame処理 |
| openpyxl | Excelファイルの読み書き |
| tqdm | プログレスバー表示 |

---

### 3. JPX上場銘柄ファイルの取得
1. https://www.jpx.co.jp/markets/statistics-equities/misc/01.html を開く
2. 「上場銘柄一覧」の **Excelファイル** をダウンロード
3. ダウンロードした `data_j.xlsx` をプロジェクトフォルダに置く

---

### 4. email_config.py の作成（メール通知を使う場合）
プロジェクトフォルダに `email_config.py` を新規作成して以下を記入：

```python
SMTP_SERVER  = "smtp.gmail.com"   # Gmailの場合
SMTP_PORT    = 587
EMAIL_ADDRESS = "your_address@gmail.com"   # 送信元アドレス
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"     # Gmailアプリパスワード
TO_EMAIL     = "your_address@gmail.com"    # 送信先（自分宛でOK）
```

> **Gmailアプリパスワードの取得方法**
> 1. Googleアカウント → セキュリティ → 2段階認証をONにする
> 2. 「アプリパスワード」を検索 → 新しいパスワードを生成
> 3. 生成された16桁をEMAIL_PASSWORDに貼り付ける

※ メール通知が不要な場合は `email_config.py` を作らなくてもOK。
　 その場合はメール送信がスキップされ、コンソールに結果が表示されます。

---

## 🚀 実行方法

### 年1回（4月1日推奨 ※休日の場合は翌営業日）
```bash
python annual_select.py
```
- 全プライム銘柄をスキャン（2〜3時間かかる場合あります）
- `output/annual_result.xlsx` と `output/final7.json` が生成される

### 月末（毎月末の営業日）
```bash
python monthly_monitor.py
```
- 7社のみチェックするので **数分で完了**
- 異常がなければメールは送信されない
- 異常がある場合のみアラートメールが届く

---

## 📊 アラート条件（monthly_monitor.py）

| 条件 | 閾値 | 意味 |
|------|------|------|
| 配当大幅減 | 前年比 -20% 以上 | 減配が大きい |
| 株価大幅下落 | 選定時比 -30% 以上 | 財務悪化の可能性 |
| 配当停止 | 配当データなし | 無配転落 |
| 上場廃止 | 株価データなし | 上場廃止・取引停止 |

> 閾値は `monthly_monitor.py` の上部の設定値で変更できます。

---

## 📅 推奨運用カレンダー

```
4月1日  → annual_select.py を実行（年次選定・株購入判断）
毎月末  → monthly_monitor.py を実行（月次監視）
アラート → メールを確認してアクションを検討
翌年4月 → annual_select.py を再実行（リスト更新）
```
