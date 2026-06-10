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


---

## 🔮 今後の改良候補（TODO）

### 1. 業種別平均指標による相対評価
**背景：** 現在のIsolation Forestは全銘柄を同一基準で評価するため、業種特有の財務構造（自動車の高負債比率、IT株の高PBR等）が「異常」として検出されやすい。  
**方針：** J-Quants APIまたは東証公開データから業種別平均（PBR・ROE・負債比率等）を取得し、業種内相対評価に切り替える。  
**保留理由：** データソース調査・業種分類の統一・定期更新の仕組みが必要。来月以降のmonthly結果を見て必要性を再判断する。

### 2. select_final7()へのexclusion_candidate除外ロジック組み込み
**背景：** monthly_monitor.pyで2回連続異常検知時にfinal7.jsonへ`exclusion_candidate: true`を付与する仕組みは実装済み。ただし次回annual_select.py実行時の自動除外はまだ未実装。  
**方針：** `select_final7()`の冒頭でfinal7.jsonを読み込み、`exclusion_candidate: true`の銘柄をスクリーニング対象から除外する。  
**注意：** 除外は自動判断せず、メール通知と目視確認を経てから実施することを推奨。

### 3. anomaly閾値（ANOMALY_SCORE_THRESH）の再検討
**背景：** 現在の閾値`-0.50`に対し、初回monthly実行（2026-06-10）でSUBARU(-0.5139)・DAI NIPPON TORYO(-0.5341)・CYBOZU(-0.6092)が異常判定。SUBARUは負債比率13.82と自動車業種の構造的特性が影響している可能性あり。  
**方針：** 2026年7月末のmonthly結果を確認後、閾値を`-0.55`に変更するか判断する。変更箇所は`monthly_monitor.py`の`ANOMALY_SCORE_THRESH`のみ。

### 4. feature_importance.pngの日本語フォント
**背景：** Linux機（omni-2026）でIPAGothicフォント未インストールのためグラフラベルが文字化けする場合がある。  
**対処：**
```bash
sudo apt install fonts-ipafont -y && fc-cache -fv
```
