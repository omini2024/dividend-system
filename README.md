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
    ├── annual_result.xlsx      # 年次選定結果（4シート構成）
    │     ├── Final7              # 選定7社（異常銘柄は黄色警告表示）
    │     ├── AllCandidates       # 全正常候補銘柄
    │     ├── DataCheck_要確認    # データ品質要確認銘柄
    │     └── AnomalyReport       # Isolation Forest異常検知レポート
    ├── final7.json             # 月次監視用・選定結果
    │                             （連続異常時はexclusion_candidate付与）
    ├── monitor_log.xlsx        # 月次監視ログ（累積）
    └── feature_importance.png  # 特徴量重要度グラフ（annual実行時生成）
```

---

## 🖥️ 動作環境

| 項目 | 推奨 |
|------|------|
| OS | Ubuntu 24.04 / macOS / Windows 10以上 |
| Python | 3.10 以上 |
| メモリ | 4GB以上（全銘柄スキャン時） |
| ネット | 必須（yfinance・J-Quants APIアクセス） |

---

## ⚙️ セットアップ手順

### 1. Pythonのインストール確認
```bash
python3 --version
# Python 3.10.x などと表示されればOK
```
※ 入っていない場合は https://www.python.org からインストール

---

### 2. 必要ライブラリのインストール
```bash
pip install yfinance pandas openpyxl tqdm scikit-learn matplotlib jquantsapi
```

Ubuntu / Linux環境では：
```bash
pip install yfinance pandas openpyxl tqdm scikit-learn matplotlib jquantsapi --break-system-packages
```

| ライブラリ | 用途 |
|-----------|------|
| yfinance | Yahoo! Financeから株価・配当データ取得 |
| pandas | データ集計・DataFrame処理 |
| openpyxl | Excelファイルの読み書き・セル色付け |
| tqdm | プログレスバー表示 |
| scikit-learn | Isolation Forest異常検知・特徴量重要度分析 |
| matplotlib | 特徴量重要度グラフ出力 |
| jquantsapi | J-Quants APIによる株式分割検出（公式データ） |

日本語フォント（feature_importance.png文字化け防止）：
```bash
sudo apt install fonts-ipafont -y && fc-cache -fv
```

---

### 3. JPX上場銘柄ファイルの取得
1. https://www.jpx.co.jp/markets/statistics-equities/misc/01.html を開く
2. 「上場銘柄一覧」の **Excelファイル** をダウンロード
3. ダウンロードした `data_j.xlsx` をプロジェクトフォルダに置く

---

### 4. J-Quants APIキーの設定
株式分割検出にJ-Quants API（Lightプラン・月額1,650円）を使用。

1. https://jpx-jquants.com にアクセスしてアカウント登録
2. APIキーを取得
3. `annual_select.py` 冒頭の `JQUANTS_API_KEY` に設定：

```python
JQUANTS_API_KEY = "your_api_key_here"
```

> J-Quantsが取得できない場合はyfinanceにフォールバックします。

---

### 5. email_config.py の作成（メール通知を使う場合）
プロジェクトフォルダに `email_config.py` を新規作成して以下を記入：

```python
SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 587
EMAIL_ADDRESS = "your_address@gmail.com"
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"   # Gmailアプリパスワード
TO_EMAIL      = "your_address@gmail.com"
```

> **Gmailアプリパスワードの取得方法**
> 1. Googleアカウント → セキュリティ → 2段階認証をON
> 2. 「アプリパスワード」を検索 → 新しいパスワードを生成
> 3. 生成された16桁を `EMAIL_PASSWORD` に貼り付ける

※ 作成しない場合はメール送信がスキップされ、コンソールに結果が表示されます。

---

## 🚀 実行方法

### 年1回（4月1日推奨 ※休日の場合は翌営業日）
```bash
python3 annual_select.py
```
- 全プライム銘柄をスキャン（2〜3時間かかる場合あります）
- `output/annual_result.xlsx`（4シート）と `output/final7.json` が生成される
- `output/feature_importance.png` に特徴量重要度グラフを出力
- Final7に異常検知銘柄がある場合、Excelで**黄色背景・赤太字**で警告表示

### 月末（毎月末の営業日）
```bash
python3 monthly_monitor.py
```
- 7社のみチェックするので **数分で完了**
- 月次Isolation Forestスコアを `monitor_log.xlsx` に記録
- 異常なし時もスコアサマリーをメール送信（毎月スコアを確認可能）
- 異常がある場合はアラートメールを送信

---

## 📊 アラート条件（monthly_monitor.py）

| 条件 | 閾値 | 意味 |
|------|------|------|
| 配当大幅減 | 前年比 -20% 以上 | 減配が大きい |
| 株価大幅下落 | 選定時比 -30% 以上 | 財務悪化の可能性 |
| 配当停止 | 配当データなし | 無配転落 |
| 上場廃止 | 株価データなし | 上場廃止・取引停止 |

> 閾値は `monthly_monitor.py` 上部の設定値で変更できます。

---

## 🤖 機械学習機能

### Isolation Forest 異常検知（annual・monthly共通）

スコアリングには以下の特徴量を使用：

| 特徴量 | annual | monthly |
|--------|--------|---------|
| 利回り%（3年平均） | ✅ | ✅ |
| 配当性向%（複数年平均） | ✅ | ✅ |
| ROE%（複数年平均） | ✅ | ✅ |
| DOE% | ✅ | ✅ |
| PBR | ✅ | ✅ |
| 理論利回り% | ✅ | ✅ |
| 実質PBR倍率 | ✅ | ✅ |
| 売上成長率% | ✅ | ✅ |
| 負債比率 | ✅ | ✅ |
| 株価変化率%（選定時比） | — | ✅ |
| 配当変化率%（前年比） | — | ✅ |

**anomaly_score**：低いほど異常度が高い（-0.5以下が警戒ライン）  
**anomaly_flag**：-1 = 異常、1 = 正常

### 連続異常による除外候補フラグ

```
monthly_monitor.py 実行
    ↓
前回ログと比較：2回連続 anomaly_score ≤ -0.50
    ↓ YES
final7.json に exclusion_candidate: true を付与
    ↓
メールで「次回年次選定除外候補」として通知
    ↓
翌年の annual_select.py 実行時に除外対象として扱う（手動確認推奨）
```

> 連続異常が解消された場合（スコアが回復）、フラグは自動でリセットされます。

### 特徴量重要度グラフ（feature_importance.png）

annual実行時に `output/feature_importance.png` を生成。  
総合点上位30%と下位30%を疑似ラベルとしてRandomForestで重要度を算出。  
現在の固定重みが実データに照らして妥当かを確認するために使用する。

---

## 📅 推奨運用カレンダー

```
4月1日        → annual_select.py を実行（年次選定・株購入判断）
               → feature_importance.png で重みの妥当性を確認
               → Final7の黄色警告銘柄を目視確認
毎月末        → monthly_monitor.py を実行（月次監視）
               → メールのスコアサマリーで異常度の推移を確認
アラートあり  → メールを確認してアクションを検討
連続異常通知  → exclusion_candidate銘柄を翌年選定から除外するか判断
翌年4月       → annual_select.py を再実行（リスト更新）
```

---

## 🔧 主要設定値一覧

### annual_select.py

| 変数 | デフォルト | 内容 |
|------|-----------|------|
| `YEARS` | 15 | 連続配当チェック年数 |
| `AVG_YEARS` | 3 | 利回り・ROE・配当性向の平均年数 |
| `SPLIT_WATCH_MONTHS` | 3 | 株式分割警戒月数 |
| `ANOMALY_CONTAMINATION` | 0.05 | 異常の想定割合（5%） |
| `EXPORT_FEATURE_IMPORTANCE` | True | 特徴量重要度PNG出力フラグ |

### monthly_monitor.py

| 変数 | デフォルト | 内容 |
|------|-----------|------|
| `DIV_DROP_THRESHOLD` | -0.20 | 配当減アラート閾値（-20%） |
| `PRICE_DROP_THRESHOLD` | -0.30 | 株価下落アラート閾値（-30%） |
| `ANOMALY_SCORE_THRESH` | -0.50 | 連続異常判定スコア閾値 |
