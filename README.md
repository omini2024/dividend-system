# Dividend System (配当成長投資スクリーナー)

日本株の中から  
「長期・増配・減配耐性の高い企業」を自動抽出するツールです。

---

## 📌 特徴

- プライム市場のみ対象
- 15年連続配当企業
- 配当成長率（CAGR）計算
- 減配回数カウント
- 財務・収益性スコアリング
- 業種最大2社ルール
- Final7銘柄を自動選出

---

## 🛠 必要環境

- Python 3.11 以上
- pip

---

## 📦 セットアップ

```bash
python3 -m venv venv
source venv/bin/activate
pip install yfinance pandas tqdm openpyxl
