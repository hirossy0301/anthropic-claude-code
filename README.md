# 競艇 統計予想アプリ

ボートレース公式の過去データ（番組表・競走成績）を使います。各艇の **1着確率** と **3連単120通りの確率** を統計モデルで推定します。

データ元: https://www.boatrace.jp/owpc/pc/extra/data/download.html

詳しい使い方と予測の手法は [docs/spec.html](docs/spec.html) にまとめています。

## 構成

| ファイル | 役割 |
| --- | --- |
| `boatrace/download.py` | B(番組表)/K(競走成績) の LZH を日付範囲で取得・解凍（1.5秒間隔、取得済みはスキップ） |
| `boatrace/parsers.py` | Shift_JIS 固定長テキストを NFKC 正規化 + 正規表現で解析 |
| `boatrace/db.py` | SQLite（`entries` 出走表 / `results` 着順・ST / `races` 払戻） |
| `boatrace/features.py` | 特徴量。履歴系は **前日までのデータのみ** で計算（リーク防止） |
| `boatrace/model.py` | 1着の二値分類 → レース内で正規化。3連単は Plackett-Luce で計算 |
| `boatrace/backtest.py` | 時系列分割で評価（1着的中率・対数損失・3連単上位N点の的中率/回収率） |
| `boatrace/synthetic.py` | 動作確認用の **合成データ**（実データではない） |
| `app.py` | Streamlit 画面 |

## 使い方

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 実データ（直近3年）
python -m boatrace.cli download --start 2023-10-01 --end 2026-09-28
python -m boatrace.cli ingest
python -m boatrace.cli backtest --cutoff 2026-04-01
python -m boatrace.cli train
python -m boatrace.cli predict --date 2026-09-29 --venue 12 --race 1
streamlit run app.py
```

ネットに接続できない環境では、`download` の代わりに `python -m boatrace.cli demo-data --days 180` を実行してください。合成データで一通り試せます。

## 注意

- **パーサーは実ファイルでまだ検証していません。** 開発環境から公式サイトに接続できなかったためです。初回の `ingest` の後は、件数（1日あたり 出走数 ≈ レース数×6）を確認してください。
- 3年分の取得には、1.5秒間隔で約2,200リクエスト、1時間程度かかります。
- 控除率は約25%です。確率予測が当たっても、回収率100%を超えるとは限りません。合成データでのバックテストでも、回収率は約75%（還元率そのもの）でした。
