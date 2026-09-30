# 競艇 統計予想アプリ

ボートレース公式の過去データ（番組表・競走成績）を使います。各艇の **1着確率** と **3連単120通りの確率** を統計モデルで推定します。

データ元: https://www.boatrace.jp/owpc/pc/extra/data/download.html

詳しい使い方と予測の手法は [docs/spec.html](docs/spec.html) にまとめています。

## 構成

| ファイル | 役割 |
| --- | --- |
| `boatrace/download.py` | B(番組表)/K(競走成績) の LZH を日付範囲で取得・解凍（1.5秒間隔、取得済みはスキップ） |
| `boatrace/weather.py` | 場ごとの1時間ごとの気温を Open-Meteo から取得 |
| `boatrace/venues.py` | 24場の水面の種類（湖・池／海／川・河口）・水質・緯度経度 |
| `boatrace/parsers.py` | Shift_JIS 固定長テキストを NFKC 正規化 + 正規表現で解析 |
| `boatrace/db.py` | SQLite（`entries` 出走表 / `results` 着順・ST / `races` 払戻） |
| `boatrace/features.py` | 特徴量。履歴系は **前日までのデータのみ** で計算（リーク防止） |
| `boatrace/model.py` | 1着の二値分類 → レース内で正規化。3連単は Plackett-Luce で計算。朝用・直前用の2モデル |
| `boatrace/backtest.py` | 時系列分割で評価（1着的中率・対数損失・3連単上位N点の的中率/回収率） |
| `boatrace/synthetic.py` | 動作確認用の **合成データ**（実データではない） |
| `app.py` | Streamlit 画面 |

## 使い方

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 実データ（直近3年）
python -m boatrace.cli download --start 2023-10-01 --end 2026-09-28 --workers 3   # 約2.5時間
python -m boatrace.cli ingest
python -m boatrace.cli weather --start 2023-10-01 --end 2026-09-30   # 気温 (Open-Meteo)
python -m boatrace.cli backtest --cutoff 2026-04-01
python -m boatrace.cli train
python -m boatrace.cli predict --date 2026-09-29 --venue 12 --race 1   # 朝の予想
python -m boatrace.cli predict --date 2026-09-29 --venue 12 --race 1 \
  --mode prerace --wind-dir 北西 --wind-speed 3 --wave 2 --courses 1,2,4,3,5,6   # 直前の予想
python -m streamlit run app.py
```

### Windows（PowerShell）

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # 実行できない場合: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
python -m streamlit run app.py      # "streamlit run" だと別の Python で起動され ModuleNotFoundError になることがある
```

- Python は 3.11〜3.13 を使ってください（`lhafile` の Windows 用ビルドがこの範囲のみ）。

ネットに接続できない環境では、`download` の代わりに `python -m boatrace.cli demo-data --days 180` を実行してください。合成データで一通り試せます。

## 注意

- 取得先として `www1.mbrace.or.jp`（公式データ）と `archive-api.open-meteo.com` / `api.open-meteo.com`（気温）への接続が必要です。
- **パーサー、気温の取得、24場の水面の分類は、実データでまだ検証していません。** 開発環境から公式サイトに接続できなかったためです。初回の `ingest` の後は、件数（1日あたり 出走数 ≈ レース数×6）を確認してください。
- 3年分は約2,200ファイルです。公式サーバーは1ファイルの応答に約11秒かかるため、1本ずつだと約7.5時間、`--workers 3` で約2.5時間かかります。途中で止めても、再実行すれば取得済みの日を飛ばして続きから取得します。
- 控除率は約25%です。確率予測が当たっても、回収率100%を超えるとは限りません。合成データでのバックテストでも、回収率は約75%（還元率そのもの）でした。
