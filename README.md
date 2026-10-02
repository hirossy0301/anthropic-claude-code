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
| `boatrace/kimarite.py` | 1マークの展開予想（決まり手）。1着確率 × 勝ったときの決まり手の確率 |
| `boatrace/odds.py` | 公式サイトの単勝オッズ（締切時）の取得・解析・キャッシュ |
| `boatrace/ev.py` | 単勝の期待値（確率×オッズ）で買うバックテスト、モデルと市場の確率の比較 |
| `boatrace/synthetic.py` | 動作確認用の **合成データ**（実データではない） |
| `app.py` | Streamlit 画面の入口（上部メニューで「予想」と「予測の仕組み」を切り替え） |
| `views/predict.py` / `views/method.py` | 予想の画面 / 予測の仕組みの説明 |

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

### Windows（コマンドプロンプト）

仮想環境を有効化せず、仮想環境の Python を直接指定するのが確実です。

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m boatrace.cli demo-data --days 180
.venv\Scripts\python -m boatrace.cli ingest
.venv\Scripts\python -m boatrace.cli train
.venv\Scripts\python -m streamlit run app.py
```

- Python は 3.11〜3.13 を使ってください（`lhafile` の Windows 用ビルドがこの範囲のみ）。
- `ModuleNotFoundError` が出る場合は、グローバル環境の Python で起動しています。上の `.venv\Scripts\python` から始まるコマンドで起動してください。

ネットに接続できない環境では、`download` の代わりに `python -m boatrace.cli demo-data --days 180` を実行してください。合成データで一通り試せます。

## デプロイ（Streamlit Community Cloud）

GitHub Actions（`.github/workflows/daily-update`）が毎朝 06:17 / 09:17（日本時間）に次を自動実行し、`serve/` をコミットします。

1. 前日までの DB を Actions のキャッシュから復元（無い場合は3年分を取り直す。約3時間）
2. 前日の成績・当日と翌日の番組表・気温を取り込む（未確定の成績は保存せず、次回取り直す）
3. 直近8日分＋当日の特徴量を `serve/features.parquet` に書き出す
4. 月曜朝・初回・手動指定・モデルの作り方を変えたとき（`MODEL_VERSION` が上がったとき）は、モデルを再学習して `serve/model_*.pkl` に保存する

GitHub の定期実行は遅れたり飛ばされたりすることがあります。データが古いままのときは、**Actions → daily-update → Run workflow** で手動実行してください。

アプリは `serve/` があればそれだけを読みます（3年分の再計算をしないので、無料枠のメモリで動きます）。

### 初回の手順

1. GitHub のリポジトリで **Actions → daily-update → Run workflow** を押して初回を実行する（3年分の取得があるので約3時間）
2. 最後の「Commit serve/」で 403 エラーが出た場合は、**Settings → Actions → General → Workflow permissions** を「Read and write permissions」にして、もう一度実行する
3. `serve/` がコミットされたら、https://share.streamlit.io に GitHub アカウントでログインし、**Create app** でこのリポジトリ・ブランチ・`app.py` を指定する。**Advanced settings** で Python は 3.12 を選ぶ
4. 閲覧者を限定する場合は、アプリの **Settings → Sharing** で閲覧できる人のメールアドレスを登録する

### 3年分のバックテスト

Actions のキャッシュにある3年分のデータで実行します。結果は実行ページの **Summary** に出ます（全体の指標、要因を追加した効果の95%信頼区間、較正、四半期ごとの回収率、特徴量の寄与）。

- Actions の画面で **backtest → Run workflow**、または
- `backtest-request.txt` の `cutoff` / `top_n` / `suite`（`win` 1着・3連単 / `kimarite` 決まり手 / `all`）を書き換えて push する

### 単勝の期待値バックテスト

検証期間（2025年10月〜）から無作為に選んだ 2,000 レースの締切時単勝オッズを公式サイトから取得し、「1着確率 × オッズ」が閾値以上の艇だけを買った場合の回収率を出します。

- `odds-request.txt` の `run` の数字を変えて push する（または **Actions → odds-backtest → Run workflow**）
- 公式サイトへの負荷を避けるため 1本ずつ3秒以上あけて取得し、1回の実行は約4.5時間で止まります。2,000 レースには **2回の実行**が要ります（取得済みはキャッシュから再利用）
- 結果は実行ページの **Summary** と成果物 `odds-backtest` に出ます

画面では「単勝オッズと期待値」を開いてボタンを押すと、そのレースの単勝オッズを1回だけ取得して期待値を表示します。

#### 結果（2026-10-02、2,000 レースを取得・6艇そろった 1,885 レースで評価）

学習は 2023-10〜2025-09、検証レースは 2025-10〜2026-10 から無作為抽出。締切時オッズで、各100円。
公平な比較は **朝のモデル**（締切前に分かる情報だけ）。直前のモデルは実際の進入コース（スタート時に決まる）を使うので良く見えすぎる。

| 比較 | 値 [95%信頼区間] |
|---|---|
| 確認: 勝った艇の オッズ×100 と単勝払戻の一致率 | 98.6%（1回目 1,248 レース時点） |
| 市場（オッズから逆算した確率）の対数損失 | 1.2517 |
| 朝のモデル − 市場（対数損失） | **−0.0311 [−0.0527, −0.0095]**（モデルの方が良い） |
| p ∝ モデル^a × 市場^b（前半で決めて後半で評価） | a = 0.60, b = 0.46、市場との差 −0.0518 [−0.0694, −0.0343] |

| 買い方（朝のモデル） | 点数 | 的中率 | 平均オッズ | 回収率 [95%信頼区間] |
|---|---|---|---|---|
| 全艇を買う | 11,310 | 16.7% | 14.5 | 69.1% [64.0, 74.8] |
| 1番人気を買う | 1,917 | 55.5% | 1.6 | 80.3% [77.0, 83.9] |
| 確率1位を買う | 1,885 | 56.8% | 1.8 | 89.5% [85.0, 94.0] |
| EV≥1 | 2,662 | 16.2% | 21.6 | 110.4% [92.5, 131.3] |
| EV≥1.2 | 1,604 | 13.2% | 27.4 | 130.2% [101.9, 163.9] |
| EV≥1.5 | 864 | 11.0% | 35.5 | 155.5% [106.7, 213.3] |

読み方:

- **モデルには、締切時オッズが織り込んでいない情報がある**（対数損失の差の区間が 0 をまたがない）。
- **EV 買いで儲かるとはまだ言えない。** EV≥1 は区間が 100% をまたぐ。EV≥1.2 以上は区間が 100% を超えるが、閾値を6つ試した中から良いものを見ているので偶然の可能性が残り、高配当の数回の当たりで大きく動く。
- 実際に買うのは締切前で、高配当の艇ほど締切直前にオッズが下がりやすい。締切時オッズでの回収率は実際より良く出やすい。
- そこで **EV≥1 と EV≥1.2 を仮説として**、下の「EV 買いの記録」（締切5分前のオッズで判断、実際の払戻で精算）で2か月確かめる。それまで実際のお金は賭けない。

### EV 買いの記録（Supabase、PC 不要・お金は賭けない）

締切5分前の単勝オッズで期待値（朝のモデルの1着確率 × オッズ）を記録し、翌朝に実際の単勝払戻で精算します。アプリの **EV記録** ページで集計を見られます。

```
毎朝 GitHub Actions ─ paper-settle（前日分を精算）→ paper-plan（今日の4場の1着確率と締切時刻）→ Supabase
Supabase pg_cron（毎分）→ Edge Function paper-odds（締切4〜6分前のレースだけオッズを1回取得）→ paper_records
Streamlit アプリ（EV記録）← paper_records を読み取り専用で表示
```

初回の設定（画面の名前は Supabase の更新で変わることがあります）:

1. https://supabase.com で **New project**（Free プラン、リージョンは Tokyo）を作る
2. **SQL Editor** に `supabase/migrations/0001_paper.sql` を貼り付けて実行する（テーブル3つ・読み取り専用のポリシー・pg_cron / pg_net）
3. **Edge Functions → Deploy a new function → Via Editor** で名前を `paper-odds` にし、`supabase/functions/paper-odds/index.ts` を貼り付けて Deploy する。関数の設定で **Verify JWT（JWT の検証）をオフ**にする（代わりに下の CRON_SECRET で確かめる）
4. **Edge Functions → Secrets** に `CRON_SECRET`（長いランダムな文字列）を追加する
5. `supabase/cron.sql` の `<PROJECT_REF>` と `<CRON_SECRET>` を置き換えて、SQL Editor で実行する（日本時間 8:00〜23:59 の毎分）
6. GitHub の **Settings → Secrets and variables → Actions** に `SUPABASE_URL`（https://xxxx.supabase.co）と `SUPABASE_SECRET_KEY`（Project Settings → API Keys の secret key）を登録する
7. Streamlit のアプリの **Settings → Secrets** に次を登録する（読み取り用の publishable key。secret key は入れない）
   ```toml
   SUPABASE_URL = "https://xxxx.supabase.co"
   SUPABASE_ANON_KEY = "sb_publishable_..."
   ```
8. 確認: **Actions → daily-update → Run workflow** を実行し、「計画に書き込みました」と出ること。最初のレースの締切後に Table Editor の `paper_records` に行が増えること。定期実行の記録は `select * from cron.job_run_details order by start_time desc limit 20;`

- 公式サイトへのアクセスは1日40〜50回（4場 × 12レース、1レース1回）です。
- 毎朝6時17分の時点で番組表が出ていない場は、9時17分の更新で計画に入ります。それより前に締切があるモーニングレースは記録されないことがあります。
- 止めるときは SQL Editor で `select cron.unschedule('paper-odds');` を実行します。

### EV 買いの記録（自宅 PC で動かす場合）

締切5分前の単勝オッズで期待値（朝のモデルの1着確率 × オッズ）を計算し、6艇分を `data/paper/日付.csv` に記録します。精算は実際の単勝払戻（締切時オッズ）で行うので、実際に買った場合と同じ条件です。

```bat
git pull
.venv\Scripts\python -m boatrace.cli paper-trade          :: レースがある時間帯は起動したままにする（Ctrl+C で止める）
.venv\Scripts\python -m boatrace.cli paper-report         :: 翌朝以降に git pull してから。結果を書き込み、回収率を表示
```

- 予想は `serve/`（毎朝の更新）を使うので、**起動前に `git pull`** してください。手元の DB やダウンロードは不要です。
- 記録する場は毎日4場（日付で決まる無作為抽出、`--venues` で変更）。1レース1回だけオッズを取得します（1日40〜50回）。
- 途中で止めても、再起動すれば記録済みのレースを飛ばして続きから記録します。締切を過ぎたレースは飛ばします。
- 結果は `serve/` に8日分しか残らないので、**1週間に1回以上** `paper-report` を実行してください（結果は記録の CSV に書き込まれます）。

- 毎朝 Actions がコミットするので、手元で作業する前に `git pull` してください。
- 非公開リポジトリの Actions 無料枠は月 2,000 分です。毎日の更新は1回数分です。

## 注意

- 公式サイトの情報は私的使用に限られ、大量のアクセスは禁止されています（サイトポリシー）。オッズは上の範囲だけ取得し、取得したオッズを他の人に配ったり転載したりしないでください。
- 取得先として `www.boatrace.jp`（オッズ）、`www1.mbrace.or.jp`（公式データ）と `archive-api.open-meteo.com` / `api.open-meteo.com`（気温）への接続が必要です。
- パーサーと気温の取得は実データで検証済みです（2026-09-28 の全864艇で番組表と成績の登録番号が一致、3連単144レースが払戻一覧と一致）。**24場の水面の分類は未検証**です（`boatrace/venues.py`）。
- 3年分は約2,200ファイルです。公式サーバーは1ファイルの応答に約11秒かかるため、1本ずつだと約7.5時間、`--workers 3` で約2.5時間かかります。途中で止めても、再実行すれば取得済みの日を飛ばして続きから取得します。
- 控除率は約25%です。確率予測が当たっても、回収率100%を超えるとは限りません。
