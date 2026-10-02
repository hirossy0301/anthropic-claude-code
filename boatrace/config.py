"""パスと定数。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 環境変数 BOATRACE_DATA_DIR で保存先を切り替えられる (例: 実データと合成データを分ける)
DATA_DIR = Path(os.environ.get("BOATRACE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # 解凍済みテキスト (b/k YYMMDD.txt)
LZH_DIR = DATA_DIR / "lzh"  # ダウンロードした圧縮ファイル
DB_PATH = DATA_DIR / "boatrace.db"
MODEL_DIR = DATA_DIR
# デプロイ用の軽量データ (直近の特徴量とモデル)。GitHub Actions が毎朝更新してコミットする
SERVE_DIR = Path(os.environ.get("BOATRACE_SERVE_DIR", ROOT / "serve"))
# 取得済みの単勝オッズ (締切時) のキャッシュ。1レース1ファイルの JSON
ODDS_DIR = Path(os.environ.get("BOATRACE_ODDS_DIR", DATA_DIR / "odds"))


# モデルの作り方を変えたら上げる。serve/model_version.txt と違えば daily-update が再学習する
MODEL_VERSION = 4  # 2: 枠ごとの確率の較正 / 3: 決まり手のモデル / 4: 展示タイムを使う直前モデル (prerace_exh)
# 枠ごとの確率の較正を本番の学習で使うか。3年分のバックテスト (直近1年 54,339 レース) で
# 対数損失 -0.0039 [95%CI -0.0047, -0.0031]、1着的中率 +0.26% の改善を確認して有効にした
CALIBRATE = True


def model_path(mode: str) -> Path:
    """mode: "morning" (朝の予想) / "prerace" (直前の予想) / "prerace_exh" (直前の予想 + 展示タイム)"""
    return MODEL_DIR / f"model_{mode}.pkl"


# 公式ダウンロードページ (https://www.boatrace.jp/owpc/pc/extra/data/download.html)
# から配布されているファイルの実体。B=番組表, K=競走成績。
DOWNLOAD_URL = "https://www1.mbrace.or.jp/od2/{kind}/{yyyymm}/{prefix}{yymmdd}.lzh"

# サーバー負荷を避けるためのリクエスト間隔 (秒)
REQUEST_INTERVAL = 1.5

VENUES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村",
}
