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


def model_path(mode: str) -> Path:
    """mode: "morning" (朝の予想) / "prerace" (直前の予想)"""
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
