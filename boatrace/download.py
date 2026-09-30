"""番組表(B)・競走成績(K)ファイルのダウンロードと解凍。"""
from __future__ import annotations

import queue
import threading
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from . import config

KINDS = {"B": "b", "K": "k"}


def build_url(kind: str, d: date) -> str:
    return config.DOWNLOAD_URL.format(
        kind=kind, yyyymm=d.strftime("%Y%m"), prefix=KINDS[kind], yymmdd=d.strftime("%y%m%d")
    )


def raw_path(kind: str, d: date) -> Path:
    return config.RAW_DIR / f"{KINDS[kind]}{d.strftime('%y%m%d')}.txt"


def extract_lzh(lzh_path: Path) -> str:
    """LZH を解凍し、Shift_JIS(cp932) のテキストを返す。"""
    import lhafile

    archive = lhafile.Lhafile(str(lzh_path))
    parts = [archive.read(info.filename) for info in archive.infolist()]
    return b"".join(parts).decode("cp932", errors="replace")


def fetch_day(kind: str, d: date, session: requests.Session) -> Path | None:
    """1日分を取得して解凍テキストを保存する。開催なし(404)の日は None。"""
    out = raw_path(kind, d)
    if out.exists():
        return out
    resp = session.get(build_url(kind, d), timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    config.LZH_DIR.mkdir(parents=True, exist_ok=True)
    lzh = config.LZH_DIR / f"{KINDS[kind]}{d.strftime('%y%m%d')}.lzh"
    lzh.write_bytes(resp.content)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")  # 途中で止まっても壊れたファイルを「取得済み」と誤認しないように
    tmp.write_text(extract_lzh(lzh), encoding="utf-8")
    tmp.replace(out)
    return out


def download_range(start: date, end: date, kinds=("B", "K"), log=print, workers: int = 1) -> None:
    """start〜end (両端含む) を取得する。取得済みの日はスキップ。

    workers > 1 で並列取得する (公式サーバーは1ファイルの応答に10秒前後かかるため)。
    各ワーカーは自分のリクエストの間に REQUEST_INTERVAL 秒あける。
    """
    tasks: queue.Queue = queue.Queue()
    d = start
    while d <= end:
        for kind in kinds:
            if not raw_path(kind, d).exists():
                tasks.put((kind, d))
        d += timedelta(days=1)

    def worker() -> None:
        session = requests.Session()
        session.headers["User-Agent"] = "boatrace-stats-research/0.1 (personal use)"
        while True:
            try:
                kind, day = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                path = fetch_day(kind, day, session)
                log(f"{kind} {day}: {'ok' if path else 'no data'}")
            except Exception as e:  # 通信エラー・壊れた LZH でもワーカーを止めない (再実行で再取得される)
                log(f"{kind} {day}: error {e}")
            time.sleep(config.REQUEST_INTERVAL)

    threads = [threading.Thread(target=worker) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
