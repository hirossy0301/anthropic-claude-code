"""デプロイ用の軽量データの書き出しと読み込み。

Streamlit Community Cloud では3年分の DB から特徴量を計算し直すとメモリが足りないため、
GitHub Actions で計算した「直近の特徴量」だけを Parquet で serve/ に置き、アプリはそれを読む。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .features import FEATURE_SETS, KIMARITE_HISTORY, RACE_KEYS

JST = ZoneInfo("Asia/Tokyo")
DISPLAY_COLUMNS = ["racer_id", "racer_name", "racer_class", "deadline", "finish", "wind_dir", "kimarite"]


def today_jst() -> date:
    return datetime.now(JST).date()


def export(feat: pd.DataFrame, races: pd.DataFrame, out_dir: Path, days: int,
           today: date | None = None) -> dict:
    """today の days 日前以降 (未来の出走表を含む) の特徴量とレース結果を書き出す。"""
    today = today or today_jst()
    start = (today - timedelta(days=days)).isoformat()
    features = sorted({c for cols in FEATURE_SETS.values() for c in cols} | {c for c, *_ in KIMARITE_HISTORY})
    cols = list(dict.fromkeys(RACE_KEYS + ["lane"] + DISPLAY_COLUMNS + features))
    f = feat.loc[feat["race_date"] >= start, cols]
    r = races[races["race_date"] >= start]
    out_dir.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out_dir / "features.parquet", index=False)
    r.to_parquet(out_dir / "races.parquet", index=False)
    meta = {
        "generated_at": datetime.now(JST).isoformat(timespec="minutes"),
        "first_date": f["race_date"].min() if len(f) else None,
        "last_date": f["race_date"].max() if len(f) else None,
        "races": int(f.groupby(RACE_KEYS).ngroups),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def available(serve_dir: Path) -> bool:
    return (serve_dir / "features.parquet").exists()


def load(serve_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    meta_path = serve_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return (pd.read_parquet(serve_dir / "features.parquet"),
            pd.read_parquet(serve_dir / "races.parquet"), meta)
