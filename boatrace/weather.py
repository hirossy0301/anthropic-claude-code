"""Open-Meteo から場ごとの1時間ごとの気温を取得する。

過去分は archive API、直近 (数日前〜予報) は forecast API を使う。
API キーは不要。無料枠は非商用利用が条件 (https://open-meteo.com/en/terms)。
"""
from __future__ import annotations

import sqlite3
import time
from datetime import date, timedelta

import requests

from .venues import VENUE_INFO

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_DELAY_DAYS = 6  # archive API は数日遅れで確定するため、それより新しい日は forecast を使う


def parse_hourly(venue: int, payload: dict) -> list[dict]:
    """Open-Meteo の hourly 応答 -> weather テーブルの行。"""
    hourly = payload.get("hourly", {})
    rows = []
    for t, temp in zip(hourly.get("time", []), hourly.get("temperature_2m", [])):
        if temp is None:
            continue
        day, hh = t.split("T")
        rows.append({"venue": venue, "race_date": day, "hour": int(hh[:2]), "temperature": temp})
    return rows


def _fetch(url: str, venue: int, start: date, end: date, session: requests.Session) -> list[dict]:
    info = VENUE_INFO[venue]
    resp = session.get(url, timeout=30, params={
        "latitude": info["lat"], "longitude": info["lon"],
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "temperature_2m", "timezone": "Asia/Tokyo",
    })
    resp.raise_for_status()
    return parse_hourly(venue, resp.json())


def fetch_temperatures(conn: sqlite3.Connection, start: date, end: date,
                       venues=None, today: date | None = None, log=print) -> None:
    """start〜end の気温を場ごとに取得して weather テーブルに保存する。"""
    from .db import upsert

    today = today or date.today()
    split = today - timedelta(days=ARCHIVE_DELAY_DAYS)
    session = requests.Session()
    for venue in venues or VENUE_INFO:
        spans = []
        if start < split:
            spans.append((ARCHIVE_URL, start, min(end, split - timedelta(days=1))))
        if end >= split:
            spans.append((FORECAST_URL, max(start, split), end))
        for url, s, e in spans:
            rows = _fetch(url, venue, s, e, session)
            upsert(conn, "weather", rows)
            log(f"気温 {VENUE_INFO[venue]['name']} {s}〜{e}: {len(rows)} 時間分")
            time.sleep(0.5)
    conn.commit()
