"""SQLite への保存と読み出し。"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import config
from .parsers import parse_program, parse_result

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    race_date TEXT, venue INTEGER, race_no INTEGER, race_name TEXT, distance INTEGER,
    lane INTEGER, racer_id INTEGER, racer_name TEXT, age INTEGER, branch TEXT,
    weight INTEGER, racer_class TEXT, nat_win_rate REAL, nat_2_rate REAL,
    loc_win_rate REAL, loc_2_rate REAL, motor_no INTEGER, motor_2_rate REAL,
    boat_no INTEGER, boat_2_rate REAL, deadline TEXT,
    PRIMARY KEY (race_date, venue, race_no, lane)
);
CREATE TABLE IF NOT EXISTS results (
    race_date TEXT, venue INTEGER, race_no INTEGER, lane INTEGER, racer_id INTEGER,
    finish INTEGER, finish_code TEXT, exhibition_time REAL, course INTEGER,
    start_timing REAL, flying INTEGER,
    PRIMARY KEY (race_date, venue, race_no, lane)
);
CREATE TABLE IF NOT EXISTS races (
    race_date TEXT, venue INTEGER, race_no INTEGER, race_name TEXT, distance INTEGER,
    trifecta TEXT, trifecta_payout INTEGER, win_payout INTEGER,
    wind_dir TEXT, wind_speed INTEGER, wave INTEGER,
    PRIMARY KEY (race_date, venue, race_no)
);
CREATE TABLE IF NOT EXISTS weather (
    venue INTEGER, race_date TEXT, hour INTEGER, temperature REAL,
    PRIMARY KEY (venue, race_date, hour)
);
"""

# 既存 DB に後から追加した列 (table, column, type)
MIGRATIONS = [("entries", "deadline", "TEXT")]


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    for table, col, typ in MIGRATIONS:
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        if col not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    return conn


def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    sql = (f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
           f"VALUES ({','.join('?' * len(cols))})")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])


def ingest_dir(conn: sqlite3.Connection, raw_dir: Path = config.RAW_DIR, log=print) -> None:
    """raw_dir の b*.txt / k*.txt をすべて DB に取り込む。"""
    for path in sorted(raw_dir.glob("[bk]*.txt")):
        d = datetime.strptime(path.stem[1:], "%y%m%d").date()
        text = path.read_text(encoding="utf-8")
        if path.stem[0] == "b":
            rows = parse_program(text, d)
            upsert(conn, "entries", rows)
            log(f"{path.name}: entries {len(rows)}")
        else:
            results, races = parse_result(text, d)
            upsert(conn, "results", results)
            upsert(conn, "races", races)
            log(f"{path.name}: results {len(results)}, races {len(races)}")
    conn.commit()


def load_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    """出走表 + 結果 + 風・波 + 気温を1艇1行で結合した DataFrame。

    未確定レースは finish が NaN。気温は締切予定時刻の「時」の値 (締切不明なら12時)。
    """
    return pd.read_sql_query(
        """
        SELECT e.*, r.finish, r.finish_code, r.exhibition_time, r.course,
               r.start_timing, r.flying,
               ra.wind_dir, ra.wind_speed, ra.wave, w.temperature
        FROM entries e
        LEFT JOIN results r USING (race_date, venue, race_no, lane)
        LEFT JOIN races ra USING (race_date, venue, race_no)
        LEFT JOIN weather w ON w.venue = e.venue AND w.race_date = e.race_date
             AND w.hour = COALESCE(CAST(substr(e.deadline, 1, instr(e.deadline, ':') - 1) AS INTEGER), 12)
        ORDER BY e.race_date, e.venue, e.race_no, e.lane
        """,
        conn,
    )


def load_races(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM races", conn)
