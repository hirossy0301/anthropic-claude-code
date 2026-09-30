"""番組表(B)・競走成績(K)テキストのパーサー。

公式配布ファイルは Shift_JIS の固定長テキストで、全角数字・全角空白が混在する。
行ごとに NFKC 正規化して半角に揃えたうえで、桁位置ではなく正規表現で読み取る
(桁位置は全角/半角の混在でずれやすいため)。
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date

VENUE_BEGIN = re.compile(r"^(\d{2})[BK]BGN")
VENUE_END = re.compile(r"^(\d{2})[BK]END")
RACE_HEADER = re.compile(r"^\s*(\d{1,2})R\s+(.*?)\s+H(\d{3,4})m")

PROGRAM_ENTRY = re.compile(
    r"^([1-6])\s(\d{4})(\D+?)(\d{2})(\D+?)(\d{2})([AB][12])\s+"
    r"(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"
    r"(\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)"
)
RESULT_ENTRY = re.compile(
    r"^\s*(\d{2}|[A-Z]{1,2}\d?)\s+([1-6])\s+(\d{4})\s+(.+?)\s+(\d+)\s+(\d+)\s+"
    r"(\S+)\s+(\S+)\s+(\S+)(?:\s+(.*))?$"
)
TRIFECTA = re.compile(r"^\s*3連単\s+(\d-\d-\d)\s+(\d+)")
WIN = re.compile(r"^\s*単勝\s+(\d)\s+(\d+)")
WIND = re.compile(r"風\s*(\S+?)\s+(\d+)m")
DEADLINE = re.compile(r"締切予定\s*(\d{1,2}:\d{2})")
WAVE = re.compile(r"波\s*(\d+)cm")


def _norm(line: str) -> str:
    return unicodedata.normalize("NFKC", line.rstrip("\r\n"))


def _float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _int(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_program(text: str, race_date: date) -> list[dict]:
    """番組表テキスト -> 出走表の行 (1艇1行) のリスト。"""
    rows, venue, race = [], None, None
    for raw in text.splitlines():
        line = _norm(raw)
        if m := VENUE_BEGIN.match(line):
            venue, race = int(m.group(1)), None
            continue
        if VENUE_END.match(line):
            venue = race = None
            continue
        if venue is None:
            continue
        if m := RACE_HEADER.match(line):
            race = {"race_no": int(m.group(1)), "race_name": m.group(2).strip(),
                    "distance": int(m.group(3)), "deadline": None}
            if d := DEADLINE.search(line):
                race["deadline"] = d.group(1)  # 電話投票締切予定 (HH:MM)
            continue
        if race and (m := PROGRAM_ENTRY.match(line)):
            g = m.groups()
            rows.append({
                "race_date": race_date.isoformat(), "venue": venue, **race,
                "lane": int(g[0]), "racer_id": int(g[1]),
                "racer_name": re.sub(r"\s+", "", g[2]), "age": int(g[3]),
                "branch": g[4].strip(), "weight": int(g[5]), "racer_class": g[6],
                "nat_win_rate": float(g[7]), "nat_2_rate": float(g[8]),
                "loc_win_rate": float(g[9]), "loc_2_rate": float(g[10]),
                "motor_no": int(g[11]), "motor_2_rate": float(g[12]),
                "boat_no": int(g[13]), "boat_2_rate": float(g[14]),
            })
    return rows


def parse_result(text: str, race_date: date) -> tuple[list[dict], list[dict]]:
    """競走成績テキスト -> (出走結果の行, レース単位の払戻の行)。"""
    results, races = [], []
    venue, race = None, None
    for raw in text.splitlines():
        line = _norm(raw)
        if m := VENUE_BEGIN.match(line):
            venue, race = int(m.group(1)), None
            continue
        if VENUE_END.match(line):
            venue = race = None
            continue
        if venue is None:
            continue
        if m := RACE_HEADER.match(line):
            race = {"race_date": race_date.isoformat(), "venue": venue,
                    "race_no": int(m.group(1)), "race_name": m.group(2).strip(),
                    "distance": int(m.group(3)), "trifecta": None,
                    "trifecta_payout": None, "win_payout": None,
                    "wind_dir": None, "wind_speed": None, "wave": None}
            if w := WIND.search(line):
                race["wind_dir"], race["wind_speed"] = w.group(1), int(w.group(2))
            if w := WAVE.search(line):
                race["wave"] = int(w.group(1))
            races.append(race)
            continue
        if race is None:
            continue
        if m := TRIFECTA.match(line):
            race["trifecta"], race["trifecta_payout"] = m.group(1), int(m.group(2))
            continue
        if m := WIN.match(line):
            race["win_payout"] = int(m.group(2))
            continue
        if m := RESULT_ENTRY.match(line):
            g = m.groups()
            st = g[8]
            flying = st.startswith("F")
            results.append({
                "race_date": race["race_date"], "venue": venue, "race_no": race["race_no"],
                "finish": _int(g[0]),  # F/L/K/S 等は None (失格・欠場)
                "finish_code": g[0], "lane": int(g[1]), "racer_id": int(g[2]),
                "exhibition_time": _float(g[6]), "course": _int(g[7]),
                "start_timing": _float(st.lstrip("FL")) if st.lstrip("FL") else None,
                "flying": int(flying),
            })
    return results, races
