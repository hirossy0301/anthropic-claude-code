"""公式サイトの直前情報 (展示タイム・風・波・気温・スタート展示の進入) の取得。

アプリのボタンを押したときだけ、そのレースの1ページを取得する (公式サイトのポリシーは私的使用に限るため)。

風向は矢印の画像 (is-wind1〜16) で、場のコースの向き (is-direction1〜16) を基準にした向きで描かれている。
2026-09-28 の9場で、直前情報 (11R時点) と競走成績の 11R の風向を照らし合わせ、
  方位 = (風の番号 − 向きの番号 + 8) mod 16   (北 = 0 から時計回りに 22.5° ずつ)
で全場一致することを確かめた。is-wind17 は無風。
"""
from __future__ import annotations

import re
import time

import requests

from .venues import WIND_DEG

BEFOREINFO_URL = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race}&jcd={venue:02d}&hd={hd}"
USER_AGENT = "boatrace-stats-research/0.1 (personal use)"
COMPASS = list(WIND_DEG)  # 北, 北北東, …, 北北西
CALM = 17


def wind_direction(wind_no: int, direction_no: int) -> str | None:
    """風の矢印の番号と場の向きの番号から 16方位を返す。無風は None。"""
    if wind_no == CALM:
        return None
    return COMPASS[(wind_no - direction_no + 8) % 16]


def _label(html: str, title: str) -> str | None:
    m = re.search(rf"{title}</span>\s*<span[^>]*>\s*([^<]*?)\s*</span>", html)
    return m.group(1) if m else None


def _number(text: str | None) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def parse_beforeinfo(html: str) -> dict:
    """直前情報のページから、風向・風速・波高・気温・水温・スタート展示の進入コースを読み取る。

    courses は枠1〜6の順に並べた進入コース (例: 4号艇が3コースなら [1, 2, 4, 3, 5, 6])。
    6艇そろっていないとき (欠場など) は None。
    """
    out: dict = {"updated": None, "wind_dir": None, "wind_speed": None, "wave": None,
                 "temperature": None, "water_temperature": None, "courses": None, "exhibition_st": None,
                 "exhibition_times": None}
    if m := re.search(r"水面気象情報\s*([^<]*?)\s*</p>", html):
        out["updated"] = m.group(1).replace("　", " ").strip()
    wind = re.search(r'weather1_bodyUnitImage is-wind(\d+)"', html)
    direction = re.search(r'weather1_bodyUnitImage is-direction(\d+)"', html)
    out["wind_speed"] = _number(_label(html, "風速"))
    if wind and direction:
        out["wind_dir"] = wind_direction(int(wind.group(1)), int(direction.group(1)))
        if out["wind_dir"] is None:
            out["wind_speed"] = 0.0
    out["wave"] = _number(_label(html, "波高"))
    out["temperature"] = _number(_label(html, "気温"))
    out["water_temperature"] = _number(_label(html, "水温"))

    # 出走表の展示タイム: 枠番のセルの後、選手名・体重の次の列 (欠場などで空欄の艇は入れない)
    times = {}
    for lane, value in re.findall(
            r'is-boatColor([1-6]) is-fs14" rowspan="4">\s*\d\s*</td>.*?kg</td>\s*<td rowspan="4">\s*([^<]*?)\s*</td>',
            html, re.S):
        if re.fullmatch(r"\d+\.\d+", value):
            times[int(lane)] = float(value)
    out["exhibition_times"] = times or None

    # スタート展示: 上の行から 1コース, 2コース, … の順に、その艇の枠番が並ぶ
    start = html.find("スタート展示")
    if start >= 0:
        rows = re.findall(r'table1_boatImage1Number is-type\d">\s*(\d)\s*<(.*?)(?=table1_boatImage1Number|</tbody>)',
                          html[start:], re.S)
        lanes = [int(lane) for lane, _ in rows]
        if sorted(lanes) == [1, 2, 3, 4, 5, 6]:
            courses = [0] * 6
            for course, lane in enumerate(lanes, start=1):
                courses[lane - 1] = course
            out["courses"] = courses
        st = {}
        for lane, rest in rows:
            t = re.search(r'table1_boatImage1Time[^"]*">\s*([^<]*?)\s*<', rest)
            st[int(lane)] = t.group(1) if t else None
        out["exhibition_st"] = st or None
    return out


def fetch_beforeinfo(race_date: str, venue: int, race_no: int, session: requests.Session | None = None,
                     retries: int = 3) -> dict:
    session = session or requests.Session()
    url = BEFOREINFO_URL.format(race=race_no, venue=venue, hd=race_date.replace("-", ""))
    for attempt in range(retries):
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()
            return parse_beforeinfo(resp.text)
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")
