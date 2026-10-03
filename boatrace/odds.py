"""単勝オッズの取得 (公式サイトのオッズページ) と、取得済みオッズのキャッシュ。

公式サイトのサイトポリシーは私的使用に限り、大量のアクセスを禁じている。そのため
- 1本ずつ、リクエストの間を ODDS_INTERVAL 秒あける
- 取得した締切時オッズはキャッシュし、同じレースを2度取りに行かない
- バックテスト用はレースを無作為に抽出した一部 (既定 2,000 レース) だけ取得する
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/oddstf?rno={race}&jcd={venue:02d}&hd={hd}"
ODDS_INTERVAL = 3.0  # 秒。番組表・成績のダウンロード (1.5秒) より長めにあける
USER_AGENT = "boatrace-stats-research/0.1 (personal use)"

# 締切後のページにはこの見出しが出る (締切前は「オッズ更新時間」)
FINAL_MARK = "締切時オッズ"
_LANE_ODDS = re.compile(
    r'is-boatColor([1-6])">\s*\d\s*</td>.*?<td class="oddsPoint[^"]*">\s*([^<]*?)\s*</td>', re.S)


def parse_win_odds(html: str) -> dict[int, float | None]:
    """単勝オッズの表から {枠: オッズ} を返す。欠場の艇 (0.0 と表示される) や数字でない枠は None。表が無ければ空。"""
    start, end = html.find("単勝オッズ"), html.find("複勝オッズ")
    if start < 0:
        return {}
    section = html[start:end if end > start else None]
    out: dict[int, float | None] = {}
    for lane, value in _LANE_ODDS.findall(section):
        try:
            o = float(value)
        except ValueError:
            o = None
        out[int(lane)] = o if o and o > 0 else None
    return out


_CELL = re.compile(r'<td class="([^"]*)"(\s+rowspan="\d+")?>\s*([^<]*?)\s*</td>')


def _odds_value(text: str) -> float | None:
    try:
        o = float(text)
    except ValueError:
        return None
    return o if o > 0 else None


def parse_exacta_odds(html: str) -> dict[tuple[int, int], float | None]:
    """2連単オッズの表から {(1着, 2着): オッズ} を返す。

    表は列ごとに1着の艇 (1〜6) で、各行に「2着の艇・オッズ」の組が6列分並ぶ。
    """
    start, end = html.find("2連単オッズ"), html.find("2連複オッズ")
    if start < 0:
        return {}
    body = html[start:end if end > start else None]
    body = body[body.find("<tbody"):]
    out, col, second = {}, 0, None
    for cls, _, text in _CELL.findall(body):
        if "oddsPoint" in cls:
            if second is not None:
                out[(col + 1, second)] = _odds_value(text)
            col, second = (col + 1) % 6, None
        elif text.isdigit():
            second = int(text)
    return out


def parse_trifecta_odds(html: str) -> dict[tuple[int, int, int], float | None]:
    """3連単オッズの表から {(1着, 2着, 3着): オッズ} を返す (120通り)。

    表は列ごとに1着の艇。2着の艇は4行にまたがるセル (rowspan="4") で、各行に「3着の艇・オッズ」が並ぶ。
    """
    start = html.find("3連単オッズ")
    if start < 0:
        return {}
    body = html[start:]
    body = body[body.find("<tbody"):body.find("</table>")]
    out, col = {}, 0
    second: dict[int, int] = {}
    third = None
    for cls, rowspan, text in _CELL.findall(body):
        if "oddsPoint" in cls:
            if third is not None and col in second:
                out[(col + 1, second[col], third)] = _odds_value(text)
            col, third = (col + 1) % 6, None
        elif rowspan and text.isdigit():
            second[col] = int(text)
        elif text.isdigit():
            third = int(text)
    return out


PAGES = {"exacta": ("odds2tf", parse_exacta_odds), "trifecta": ("odds3t", parse_trifecta_odds)}
PAGE_URL = "https://www.boatrace.jp/owpc/pc/race/{page}?rno={race}&jcd={venue:02d}&hd={hd}"


def fetch_combo_odds(kind: str, race_date: str, venue: int, race_no: int,
                     session: requests.Session | None = None, retries: int = 3) -> dict:
    """2連単 ("exacta") / 3連単 ("trifecta") のオッズを1ページ取得する: {"final": 締切時か, "odds": {組: オッズ}}。"""
    page, parser = PAGES[kind]
    session = session or _session()
    url = PAGE_URL.format(page=page, race=race_no, venue=venue, hd=race_date.replace("-", ""))
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    return {"final": is_final(resp.text), "odds": parser(resp.text)}


def is_final(html: str) -> bool:
    return FINAL_MARK in html


def cache_path(race_date: str, venue: int, race_no: int, odds_dir: Path | None = None) -> Path:
    return (odds_dir or config.ODDS_DIR) / f"{race_date.replace('-', '')}_{venue:02d}_{race_no:02d}.json"


def trifecta_cache_path(race_date: str, venue: int, race_no: int, odds_dir: Path | None = None) -> Path:
    """3連単の締切時オッズのキャッシュ (単勝と混ざらないようサブフォルダに置く)。"""
    return (odds_dir or config.ODDS_DIR) / "trifecta" / f"{race_date.replace('-', '')}_{venue:02d}_{race_no:02d}.json"


def fetch_trifecta_cached(race_date: str, venue: int, race_no: int, session: requests.Session | None = None,
                          odds_dir: Path | None = None) -> dict:
    """3連単の締切時オッズを取得し、締切時のものだけキャッシュする。"""
    path = trifecta_cache_path(race_date, venue, race_no, odds_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rec = fetch_combo_odds("trifecta", race_date, venue, race_no, session=session)
    out = {"race_date": race_date, "venue": venue, "race_no": race_no, "final": rec["final"],
           "odds": {"-".join(map(str, k)): v for k, v in rec["odds"].items()}}
    if out["final"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    return out


def load_cached_trifecta(odds_dir: Path | None = None) -> pd.DataFrame:
    """キャッシュ済みの3連単の締切時オッズを1組1行で返す (race_date, venue, race_no, combo, odds)。"""
    rows = []
    for p in sorted(((odds_dir or config.ODDS_DIR) / "trifecta").glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rows += [{"race_date": rec["race_date"], "venue": int(rec["venue"]), "race_no": int(rec["race_no"]),
                  "combo": c, "odds": o} for c, o in rec["odds"].items()]
    return pd.DataFrame(rows, columns=["race_date", "venue", "race_no", "combo", "odds"])


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def fetch_win_odds(race_date: str, venue: int, race_no: int, session: requests.Session | None = None,
                   odds_dir: Path | None = None, retries: int = 3) -> dict:
    """1レースの単勝オッズを返す: {"final": 締切時オッズか, "win": {枠: オッズ}}。

    締切時オッズはキャッシュに保存し、次からはキャッシュを返す (締切前のオッズは変わるので保存しない)。
    """
    path = cache_path(race_date, venue, race_no, odds_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    session = session or _session()
    url = ODDS_URL.format(race=race_no, venue=venue, hd=race_date.replace("-", ""))
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            break
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    html = resp.text
    rec = {"race_date": race_date, "venue": venue, "race_no": race_no,
           "final": is_final(html), "win": {str(k): v for k, v in parse_win_odds(html).items()}}
    if rec["final"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    rec["win"] = {int(k): v for k, v in rec["win"].items()}
    return rec


def sample_races(races: pd.DataFrame, start: str, n: int, seed: int = 0) -> list[tuple[str, int, int]]:
    """start 以降の確定済みレースから n レースを無作為に選ぶ。

    seed が同じなら同じ順番になるので、途中で止めても次の実行で同じ抽出の続きを取得できる。
    """
    done = races[(races["race_date"] >= start) & races["win_payout"].notna()]
    keys = sorted(zip(done["race_date"], done["venue"].astype(int), done["race_no"].astype(int)))
    random.Random(seed).shuffle(keys)
    return keys[:n]


def fetch_sample(keys, budget_minutes: float, odds_dir: Path | None = None, log=print,
                 interval: float = ODDS_INTERVAL, kind: str = "win") -> dict:
    """keys のうち未取得のものを、時間の上限まで1本ずつ取得する。kind は "win" (単勝) / "trifecta" (3連単)。"""
    path_of, fetch = ((cache_path, fetch_win_odds) if kind == "win"
                      else (trifecta_cache_path, fetch_trifecta_cached))
    deadline = time.monotonic() + budget_minutes * 60
    session = _session()
    todo = [k for k in keys if not path_of(*k, odds_dir).exists()]
    stats = {"cached": len(keys) - len(todo), "fetched": 0, "not_final": 0, "errors": 0}
    for i, key in enumerate(todo):
        if time.monotonic() > deadline:
            log(f"時間の上限に達したので終了 (残り {len(todo) - i} レースは次回)")
            break
        try:
            rec = fetch(*key, session=session, odds_dir=odds_dir)
            stats["fetched" if rec["final"] else "not_final"] += 1
        except Exception as e:  # 1レースの失敗で止めない (キャッシュされないので次回取り直す)
            stats["errors"] += 1
            log(f"{key}: error {e}")
        if (i + 1) % 50 == 0:
            log(f"{i + 1}/{len(todo)} {stats}")
        time.sleep(interval)
    stats["remaining"] = sum(not path_of(*k, odds_dir).exists() for k in keys)
    return stats


def load_cached(odds_dir: Path | None = None) -> pd.DataFrame:
    """キャッシュ済みの締切時オッズを1艇1行で返す (race_date, venue, race_no, lane, odds)。"""
    rows = []
    for p in sorted((odds_dir or config.ODDS_DIR).glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        for lane, o in rec["win"].items():
            rows.append({"race_date": rec["race_date"], "venue": int(rec["venue"]),
                         "race_no": int(rec["race_no"]), "lane": int(lane), "odds": o})
    return pd.DataFrame(rows, columns=["race_date", "venue", "race_no", "lane", "odds"])
