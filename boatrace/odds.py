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


def is_final(html: str) -> bool:
    return FINAL_MARK in html


def cache_path(race_date: str, venue: int, race_no: int, odds_dir: Path | None = None) -> Path:
    return (odds_dir or config.ODDS_DIR) / f"{race_date.replace('-', '')}_{venue:02d}_{race_no:02d}.json"


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
                 interval: float = ODDS_INTERVAL) -> dict:
    """keys のうち未取得のものを、時間の上限まで1本ずつ取得する。"""
    deadline = time.monotonic() + budget_minutes * 60
    session = _session()
    todo = [k for k in keys if not cache_path(*k, odds_dir).exists()]
    stats = {"cached": len(keys) - len(todo), "fetched": 0, "not_final": 0, "errors": 0}
    for i, key in enumerate(todo):
        if time.monotonic() > deadline:
            log(f"時間の上限に達したので終了 (残り {len(todo) - i} レースは次回)")
            break
        try:
            rec = fetch_win_odds(*key, session=session, odds_dir=odds_dir)
            stats["fetched" if rec["final"] else "not_final"] += 1
        except Exception as e:  # 1レースの失敗で止めない (キャッシュされないので次回取り直す)
            stats["errors"] += 1
            log(f"{key}: error {e}")
        if (i + 1) % 50 == 0:
            log(f"{i + 1}/{len(todo)} {stats}")
        time.sleep(interval)
    stats["remaining"] = sum(not cache_path(*k, odds_dir).exists() for k in keys)
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
