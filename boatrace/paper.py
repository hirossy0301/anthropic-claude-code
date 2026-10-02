"""お金を賭けずに「EV 買い」を記録して検証する (ペーパートレード)。

自宅 PC で動かす:
  python -m boatrace.cli paper-trade      締切5分前ごとに単勝オッズを取得し、期待値を CSV に記録
  python -m boatrace.cli paper-report     結果と照らし合わせて回収率を出す

- 予想は serve/ (GitHub Actions が毎朝更新) の当日の特徴量と朝のモデルを使う。朝に git pull しておく。
- 公式サイトへの負荷を抑えるため、記録する場は毎日 n 場だけ (日付で決まる無作為抽出)。1レース1回だけ取得する。
- 払戻は成績の単勝払戻 (締切時オッズ) で精算する。単勝は締切時のオッズで払い戻されるので、実際に買った場合と同じ。
"""
from __future__ import annotations

import random
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import RACE_KEYS
from .serve import JST

COLUMNS = ["recorded_at", "race_date", "venue", "race_no", "deadline", "lane", "racer_name",
           "win_prob", "odds", "ev", "finish", "win_payout"]
THRESHOLDS = (1.0, 1.2, 1.5)


def paper_dir() -> Path:
    return config.DATA_DIR / "paper"


def pick_venues(venues, day: date, n: int) -> list[int]:
    """その日に開催している場から n 場を選ぶ。日付で決まるので、途中で再起動しても同じ場になる。"""
    venues = sorted(int(v) for v in venues)
    return sorted(random.Random(day.isoformat()).sample(venues, min(n, len(venues))))


def schedule(feat: pd.DataFrame, day: date, venues: list[int]) -> list[tuple[datetime, int, int]]:
    """(締切予定の日時, 場, レース) を締切順に返す。締切時刻が分からないレースは除く。"""
    g = feat[(feat["race_date"] == day.isoformat()) & feat["venue"].isin(venues)]
    out = []
    for (v, r), d in g.groupby(["venue", "race_no"])["deadline"].first().items():
        if isinstance(d, str) and ":" in d:
            hh, mm = (int(x) for x in d.split(":"))
            out.append((datetime(day.year, day.month, day.day, hh, mm, tzinfo=JST), int(v), int(r)))
    return sorted(out)


def record_rows(g: pd.DataFrame, win_prob: pd.Series, odds: dict, now: datetime) -> pd.DataFrame:
    """1レース分の記録 (6艇)。オッズが無い艇 (欠場など) は odds / ev が空。"""
    rows = g[["race_date", "venue", "race_no", "deadline", "lane", "racer_name"]].copy()
    rows.insert(0, "recorded_at", now.isoformat(timespec="seconds"))
    rows["win_prob"] = win_prob.to_numpy()
    rows["odds"] = rows["lane"].map(lambda lane: odds.get(int(lane)))
    rows["ev"] = rows["win_prob"] * rows["odds"]
    rows["finish"] = np.nan
    rows["win_payout"] = np.nan
    return rows[COLUMNS]


def _csv(out_dir: Path, day: date) -> Path:
    return out_dir / f"{day.isoformat()}.csv"


def run(day: date | None = None, n_venues: int = 4, minutes_before: float = 5, out_dir: Path | None = None,
        log=print, now_fn=None, sleep_fn=time.sleep, fetch=None) -> int:
    """当日のレースを締切 minutes_before 分前に1回ずつ記録する。記録したレース数を返す。

    途中で止めても、再起動すれば記録済みのレースを飛ばして続きから記録する。
    """
    from . import serve
    from .model import WinModel
    from .odds import fetch_win_odds

    now_fn = now_fn or (lambda: datetime.now(JST))
    fetch = fetch or (lambda d, v, r: fetch_win_odds(d, v, r))
    day = day or now_fn().date()
    out_dir = out_dir or paper_dir()
    feat, _, meta = serve.load(config.SERVE_DIR)
    if not (feat["race_date"] == day.isoformat()).any():
        raise SystemExit(f"{day} の出走表が serve/ にありません。git pull してから実行してください"
                         f"（データ更新: {meta.get('generated_at')}）")
    model = WinModel.load(config.SERVE_DIR / config.model_path("morning").name)
    venues = pick_venues(feat.loc[feat["race_date"] == day.isoformat(), "venue"].unique(), day, n_venues)
    plan = schedule(feat, day, venues)
    path = _csv(out_dir, day)
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev["venue"], prev["race_no"]))
    log(f"{day} 記録する場: {', '.join(config.VENUES.get(v, str(v)) for v in venues)}"
        f" ／ {len(plan)} レース（記録済み {len(done)}）")

    count = 0
    for deadline, venue, race_no in plan:
        if (venue, race_no) in done:
            continue
        at = deadline - timedelta(minutes=minutes_before)
        if now_fn() > deadline - timedelta(minutes=1):
            log(f"{config.VENUES.get(venue)} {race_no}R: 締切を過ぎたので飛ばします")
            continue
        wait = (at - now_fn()).total_seconds()
        if wait > 0:
            sleep_fn(wait)
        g = feat[(feat["race_date"] == day.isoformat()) & (feat["venue"] == venue) & (feat["race_no"] == race_no)]
        try:
            rec = fetch(day.isoformat(), venue, race_no)
        except Exception as e:  # 通信エラーでも次のレースへ進む
            log(f"{config.VENUES.get(venue)} {race_no}R: オッズを取得できませんでした ({e})")
            continue
        odds = {int(k): v for k, v in rec["win"].items()}
        if not odds:
            log(f"{config.VENUES.get(venue)} {race_no}R: オッズがありません")
            continue
        rows = record_rows(g, model.predict_win_prob(g), odds, now_fn())
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(path, mode="a", header=not path.exists(), index=False)
        bets = rows[rows["ev"] >= 1]
        log(f"{deadline:%H:%M} {config.VENUES.get(venue)} {race_no}R: "
            + (", ".join(f"{int(b.lane)}号艇 EV {b.ev:.2f} ({b.odds:.1f}倍)" for b in bets.itertuples())
               if len(bets) else "EV≥1 なし"))
        count += 1
    log(f"{day} の記録を終了しました（{count} レース）")
    return count


def load_records(out_dir: Path | None = None) -> pd.DataFrame:
    files = sorted((out_dir or paper_dir()).glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def fill_results(df: pd.DataFrame, feat: pd.DataFrame, races: pd.DataFrame) -> pd.Series:
    """df の未精算の行に着順と単勝払戻を書き込み、今回書き込んだ行の印を返す。"""
    todo = df["win_payout"].isna().to_numpy()
    fin = feat[RACE_KEYS + ["lane", "finish"]].rename(columns={"finish": "_finish"})
    pay = races[RACE_KEYS + ["win_payout"]].rename(columns={"win_payout": "_payout"})
    key = df[RACE_KEYS + ["lane"]].astype({"race_date": str, "venue": int, "race_no": int, "lane": int})
    m = key.merge(fin, on=RACE_KEYS + ["lane"], how="left").merge(pay, on=RACE_KEYS, how="left")
    known = todo & m["_payout"].notna().to_numpy()
    df.loc[known, "finish"] = m.loc[known, "_finish"].to_numpy()
    df.loc[known, "win_payout"] = m.loc[known, "_payout"].to_numpy()
    return pd.Series(known, index=df.index)


def settle(feat: pd.DataFrame, races: pd.DataFrame, out_dir: Path | None = None) -> int:
    """結果が出たレースの着順と単勝払戻を CSV に書き込む (serve/ は直近8日分しか持たないため、記録側に残す)。"""
    n = 0
    for path in sorted((out_dir or paper_dir()).glob("*.csv")):
        df = pd.read_csv(path)
        if not df["win_payout"].isna().any():
            continue
        known = fill_results(df, feat, races)
        df.to_csv(path, index=False)
        n += int(known.sum() // 6)
    return n


# ---- Supabase で記録する場合 (PC 不要。Edge Function が締切前のオッズを記録する) ----

def plan_rows(feat: pd.DataFrame, model, day: date, n_venues: int = 4) -> list[dict]:
    """その日に記録する n 場の全レースについて、朝のモデルの1着確率と締切日時を返す (paper_plan 用)。"""
    today = feat[feat["race_date"] == day.isoformat()]
    if today.empty:
        return []
    venues = pick_venues(today["venue"].unique(), day, n_venues)
    g = today[today["venue"].isin(venues)].copy()
    g["win_prob"] = model.predict_win_prob(g).to_numpy()
    deadlines = {(v, r): d for d, v, r in schedule(g, day, venues)}
    rows = []
    for row in g.itertuples():
        d = deadlines.get((int(row.venue), int(row.race_no)))
        if d is None:
            continue
        rows.append({"race_date": row.race_date, "venue": int(row.venue), "race_no": int(row.race_no),
                     "lane": int(row.lane), "deadline": d.isoformat(), "racer_name": row.racer_name,
                     "win_prob": float(row.win_prob)})
    return rows


def settle_rows(records: pd.DataFrame, feat: pd.DataFrame, races: pd.DataFrame) -> list[dict]:
    """Supabase の未精算の記録に、着順と単勝払戻を付けた行 (主キー + 結果の列だけ) を返す。"""
    if records.empty:
        return []
    df = records.copy()
    for c in ("finish", "win_payout"):
        if c not in df:
            df[c] = np.nan
    known = fill_results(df, feat, races)
    out = df.loc[known, RACE_KEYS + ["lane", "finish", "win_payout"]]
    return [{"race_date": r.race_date, "venue": int(r.venue), "race_no": int(r.race_no), "lane": int(r.lane),
             "finish": None if pd.isna(r.finish) else int(r.finish), "win_payout": int(r.win_payout)}
            for r in out.itertuples()]


def report(records: pd.DataFrame) -> str:
    """精算済みの記録から、閾値ごとの回収率と、締切前と締切時のオッズのずれをまとめる。"""
    from .ev import roi_ci

    done = records[records["win_payout"].notna()].copy()
    if done.empty:
        return "精算済みの記録がまだありません（結果が serve/ に入るのは翌朝の更新後です）。"
    done["won"] = done["finish"] == 1
    n_races = done.groupby(RACE_KEYS).ngroups
    lines = [f"# EV 買いの記録（{done['race_date'].min()}〜{done['race_date'].max()}、精算済み {n_races:,} レース）",
             "", "締切前のオッズで判断し、実際の単勝払戻（締切時オッズ）で精算。各100円。", "",
             "| 買い方 | 点数 | 的中 | 購入 | 払戻 | 回収率 | 95%CI |", "|---|---|---|---|---|---|---|"]
    for t in THRESHOLDS:
        bet = done["ev"] >= t
        per = done.assign(bet=bet * 100, payout=np.where(bet & done["won"], done["win_payout"], 0))
        per = per.groupby(RACE_KEYS)[["bet", "payout"]].sum().reset_index()
        roi, lo, hi = roi_ci(per)
        k, hit = int(bet.sum()), int((bet & done["won"]).sum())
        lines.append(f"| EV≥{t:g} | {k:,} | {hit:,} | {k * 100:,}円 | {int(per['payout'].sum()):,}円 | "
                     + (f"{roi:.1%} | [{lo:.1%}, {hi:.1%}] |" if k else "- | - |"))
    w = done[done["won"] & done["odds"].notna()]
    if len(w):
        drift = w["win_payout"] / 100 / w["odds"]
        bw = w[w["ev"] >= 1]
        lines += ["", "## 締切前と締切時のオッズのずれ（勝った艇）",
                  f"- 全体: 締切時 ÷ 締切前 の中央値 {drift.median():.2f}（{len(w):,} 艇）"]
        if len(bw):
            lines.append(f"- EV≥1 で買った艇: 中央値 {(bw['win_payout'] / 100 / bw['odds']).median():.2f}"
                         f"（{len(bw):,} 艇）。1 より小さいと、締切までにオッズが下がった")
    daily = done.assign(b=(done["ev"] >= 1) * 100,
                        p=np.where((done["ev"] >= 1) & done["won"], done["win_payout"], 0))
    daily = daily.groupby("race_date")[["b", "p"]].sum()
    daily["収支"] = daily["p"] - daily["b"]
    daily["累計"] = daily["収支"].cumsum()
    lines += ["", "## 日ごとの収支（EV≥1）", "", "| 日付 | 購入 | 払戻 | 収支 | 累計 |", "|---|---|---|---|---|"]
    lines += [f"| {d} | {int(r.b):,} | {int(r.p):,} | {int(r.収支):+,} | {int(r.累計):+,} |" for d, r in daily.iterrows()]
    return "\n".join(lines)
