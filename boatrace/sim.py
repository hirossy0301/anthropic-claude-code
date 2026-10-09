"""単勝の期待値 (EV) で毎日決まった予算を買い続けた場合の、資金の増減のシミュレーション。

ルール: 毎日の予算を、その日の「EV ≥ 閾値」の艇に均等に割り振る (100円単位、余りは買わない)。
予算が対象の艇の数 × 100円に足りない日は、EV の高い順に100円ずつ予算の分だけ買う。

使うのは ev.report と同じ検証データ (締切時オッズを取得済みのレース)。オッズは検証期間から無作為に
選んだレースなので1日あたり数レースしかなく、実際に全レースから買う場合とは日ごとの上下の出方が違う。
1日の予算が同じなので、対象の艇が少ない日ほど1点に大きく賭ける (各100円の回収率とは重みが違う)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .ev import attach_odds
from .features import FEATURE_SETS, training_rows
from .model import WinModel

BUDGETS = (500, 1000, 3000, 5000)
THRESHOLDS = (1.0, 1.2)


def allocate(day: pd.DataFrame, budget: int) -> np.ndarray:
    """その日の対象の艇 (EV の列 ev) への賭け金 (100円単位)。"""
    n = len(day)
    if n == 0:
        return np.zeros(0)
    unit = budget // n // 100 * 100
    if unit > 0:
        return np.full(n, unit, dtype=float)
    stake = np.zeros(n)
    stake[np.argsort(-day["ev"].to_numpy(), kind="stable")[:budget // 100]] = 100
    return stake


def simulate(m: pd.DataFrame, prob_col: str, budgets=BUDGETS, thresholds=THRESHOLDS) -> pd.DataFrame:
    """日ごと・予算ごと・閾値ごとの購入額と払戻、累計の収支。

    m: attach_odds の結果に確率の列 prob_col を足したもの。
    戻り値の列: rule, budget, race_date, races, bets, bet, payout, profit, cum_profit
    """
    m = m.assign(ev=m[prob_col] * m["odds"])
    payout_per_100 = m["win_payout"].fillna(m["odds"] * 100)
    m = m.assign(ret=np.where(m["win"] == 1, payout_per_100, 0.0))
    races_per_day = m.groupby("race_date")["race_no"].size() / 6
    out = []
    for t in thresholds:
        cand = m[m["ev"] >= t]
        for budget in budgets:
            for d in sorted(m["race_date"].unique()):
                day = cand[cand["race_date"] == d]
                stake = allocate(day, budget)
                bet = float(stake.sum())
                pay = float((stake / 100 * day["ret"].to_numpy()).sum()) if len(day) else 0.0
                out.append({"rule": f"EV≥{t:g}", "budget": budget, "race_date": d,
                            "races": int(races_per_day[d]), "bets": int((stake > 0).sum()),
                            "bet": bet, "payout": pay})
    df = pd.DataFrame(out)
    df["profit"] = df["payout"] - df["bet"]
    df["cum_profit"] = df.groupby(["rule", "budget"])["profit"].cumsum()
    return df


def run(feat: pd.DataFrame, races: pd.DataFrame, odds: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """cutoff より前で朝の予想モデルを学習し、cutoff 以降のオッズ取得済みレースでシミュレーションする。"""
    rows = training_rows(feat)
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    m, _ = attach_odds(test, odds, races)
    if m.empty:
        return pd.DataFrame()
    model = WinModel(features=list(FEATURE_SETS["morning"]), calibrate=config.CALIBRATE).fit(train)
    m["p"] = model.predict_win_prob(m).to_numpy()
    return simulate(m, "p")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """ルール・予算ごとの合計。最大の落ち込み = 累計収支がそれまでの最高からいちばん下がった額。"""
    rows = []
    for (rule, budget), g in df.groupby(["rule", "budget"], sort=False):
        cum = g["cum_profit"].to_numpy()
        peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
        bet = g["bet"].sum()
        rows.append({"ルール": rule, "1日の予算": budget, "日数": len(g), "購入額": bet, "払戻": g["payout"].sum(),
                     "収支": cum[-1], "回収率": g["payout"].sum() / bet if bet else float("nan"),
                     "最大の落ち込み": float((peak - cum).max())})
    return pd.DataFrame(rows)
