"""時系列分割のバックテスト。

学習期間 (cutoff より前) で学習し、検証期間 (cutoff 以降) で評価する。
ランダム分割にすると未来のレースで学習してしまうため使わない。
"""
from __future__ import annotations

import pandas as pd

from .features import RACE_KEYS, training_rows
from .model import WinModel, multiclass_log_loss, trifecta_probs


def evaluate(test: pd.DataFrame, races: pd.DataFrame, prob_col: str, top_n: int) -> dict:
    """1着的中率・対数損失、3連単上位 top_n 点買い (各100円) の的中率と回収率。"""
    races = races.set_index(RACE_KEYS)
    hits = bets = payout = top1 = 0
    for key, g in test.groupby(RACE_KEYS):
        top1 += int(g.loc[g[prob_col].idxmax(), "win"] == 1)
        if key not in races.index or pd.isna(races.loc[key, "trifecta"]):
            continue
        tri = trifecta_probs(g["lane"].tolist(), g[prob_col].tolist())
        picks = set(tri["combo"].head(top_n))
        bets += top_n
        if races.loc[key, "trifecta"] in picks:
            hits += 1
            payout += races.loc[key, "trifecta_payout"]
    n = test.groupby(RACE_KEYS).ngroups
    return {
        "races": n,
        "win_accuracy": top1 / n if n else float("nan"),
        "log_loss": multiclass_log_loss(test, prob_col),
        f"trifecta_top{top_n}_hit": hits / (bets / top_n) if bets else float("nan"),
        "roi": payout / (bets * 100) if bets else float("nan"),
    }


def run_backtest(feat: pd.DataFrame, races: pd.DataFrame, cutoff: str, top_n: int = 5) -> pd.DataFrame:
    rows = training_rows(feat)
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(f"cutoff {cutoff} で学習/検証データが空になります")

    # ベースライン: 学習期間の「場×枠の1着率」をそのまま確率にする
    lane_rate = train.groupby(["venue", "lane"])["win"].mean()
    test["p_base"] = [lane_rate.get((v, l), 1 / 6) for v, l in zip(test["venue"], test["lane"])]
    test["p_base"] /= test.groupby(RACE_KEYS)["p_base"].transform("sum")

    results = {"baseline(場×枠)": evaluate(test, races, "p_base", top_n)}
    for kind in ("logreg", "lgbm"):
        model = WinModel(kind=kind).fit(train)
        test[f"p_{kind}"] = model.predict_win_prob(test)
        results[kind] = evaluate(test, races, f"p_{kind}", top_n)
    return pd.DataFrame(results).T
