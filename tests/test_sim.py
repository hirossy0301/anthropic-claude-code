import pandas as pd

from boatrace import sim


def _race(date, race_no, probs, odds, winner):
    return pd.DataFrame({"race_date": date, "venue": 1, "race_no": race_no, "lane": range(1, 7),
                         "p": probs, "odds": odds, "win": [int(l == winner) for l in range(1, 7)],
                         "win_payout": [o * 100 if l == winner else None for l, o in zip(range(1, 7), odds)]})


def test_allocate_splits_budget_and_falls_back_to_top_ev():
    day = pd.DataFrame({"ev": [1.1, 1.5, 1.3]})
    assert sim.allocate(day, 1000).tolist() == [300, 300, 300]  # 1000/3 → 300円ずつ、余り100円は買わない
    assert sim.allocate(day, 200).tolist() == [0, 100, 100]     # 足りない日は EV の高い順に100円ずつ
    assert sim.allocate(day.iloc[:0], 500).tolist() == []


def test_simulate_cumulative_profit():
    m = pd.concat([
        # 1日目: EV = 0.6*2=1.2 (1号艇, 勝ち), 0.1*12=1.2 (2号艇)。他は EV<1
        _race("2025-10-01", 1, [0.6, 0.1, 0.1, 0.1, 0.05, 0.05], [2.0, 12.0, 5.0, 5.0, 10.0, 10.0], winner=1),
        # 2日目: 対象なし
        _race("2025-10-02", 1, [0.5, 0.1, 0.1, 0.1, 0.1, 0.1], [1.5, 5.0, 5.0, 5.0, 5.0, 5.0], winner=2),
    ])
    df = sim.simulate(m, "p", budgets=(1000,), thresholds=(1.2,))
    d1, d2 = df.iloc[0], df.iloc[1]
    assert (d1["bets"], d1["bet"], d1["payout"]) == (2, 1000, 1000)  # 500円ずつ、1号艇 2.0倍 → 1000円
    assert (d2["bets"], d2["bet"], d2["cum_profit"]) == (0, 0, 0)
    s = sim.summary(df).iloc[0]
    assert s["回収率"] == 1.0 and s["日数"] == 2
