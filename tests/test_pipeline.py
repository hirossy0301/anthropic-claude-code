from datetime import date

import pandas as pd
import pytest

from boatrace import db
from boatrace.backtest import run_backtest
from boatrace.features import build_features, training_rows
from boatrace.model import WinModel, trifecta_probs
from boatrace.parsers import parse_program, parse_result
from boatrace.synthetic import generate

D = date(2026, 1, 5)

PROGRAM = """STARTB
12BBGN
ボートレース住之江
　１Ｒ  予選　　　　　　　　 Ｈ１８００ｍ  電話投票締切予定１０：３５
-------------------------------------------------------------------------------
1 4050田口節子35岡山53A1 6.88 49.66 7.60 57.14 66 32.41 17 30.00              7
2 3200山田　太郎48大阪52B1 4.10 22.00 3.90 20.50  8 28.10  5 31.20
12BEND
"""

RESULT = """STARTK
12KBGN
[払戻金]         ３連単
   1R       1-2-3      850
   1R       予選                 H1800m  晴　  風  北西　 3m  波　  2cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ
-------------------------------------------------------------------------------
  01  1 4050 田　口　　節　子 66   17  6.70   1    0.08     1.49.8
  F   2 3200 山　田　　太　郎  8    5  6.80   2    F.01     .  .
        単勝     1          170
        ３連単   1-2-3      850  人気     2
12KEND
"""


def test_parse_program():
    rows = parse_program(PROGRAM, D)
    assert len(rows) == 2
    r = rows[0]
    assert (r["venue"], r["race_no"], r["distance"], r["lane"], r["racer_id"]) == (12, 1, 1800, 1, 4050)
    assert (r["racer_name"], r["age"], r["branch"], r["weight"], r["racer_class"]) == ("田口節子", 35, "岡山", 53, "A1")
    assert r["nat_win_rate"] == 6.88 and r["motor_no"] == 66 and r["boat_2_rate"] == 30.0
    assert rows[1]["racer_name"] == "山田太郎" and rows[1]["motor_no"] == 8


def test_parse_result():
    results, races = parse_result(RESULT, D)
    assert len(races) == 1
    assert races[0]["trifecta"] == "1-2-3" and races[0]["trifecta_payout"] == 850
    assert races[0]["win_payout"] == 170 and races[0]["wind_speed"] == 3 and races[0]["wave"] == 2
    first, flying = results
    assert (first["finish"], first["lane"], first["start_timing"], first["course"]) == (1, 1, 0.08, 1)
    assert flying["finish"] is None and flying["finish_code"] == "F" and flying["flying"] == 1


def test_trifecta_probs_sum_to_one():
    tri = trifecta_probs([1, 2, 3, 4, 5, 6], [0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    assert len(tri) == 120
    assert tri["prob"].sum() == pytest.approx(1.0)
    assert tri.iloc[0]["combo"] == "1-2-3"


def test_no_leakage_in_history_features(tmp_path):
    generate(tmp_path / "raw", date(2026, 1, 1), 3, venues=(1,))
    conn = db.connect(tmp_path / "t.db")
    db.ingest_dir(conn, tmp_path / "raw", log=lambda *_: None)
    feat = build_features(db.load_frame(conn))
    first_day = feat[feat["race_date"] == "2026-01-01"]
    # 初日は過去データがないので事前分布そのもの
    assert first_day["racer_win"].to_numpy() == pytest.approx(1 / 6)
    assert first_day["venue_lane_win"].to_numpy() == pytest.approx(1 / 6)


def test_end_to_end_backtest(tmp_path):
    generate(tmp_path / "raw", date(2026, 1, 1), 40, venues=(1, 12))
    conn = db.connect(tmp_path / "t.db")
    db.ingest_dir(conn, tmp_path / "raw", log=lambda *_: None)
    feat = build_features(db.load_frame(conn))
    assert training_rows(feat).groupby(["race_date", "venue", "race_no"]).ngroups == 40 * 2 * 12
    res = run_backtest(feat, db.load_races(conn), cutoff="2026-02-01")
    # モデルは 1/6 のランダム予想より明確に良いはず
    assert res.loc["lgbm", "win_accuracy"] > 0.35
    model = WinModel().fit(training_rows(feat))
    p = model.predict_win_prob(feat)
    assert p.groupby([feat["race_date"], feat["venue"], feat["race_no"]]).sum().round(6).eq(1).all()
