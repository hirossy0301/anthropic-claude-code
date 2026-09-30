from datetime import date

import pandas as pd
import pytest

from boatrace import db
from boatrace.backtest import run_backtest
from boatrace.features import FEATURE_SETS, apply_prerace, build_features, training_rows
from boatrace.venues import VENUE_INFO
from boatrace.weather import parse_hourly
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
    assert r["deadline"] == "10:35"


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
    assert res.loc["lgbm 朝", "win_accuracy"] > 0.35
    model = WinModel().fit(training_rows(feat))
    p = model.predict_win_prob(feat)
    assert p.groupby([feat["race_date"], feat["venue"], feat["race_no"]]).sum().round(6).eq(1).all()


def _feat(tmp_path, days, venues=(1, 24)):
    generate(tmp_path / "raw", date(2026, 1, 1), days, venues=venues)
    conn = db.connect(tmp_path / "t.db")
    db.ingest_dir(conn, tmp_path / "raw", log=lambda *_: None)
    return conn, build_features(db.load_frame(conn))


def test_venue_info_complete():
    assert set(VENUE_INFO) == set(range(1, 25))
    assert {v["water"] for v in VENUE_INFO.values()} == {"fresh", "brackish", "salt"}
    assert {v["surface"] for v in VENUE_INFO.values()} == {"lake", "sea", "river"}


def test_parse_hourly_open_meteo():
    payload = {"hourly": {"time": ["2026-09-01T00:00", "2026-09-01T13:00"], "temperature_2m": [24.1, None]}}
    assert parse_hourly(12, payload) == [{"venue": 12, "race_date": "2026-09-01", "hour": 0, "temperature": 24.1}]


def test_temperature_joined_at_deadline_hour(tmp_path):
    conn, _ = _feat(tmp_path, 1, venues=(12,))
    db.upsert(conn, "weather", [{"venue": 12, "race_date": "2026-01-01", "hour": 10, "temperature": 8.5},
                                {"venue": 12, "race_date": "2026-01-01", "hour": 11, "temperature": 9.5}])
    feat = build_features(db.load_frame(conn))
    by_race = feat.groupby("race_no")["temperature"].first()
    assert by_race[1] == 8.5  # 締切 10:35 -> 10時
    assert by_race[2] == 9.5  # 締切 11:05 -> 11時
    assert by_race[3] == 9.5  # 締切 11:35 -> 11時
    assert pd.isna(by_race[4])  # 締切 12:05 -> 12時 (データなし)


def test_wind_and_water_features(tmp_path):
    _, feat = _feat(tmp_path, 2)
    assert feat["wind_speed"].notna().all() and feat["wind_dir"].notna().all()
    assert feat[["wind_sin", "wind_cos"]].abs().max().max() <= 1
    assert set(feat.loc[feat["venue"] == 1, "water"]) == {0}  # 桐生 = 淡水
    assert set(feat.loc[feat["venue"] == 24, "surface"]) == {2}  # 大村 = 海
    assert (feat.loc[feat["venue"] == 24, "weight_diff_fresh"] == 0).all()


def test_venue_course_win_uses_only_past_days(tmp_path):
    _, feat = _feat(tmp_path, 3)
    assert feat.loc[feat["race_date"] == "2026-01-01", "venue_course_win"].to_numpy() == pytest.approx(1 / 6)
    day2 = feat[(feat["race_date"] == "2026-01-02") & (feat["venue"] == 1)]
    day1 = feat[(feat["race_date"] == "2026-01-01") & (feat["venue"] == 1)]
    wins_c1 = day1.loc[day1["course"] == 1, "win"]
    expected = (wins_c1.sum() + 50 / 6) / (wins_c1.count() + 50)
    assert day2.loc[day2["lane"] == 1, "venue_course_win"].iloc[0] == pytest.approx(expected)


def test_apply_prerace_overrides(tmp_path):
    _, feat = _feat(tmp_path, 2)
    g = feat[(feat["race_date"] == "2026-01-02") & (feat["venue"] == 1) & (feat["race_no"] == 1)]
    out = apply_prerace(g, wind_dir="北", wind_speed=5, wave=3, courses=[1, 2, 4, 3, 5, 6])
    assert out["wind_cos"].iloc[0] == pytest.approx(1.0) and out["wind_sin"].iloc[0] == pytest.approx(0.0)
    rate = dict(zip(g["lane"], g["venue_course_win"]))
    assert out.loc[out["lane"] == 4, "venue_course_win_actual"].iloc[0] == pytest.approx(rate[3])
    with pytest.raises(ValueError):
        apply_prerace(g, courses=[1, 1, 3, 4, 5, 6])


def test_migration_adds_deadline_column(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE entries (race_date TEXT, venue INTEGER, race_no INTEGER, lane INTEGER)")
    old.commit()
    old.close()
    cols = {r[1] for r in db.connect(path).execute("PRAGMA table_info(entries)")}
    assert "deadline" in cols


def test_both_models_predict(tmp_path):
    _, feat = _feat(tmp_path, 20)
    rows = training_rows(feat)
    for mode, features in FEATURE_SETS.items():
        p = WinModel(features=list(features)).fit(rows).predict_win_prob(rows)
        assert p.notna().all()
