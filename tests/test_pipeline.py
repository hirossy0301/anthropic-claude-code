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


def test_program_boat_number_three_digits_without_space():
    # 実ファイル (2026-09-28 芦屋) の行。3桁のボート番号が直前のモーター2率と連結している
    text = ("21BBGN\n　１Ｒ  サンライズＶ          Ｈ１８００ｍ  電話投票締切予定０８：４４\n"
            "1 3232山川美由59香川47A2 5.66 33.73 6.66 53.13  8 44.68114 42.16 1233313     10\n21BEND\n")
    (r,) = parse_program(text, D)
    assert (r["motor_no"], r["motor_2_rate"], r["boat_no"], r["boat_2_rate"]) == (8, 44.68, 114, 42.16)
    assert r["deadline"] == "08:44" and r["race_name"] == "サンライズV"


def test_ingest_since_skips_older_files(tmp_path):
    generate(tmp_path / "raw", date(2026, 1, 1), 3, venues=(1,))
    conn = db.connect(tmp_path / "t.db")
    db.ingest_dir(conn, tmp_path / "raw", log=lambda *_: None, since=date(2026, 1, 3))
    days = {r[0] for r in conn.execute("SELECT DISTINCT race_date FROM entries")}
    assert days == {"2026-01-03"}


def test_serve_export_roundtrip_and_predict(tmp_path):
    from boatrace import serve
    from boatrace.cli import predict_race
    conn, feat = _feat(tmp_path, 20)
    races = db.load_races(conn)
    meta = serve.export(feat, races, tmp_path / "serve", days=3, today=date(2026, 1, 20))
    assert (meta["first_date"], meta["last_date"]) == ("2026-01-17", "2026-01-20")
    f, r, m = serve.load(tmp_path / "serve")
    assert set(f["race_date"]) == {"2026-01-17", "2026-01-18", "2026-01-19", "2026-01-20"}
    assert m["races"] == 4 * 2 * 12 and len(r) == 4 * 2 * 12
    # 書き出した特徴量だけで、DB と同じ予想になる
    model = WinModel(features=list(FEATURE_SETS["prerace"])).fit(training_rows(feat))
    key = ("2026-01-20", 1, 5)
    g_db, _ = predict_race(feat, model, *key, prerace={"wind_dir": "北", "wind_speed": 4})
    g_sv, _ = predict_race(f, model, *key, prerace={"wind_dir": "北", "wind_speed": 4})
    assert g_sv["win_prob"].to_numpy() == pytest.approx(g_db["win_prob"].to_numpy())


def test_results_not_final_are_not_saved(tmp_path, monkeypatch):
    from boatrace import config, download
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "LZH_DIR", tmp_path / "lzh")
    placeholder = "STARTK\n24KBGN\nボートレース大　村\nデータは、この場の全レース終了後に登録されます。\n24KEND\n"

    class Resp:
        status_code, content = 200, b"lzh"
        def raise_for_status(self):
            pass

    class Session:
        def get(self, *a, **k):
            return Resp()

    monkeypatch.setattr(download, "extract_lzh", lambda p: placeholder)
    assert download.fetch_day("K", date(2026, 10, 1), Session()) is None
    assert not download.raw_path("K", date(2026, 10, 1)).exists()  # 次回また取りに行く
    monkeypatch.setattr(download, "extract_lzh", lambda p: RESULT)
    assert download.fetch_day("K", date(2026, 10, 1), Session()).exists()
