from datetime import date

import numpy as np
import pytest

from boatrace import db
from boatrace.features import FEATURE_SETS, RACE_KEYS, training_rows
from boatrace.kimarite import TYPES, KimariteModel, add_context, race_outlook, report
from boatrace.model import WinModel
from boatrace.parsers import parse_result

from test_pipeline import RESULT, _feat


def test_parse_kimarite_from_header_line():
    # 実ファイルでは決まり手が成績の見出し行の末尾 (ﾚｰｽﾀｲﾑ の後) に入る
    text = RESULT.replace("ﾚｰｽﾀｲﾑ", "ﾚｰｽﾀｲﾑ まくり差し　　")
    _, races = parse_result(text, date(2026, 1, 5))
    assert races[0]["kimarite"] == "まくり差し"
    _, races = parse_result(RESULT, date(2026, 1, 5))
    assert races[0]["kimarite"] is None


def test_needs_reingest_and_history_features(tmp_path):
    conn, feat = _feat(tmp_path, 20)
    assert not db.needs_reingest(conn)
    conn.execute("UPDATE races SET kimarite = NULL")
    assert db.needs_reingest(conn)
    # 初日は履歴がないので事前分布そのもの
    first = feat[feat["race_date"] == "2026-01-01"]
    assert first["racer_nige_rate"].to_numpy() == pytest.approx(0.5)
    assert first["racer_makuri_rate"].to_numpy() == pytest.approx(0.03)


def test_context_uses_entry_course(tmp_path):
    _, feat = _feat(tmp_path, 5)
    g = feat[(feat["race_date"] == "2026-01-03") & (feat["venue"] == 1) & (feat["race_no"] == 1)]
    c = add_context(g).set_index("course")
    assert (c["in_racer_avg_st"] == c.loc[1, "racer_avg_st"]).all()
    assert np.isnan(c.loc[1, "inner_racer_avg_st"])
    assert c.loc[3, "inner_racer_avg_st"] == c.loc[2, "racer_avg_st"]
    assert c.loc[3, "outer_racer_avg_st"] == c.loc[4, "racer_avg_st"]


def test_outlook_sums_to_one_and_report(tmp_path):
    _, feat = _feat(tmp_path, 40)
    rows = training_rows(feat)
    km = KimariteModel().fit(rows)
    win = WinModel(features=list(FEATURE_SETS["prerace"])).fit(rows)
    key = rows[RACE_KEYS].iloc[0]
    g = feat[(feat[RACE_KEYS] == key.to_numpy()).all(axis=1)]
    p = win.predict_win_prob(g)
    joint, by_type = race_outlook(g, p, km)
    assert list(by_type.index) == TYPES
    assert joint.to_numpy().sum() == pytest.approx(1.0)
    assert (joint.sum(axis=1).to_numpy() == pytest.approx(p.to_numpy()))
    text = report(feat, "2026-02-01")
    for heading in ("## 勝った艇が分かっているとき", "## レース全体の決まり手", "## 「逃げ」の確率の較正"):
        assert heading in text


def test_place_model_probabilities(tmp_path):
    from boatrace.place import PlaceModel, harville_combos, compare
    conn, feat = _feat(tmp_path, 30)
    rows = training_rows(feat)
    pm = PlaceModel().fit(rows[rows["race_date"] < "2026-01-20"])
    test = rows[rows["race_date"] >= "2026-01-20"].copy()
    test["p_win"] = WinModel(features=list(FEATURE_SETS["morning"])).fit(rows).predict_win_prob(test).to_numpy()
    tri, ex = pm.combo_probs(test, "p_win")
    by_race = tri.groupby(RACE_KEYS)["prob"]
    assert (by_race.size() == 120).all() and np.allclose(by_race.sum(), 1)
    assert (ex.groupby(RACE_KEYS)["prob"].size() == 30).all() and np.allclose(ex.groupby(RACE_KEYS)["prob"].sum(), 1)
    # 2連単の a-b は、3連単の a-b-* の合計と一致する
    t = tri.assign(ab=tri["combo"].str.rsplit("-", n=1).str[0]).groupby(RACE_KEYS + ["ab"])["prob"].sum()
    e = ex.set_index(RACE_KEYS + ["combo"])["prob"]
    assert np.allclose(t.sort_index().to_numpy(), e.sort_index().to_numpy())
    h_tri, _ = harville_combos(test, "p_win")
    assert len(h_tri) == len(tri)
    table = compare(test, db.load_races(conn), "p_win", pm, n_races=100)
    assert list(table.index) == ["3連単", "2連単"]
