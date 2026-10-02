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
