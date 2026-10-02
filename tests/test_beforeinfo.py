from pathlib import Path

import pytest

from boatrace.beforeinfo import parse_beforeinfo, wind_direction

FIX = Path(__file__).parent / "fixtures"


# 2026-09-28 の直前情報 (11R時点) の (風の番号, 向きの番号) と、競走成績に記録された 11R の風向
CALIBRATION = [
    (2, 6, 16, "北西"), (3, 4, 4, "南"), (4, 5, 5, "南"), (5, 3, 9, "北東"), (13, 14, 10, "西"),
    (14, 5, 15, "北西"), (15, 9, 7, "南西"), (16, 13, 13, "南"), (19, 1, 11, "北西"),
]


@pytest.mark.parametrize("venue,wind,direction,expected", CALIBRATION)
def test_wind_direction_matches_results(venue, wind, direction, expected):
    assert wind_direction(wind, direction) == expected


def test_parse_beforeinfo_real_pages():
    a = parse_beforeinfo((FIX / "beforeinfo_20260928_19_12.html").read_text(encoding="utf-8"))
    assert a["updated"] == "11R時点"
    assert (a["wind_dir"], a["wind_speed"], a["wave"]) == ("北西", 1.0, 1.0)
    assert (a["temperature"], a["water_temperature"]) == (22.0, 26.0)
    assert a["courses"] == [1, 2, 3, 4, 5, 6]
    assert a["exhibition_st"][4] == "F.07" and a["exhibition_st"][1] == ".19"
    calm = parse_beforeinfo((FIX / "beforeinfo_20260928_09_12.html").read_text(encoding="utf-8"))
    assert calm["wind_dir"] is None and calm["wind_speed"] == 0.0  # is-wind17 = 無風


def test_parse_beforeinfo_entry_change_and_missing():
    html = (FIX / "beforeinfo_20260928_19_12.html").read_text(encoding="utf-8")
    # 展示で4号艇が3コース、3号艇が4コースに入った (上から 1,2,4,3,5,6 の順)
    swapped = (html.replace('is-type3">3<', 'is-typeX">X<').replace('is-type4">4<', 'is-type3">3<')
               .replace('is-typeX">X<', 'is-type4">4<'))
    assert parse_beforeinfo(swapped)["courses"] == [1, 2, 4, 3, 5, 6]
    assert parse_beforeinfo(html.replace('is-type6">6<', 'is-type5">5<'))["courses"] is None  # 6艇そろわない
    empty = parse_beforeinfo("<html>データはありません</html>")
    assert empty["wind_speed"] is None and empty["courses"] is None


def test_exhibition_times_and_features(tmp_path):
    a = parse_beforeinfo((FIX / "beforeinfo_20260928_19_12.html").read_text(encoding="utf-8"))
    # 競走成績ファイルの展示タイムと一致 (2026-09-28 下関 12R)
    assert a["exhibition_times"] == {1: 6.82, 2: 6.85, 3: 6.83, 4: 6.74, 5: 6.82, 6: 6.77}

    from test_pipeline import _feat
    from boatrace.features import apply_prerace
    _, feat = _feat(tmp_path, 5)
    assert feat["exh_time"].notna().all() and feat["exh_rank"].between(1, 6).all()
    g = feat[(feat["race_date"] == "2026-01-05") & (feat["venue"] == 1) & (feat["race_no"] == 1)]
    g2 = apply_prerace(g, exhibition_times=a["exhibition_times"])
    assert g2["exh_time"].tolist() == [6.82, 6.85, 6.83, 6.74, 6.82, 6.77]
    assert g2["exh_rank"].tolist() == [3, 6, 5, 1, 3, 2]
    assert g2["exh_diff"].sum() == pytest.approx(0)
