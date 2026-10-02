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
