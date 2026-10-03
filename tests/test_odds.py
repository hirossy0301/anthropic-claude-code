from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boatrace import db, odds
from boatrace.features import training_rows

from test_pipeline import _feat

FIXTURE = (Path(__file__).parent / "fixtures" / "oddstf_20260928_24_01.html").read_text(encoding="utf-8")


def test_parse_win_odds_real_page():
    # 2026-09-28 大村 1R: 1号艇が1着、成績ファイルの単勝払戻は 190円 (= 1.9倍)
    assert odds.parse_win_odds(FIXTURE) == {1: 1.9, 2: 25.8, 3: 1.8, 4: 19.3, 5: 7.4, 6: 20.3}
    assert odds.is_final(FIXTURE)
    assert odds.parse_win_odds(FIXTURE.replace(">25.8<", ">欠場<"))[2] is None
    # 欠場の艇は 0.0 と表示される (2026-09-25 常滑 1R の3・6号艇)
    assert odds.parse_win_odds(FIXTURE.replace(">25.8<", ">0.0<"))[2] is None
    assert odds.parse_win_odds("<html>データがありません</html>") == {}


class FakeResp:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, text):
        self.text, self.urls = text, []

    def get(self, url, timeout):
        self.urls.append(url)
        return FakeResp(self.text)


def test_fetch_caches_only_final_odds(tmp_path):
    s = FakeSession(FIXTURE)
    rec = odds.fetch_win_odds("2026-09-28", 24, 1, session=s, odds_dir=tmp_path)
    assert s.urls == ["https://www.boatrace.jp/owpc/pc/race/oddstf?rno=1&jcd=24&hd=20260928"]
    assert rec["final"] and rec["win"][1] == 1.9
    # 2回目はキャッシュから (通信しない)
    assert odds.fetch_win_odds("2026-09-28", 24, 1, session=s, odds_dir=tmp_path)["win"]["1"] == 1.9
    assert len(s.urls) == 1
    # 締切前のオッズは変わるので保存しない
    s2 = FakeSession(FIXTURE.replace("締切時オッズ", "オッズ更新時間"))
    assert not odds.fetch_win_odds("2026-09-28", 24, 2, session=s2, odds_dir=tmp_path)["final"]
    assert not odds.cache_path("2026-09-28", 24, 2, tmp_path).exists()
    df = odds.load_cached(tmp_path)
    assert len(df) == 6 and df["odds"].tolist() == [1.9, 25.8, 1.8, 19.3, 7.4, 20.3]


def test_sample_is_reproducible_and_respects_budget(tmp_path):
    races = pd.DataFrame({"race_date": ["2025-09-30"] + ["2025-10-01"] * 12 + ["2025-10-02"] * 12,
                          "venue": 1, "race_no": [1] + list(range(1, 13)) * 2,
                          "win_payout": [100] * 24 + [None]})
    keys = odds.sample_races(races, "2025-10-01", 10, seed=0)
    assert keys == odds.sample_races(races, "2025-10-01", 10, seed=0)
    assert len(keys) == 10 and all(k[0] >= "2025-10-01" for k in keys)
    assert ("2025-10-02", 1, 12) not in odds.sample_races(races, "2025-10-01", 30)  # 未確定は除く
    stats = odds.fetch_sample(keys, budget_minutes=0, odds_dir=tmp_path, log=lambda *_: None, interval=0)
    assert stats["fetched"] == 0 and stats["remaining"] == 10


def test_roi_ci():
    from boatrace.ev import roi_ci
    r = pd.DataFrame({"bet": [100] * 1000, "payout": [0, 200] * 500})
    roi, lo, hi = roi_ci(r)
    assert roi == pytest.approx(1.0) and lo < 1.0 < hi


def test_ev_report_on_synthetic_odds(tmp_path):
    from boatrace.ev import report
    conn, feat = _feat(tmp_path, 40)
    races = db.load_races(conn)
    rows = training_rows(feat)
    test = rows[rows["race_date"] >= "2026-02-01"]
    # 市場の確率 = 枠ごとの1着率 (控除率25%) とした合成オッズ
    rate = rows.groupby("lane")["win"].mean()
    o = test[["race_date", "venue", "race_no", "lane"]].copy()
    o["odds"] = np.round(0.75 / o["lane"].map(rate), 1)
    text = report(feat, races, o, "2026-02-01")
    for heading in ("## 確率の当たり具合", "## モデルに市場が織り込んでいない情報があるか",
                    "## 買い方ごとの回収率", "全艇を買う", "EV≥1.2", "## 注意"):
        assert heading in text
    assert "オッズを取得済みのレースがありません" in report(feat, races, o.iloc[:0], "2026-02-01")


def test_parse_exacta_and_trifecta_real_pages():
    from boatrace.model import exacta_probs, trifecta_probs
    fix = Path(__file__).parent / "fixtures"
    tri_html = (fix / "odds3t_20260928_19_12.html").read_text(encoding="utf-8")
    ex_html = (fix / "odds2tf_20260928_19_12.html").read_text(encoding="utf-8")
    tri, ex = odds.parse_trifecta_odds(tri_html), odds.parse_exacta_odds(ex_html)
    assert len(tri) == 120 and len(ex) == 30
    # 2026-09-28 下関 12R の払戻: 3連単 6-4-1 48,500円、2連単 6-4 8,440円
    assert tri[(6, 4, 1)] == 485.0 and ex[(6, 4)] == 84.4
    assert tri[(1, 2, 3)] == 6.6 and tri[(2, 1, 3)] == 27.1 and ex[(1, 2)] == 3.0
    # 控除率 約25% なので 1/オッズ の合計は約 1.33
    assert sum(1 / v for v in tri.values()) == pytest.approx(1.336, abs=0.01)
    assert odds.is_final(tri_html)
    # モデル側の組番と同じ表記で引けること
    lanes, p = [1, 2, 3, 4, 5, 6], [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
    assert set(trifecta_probs(lanes, p)["combo"]) == {"-".join(map(str, k)) for k in tri}
    e = exacta_probs(lanes, p)
    assert set(e["combo"]) == {"-".join(map(str, k)) for k in ex}
    assert e["prob"].sum() == pytest.approx(1.0)
