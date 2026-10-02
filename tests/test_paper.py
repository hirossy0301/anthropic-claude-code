from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from boatrace import config, paper, serve
from boatrace.features import FEATURE_SETS, training_rows
from boatrace.model import WinModel
from boatrace.serve import JST

from test_pipeline import _feat


def test_pick_venues_is_fixed_by_date():
    v = list(range(1, 13))
    assert paper.pick_venues(v, date(2026, 10, 3), 4) == paper.pick_venues(v, date(2026, 10, 3), 4)
    assert len(paper.pick_venues(v, date(2026, 10, 3), 4)) == 4
    assert paper.pick_venues([5, 7], date(2026, 10, 3), 4) == [5, 7]


@pytest.fixture
def served(tmp_path, monkeypatch):
    conn, feat = _feat(tmp_path, 15)
    races = __import__("boatrace.db", fromlist=["db"]).load_races(conn)
    sdir = tmp_path / "serve"
    serve.export(feat, races, sdir, days=3, today=date(2026, 1, 14))
    WinModel(features=list(FEATURE_SETS["morning"])).fit(training_rows(feat)).save(sdir / "model_morning.pkl")
    monkeypatch.setattr(config, "SERVE_DIR", sdir)
    return sdir


def test_run_records_before_deadline_and_resumes(served, tmp_path):
    day = date(2026, 1, 14)
    clock = {"t": datetime(2026, 1, 14, 9, 0, tzinfo=JST)}
    calls = []

    def fetch(d, v, r):
        calls.append((d, v, r, clock["t"]))
        return {"final": False, "win": {1: 1.5, 2: 4.0, 3: 6.0, 4: 10.0, 5: 20.0, 6: 30.0}}

    def sleep(sec):
        clock["t"] += timedelta(seconds=sec)

    out = tmp_path / "paper"
    n = paper.run(day, n_venues=1, out_dir=out, log=lambda *_: None, now_fn=lambda: clock["t"],
                  sleep_fn=sleep, fetch=fetch)
    assert n == 12 and len(calls) == 12
    df = pd.read_csv(out / "2026-01-14.csv")
    assert len(df) == 72 and df["ev"].notna().all()
    # 取得は締切の5分前
    first_deadline = sorted(df["deadline"])[0]
    assert calls[0][3].strftime("%H:%M") == (datetime.strptime(first_deadline, "%H:%M") - timedelta(minutes=5)).strftime("%H:%M")
    # 再起動しても記録済みのレースは取り直さない
    assert paper.run(day, n_venues=1, out_dir=out, log=lambda *_: None, now_fn=lambda: clock["t"],
                     sleep_fn=sleep, fetch=fetch) == 0


def test_settle_and_report(served, tmp_path):
    out = tmp_path / "paper"
    feat, races, _ = serve.load(served)
    g = feat[(feat["race_date"] == "2026-01-13") & (feat["venue"] == 1) & (feat["race_no"] == 1)]
    rows = paper.record_rows(g, pd.Series([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
                             {1: 1.5, 2: 12.0, 3: 12.0, 4: 12.0, 5: 12.0, 6: 12.0}, datetime.now(JST))
    out.mkdir()
    rows.to_csv(out / "2026-01-13.csv", index=False)
    assert paper.report(paper.load_records(out)).startswith("精算済みの記録がまだありません")
    assert paper.settle(feat, races, out) == 1
    text = paper.report(paper.load_records(out))
    assert "EV≥1" in text and "5" in text.split("| EV≥1 |")[1]  # 2〜6号艇の5点を買っている
    assert paper.settle(feat, races, out) == 0
