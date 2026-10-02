import json
import shutil
import subprocess
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from boatrace import odds, paper, serve, supa
from boatrace.model import WinModel
from boatrace.serve import JST

from test_paper import served  # noqa: F401  (fixture)

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (ROOT / "tests" / "fixtures" / "oddstf_20260928_24_01.html").read_text(encoding="utf-8")
TS = ROOT / "supabase" / "functions" / "paper-odds" / "index.ts"


@pytest.mark.skipif(shutil.which("node") is None, reason="node が無い")
@pytest.mark.parametrize("html", [
    FIXTURE,
    FIXTURE.replace(">25.8<", ">0.0<"),
    FIXTURE.replace(">25.8<", ">欠場<"),
    FIXTURE.replace("締切時オッズ", "オッズ更新時間"),
    "<html>データがありません</html>",
])
def test_edge_function_parser_matches_python(tmp_path, html):
    """Edge Function (TypeScript) のオッズの読み取りが Python 版と同じ結果になる。"""
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text(
        "import { readFileSync } from 'node:fs';\n"
        f"const m = await import({json.dumps(TS.as_uri())});\n"
        "const h = readFileSync(process.argv[2], 'utf8');\n"
        "console.log(JSON.stringify({win: m.parseWinOdds(h), final: m.isFinal(h)}));\n", encoding="utf-8")
    res = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", str(script), str(page)],
                         capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    assert {int(k): v for k, v in out["win"].items()} == odds.parse_win_odds(html)
    assert out["final"] == odds.is_final(html)


def test_plan_rows(served):  # noqa: F811
    feat, _, _ = serve.load(served)
    model = WinModel.load(served / "model_morning.pkl")
    rows = paper.plan_rows(feat, model, date(2026, 1, 14), n_venues=4)
    df = pd.DataFrame(rows)
    assert len(df) == 2 * 12 * 6  # 合成データは2場
    assert df.groupby(["venue", "race_no"])["win_prob"].sum().round(6).eq(1).all()
    assert datetime.fromisoformat(df["deadline"].iloc[0]).utcoffset().total_seconds() == 9 * 3600
    assert paper.plan_rows(feat, model, date(2030, 1, 1)) == []


def test_settle_rows(served):  # noqa: F811
    feat, races, _ = serve.load(served)
    recs = pd.DataFrame({"race_date": ["2026-01-13"] * 6 + ["2030-01-01"] * 6, "venue": 1,
                         "race_no": 1, "lane": list(range(1, 7)) * 2, "finish": None, "win_payout": None})
    rows = paper.settle_rows(recs, feat, races)
    assert len(rows) == 6 and all(r["race_date"] == "2026-01-13" for r in rows)
    assert sum(r["finish"] == 1 for r in rows) == 1 and rows[0]["win_payout"] > 0
    assert paper.settle_rows(recs.iloc[:0], feat, races) == []


def test_supa_headers_and_paging(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, data):
            self.data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def get(url, params, headers, timeout):
        calls.append(("get", params["offset"], headers))
        return Resp([{"x": 1}] * (2 if params["offset"] < 4 else 1))

    def post(url, json, timeout, headers):
        calls.append(("post", len(json), headers))
        return Resp(None)

    monkeypatch.setattr(supa.requests, "get", get)
    monkeypatch.setattr(supa.requests, "post", post)
    assert len(supa.select("https://x.supabase.co", "sb_secret_abc", "t", page=2)) == 5
    assert "Authorization" not in calls[0][2] and calls[0][2]["apikey"] == "sb_secret_abc"
    supa.upsert("https://x.supabase.co", "eyJabc", "t", [{"a": 1}] * 3, chunk=2)
    posts = [c for c in calls if c[0] == "post"]
    assert [p[1] for p in posts] == [2, 1]
    assert posts[0][2]["Authorization"] == "Bearer eyJabc"
    assert "merge-duplicates" in posts[0][2]["Prefer"]
