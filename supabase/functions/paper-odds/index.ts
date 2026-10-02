// EV 買いの記録 (お金は賭けない) の Edge Function。pg_cron から毎分呼ばれる (supabase/cron.sql)。
//
// 締切 4〜6 分前のレースを paper_plan から探し、公式サイトの単勝オッズを1レース1回だけ取得して、
// 6艇分の「1着確率 × オッズ = 期待値」を paper_records に書き込む。
// 公式サイトへの負荷を抑えるため、同じレースは paper_attempts で2回取りに行かないようにし、
// 複数のレースがあるときは間を3秒あける。
//
// オッズの読み取りは boatrace/odds.py の parse_win_odds と同じ規則 (tests/test_paper_supabase.py で一致を確認)。
// 環境変数: SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY (Supabase が自動で設定)、CRON_SECRET (自分で設定)

const ODDS_URL = "https://www.boatrace.jp/owpc/pc/race/oddstf";
const USER_AGENT = "boatrace-stats-research/0.1 (personal use)";
const FINAL_MARK = "締切時オッズ";
const INTERVAL_MS = 3000;

export function parseWinOdds(html: string): Record<number, number | null> {
  const start = html.indexOf("単勝オッズ");
  if (start < 0) return {};
  const end = html.indexOf("複勝オッズ");
  const section = html.slice(start, end > start ? end : undefined);
  const re = /is-boatColor([1-6])">\s*\d\s*<\/td>[\s\S]*?<td class="oddsPoint[^"]*">\s*([^<]*?)\s*<\/td>/g;
  const out: Record<number, number | null> = {};
  for (const m of section.matchAll(re)) {
    const v = Number(m[2]);
    // 欠場の艇は 0.0、数字でないものは null
    out[Number(m[1])] = m[2] !== "" && Number.isFinite(v) && v > 0 ? v : null;
  }
  return out;
}

export function isFinal(html: string): boolean {
  return html.includes(FINAL_MARK);
}

type PlanRow = {
  race_date: string; venue: number; race_no: number; lane: number;
  deadline: string; racer_name: string | null; win_prob: number;
};

async function rest(path: string, init: RequestInit = {}): Promise<Response> {
  const url = Deno.env.get("SUPABASE_URL")!;
  const key = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const res = await fetch(`${url}/rest/v1/${path}`, {
    ...init,
    headers: { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res;
}

async function handler(req: Request): Promise<Response> {
  const secret = Deno.env.get("CRON_SECRET");
  if (!secret || req.headers.get("x-cron-secret") !== secret) {
    return new Response("unauthorized", { status: 401 });
  }
  const now = Date.now();
  const lo = new Date(now + 4 * 60_000).toISOString();
  const hi = new Date(now + 6 * 60_000).toISOString();
  const plan: PlanRow[] = await (await rest(
    `paper_plan?select=*&deadline=gte.${encodeURIComponent(lo)}&deadline=lt.${encodeURIComponent(hi)}` +
      "&order=deadline,venue,race_no,lane",
  )).json();

  const races = new Map<string, PlanRow[]>();
  for (const r of plan) {
    const k = `${r.race_date}|${r.venue}|${r.race_no}`;
    races.set(k, [...(races.get(k) ?? []), r]);
  }

  const results: string[] = [];
  let first = true;
  for (const rows of races.values()) {
    const { race_date, venue, race_no } = rows[0];
    // 試行の印を先に入れる。既にあれば (前の分の呼び出しで取得済み) 飛ばす
    const claimed = await (await rest("paper_attempts", {
      method: "POST",
      headers: { Prefer: "resolution=ignore-duplicates,return=representation" },
      body: JSON.stringify([{ race_date, venue, race_no, status: "fetching" }]),
    })).json();
    if (claimed.length === 0) continue;

    if (!first) await new Promise((r) => setTimeout(r, INTERVAL_MS));
    first = false;
    let status = "ok";
    try {
      const hd = race_date.replaceAll("-", "");
      const res = await fetch(`${ODDS_URL}?rno=${race_no}&jcd=${String(venue).padStart(2, "0")}&hd=${hd}`, {
        headers: { "User-Agent": USER_AGENT },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const html = await res.text();
      const odds = parseWinOdds(html);
      if (Object.keys(odds).length === 0) {
        status = "no_odds";
      } else {
        if (isFinal(html)) status = "ok_final"; // 取得が締切後になった (記録はするが印を残す)
        const recordedAt = new Date().toISOString();
        const records = rows.map((r) => {
          const o = odds[r.lane] ?? null;
          return {
            race_date, venue, race_no, lane: r.lane, recorded_at: recordedAt, deadline: r.deadline,
            racer_name: r.racer_name, win_prob: r.win_prob, odds: o, ev: o === null ? null : r.win_prob * o,
          };
        });
        await rest("paper_records", {
          method: "POST",
          headers: { Prefer: "resolution=merge-duplicates,return=minimal" },
          body: JSON.stringify(records),
        });
      }
    } catch (e) {
      status = `error: ${e instanceof Error ? e.message : String(e)}`.slice(0, 200);
    }
    await rest(`paper_attempts?race_date=eq.${race_date}&venue=eq.${venue}&race_no=eq.${race_no}`, {
      method: "PATCH",
      headers: { Prefer: "return=minimal" },
      body: JSON.stringify({ status }),
    });
    results.push(`${race_date} ${venue} ${race_no}R: ${status}`);
  }
  return new Response(JSON.stringify({ checked: races.size, results }), {
    headers: { "Content-Type": "application/json" },
  });
}

// Node でのテスト (parseWinOdds の確認) では Deno が無いので起動しない
if (typeof Deno !== "undefined") Deno.serve(handler);
