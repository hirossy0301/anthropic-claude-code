"""EV 買いの記録 (お金は賭けない) の画面。記録は Supabase にあり、Edge Function が締切5分前に書き込む。"""
import pandas as pd
import streamlit as st

from boatrace import config, paper, supa
from boatrace.serve import today_jst

st.title("EV 買いの記録")
st.caption("お金は賭けずに、締切5分前の単勝オッズで「期待値 ≥ 1」の艇を記録し、実際の結果で精算しています。"
           "毎日4場（無作為）を記録します。精算は翌朝です。")


def _secret(name: str):
    try:
        return st.secrets.get(name)
    except Exception:  # secrets.toml が無い (手元で起動したとき)
        return None


url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_ANON_KEY")
if not url or not key:
    st.info("Supabase の設定がありません。README の「EV 買いの記録（Supabase）」の手順で設定してください。")
    st.stop()


@st.cache_data(ttl=300, show_spinner="記録を読み込み中…")
def load() -> pd.DataFrame:
    return pd.DataFrame(supa.select(url.rstrip("/"), key, "paper_records", {"select": "*", "order": "deadline"}))


try:
    rec = load()
except Exception as e:
    st.error(f"記録を読み込めませんでした: {e}")
    st.stop()
if rec.empty:
    st.info("まだ記録がありません。")
    st.stop()
rec["race_date"] = rec["race_date"].astype(str)

st.markdown(paper.report(rec).replace("# EV 買いの記録", "### 集計", 1))

today = today_jst().isoformat()
t = rec[(rec["race_date"] == today) & (rec["ev"] >= 1)].copy()
st.subheader(f"今日の EV≥1（{today}）")
if t.empty:
    st.write("まだありません。")
else:
    t["場"] = t["venue"].map(config.VENUES)
    t["締切"] = pd.to_datetime(t["deadline"]).dt.tz_convert("Asia/Tokyo").dt.strftime("%H:%M")
    st.dataframe(t[["締切", "場", "race_no", "lane", "racer_name", "win_prob", "odds", "ev", "finish"]].rename(columns={
        "race_no": "R", "lane": "枠", "racer_name": "選手", "win_prob": "1着確率", "odds": "オッズ(5分前)",
        "ev": "期待値", "finish": "着"}).style.format(
        {"1着確率": "{:.1%}", "オッズ(5分前)": "{:.1f}", "期待値": "{:.2f}", "着": "{:.0f}"}, na_rep="-"),
        hide_index=True)
st.caption("記録のオッズは私的使用の範囲で取得したものです。他の人に配ったり転載したりしないでください。")
