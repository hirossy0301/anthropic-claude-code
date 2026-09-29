"""Streamlit 画面: streamlit run app.py"""
import streamlit as st

from boatrace import config
from boatrace.cli import load_features, predict_race
from boatrace.model import WinModel

st.set_page_config(page_title="競艇 統計予想", layout="wide")
st.title("競艇 統計予想")

if not config.DB_PATH.exists() or not config.MODEL_PATH.exists():
    st.warning("DB またはモデルがありません。README の手順で ingest → train を実行してください。")
    st.stop()


@st.cache_data
def cached_features():
    return load_features()


@st.cache_resource
def cached_model():
    return WinModel.load(config.MODEL_PATH)


feat, races = cached_features()
model = cached_model()

dates = sorted(feat["race_date"].unique(), reverse=True)
c1, c2, c3 = st.columns(3)
race_date = c1.selectbox("日付", dates)
venues = sorted(feat.loc[feat["race_date"] == race_date, "venue"].unique())
venue = c2.selectbox("場", venues, format_func=lambda v: f"{v:02d} {config.VENUES.get(v, '')}")
race_nos = sorted(feat.loc[(feat["race_date"] == race_date) & (feat["venue"] == venue), "race_no"].unique())
race_no = c3.selectbox("レース", race_nos, format_func=lambda r: f"{r}R")

g, tri = predict_race(feat, model, race_date, venue, race_no)

left, right = st.columns([3, 2])
with left:
    st.subheader("1着確率")
    view = g[["lane", "racer_name", "racer_class", "nat_win_rate", "loc_win_rate",
              "motor_2_rate", "racer_avg_st", "win_prob", "finish"]].rename(columns={
        "lane": "枠", "racer_name": "選手", "racer_class": "級", "nat_win_rate": "全国勝率",
        "loc_win_rate": "当地勝率", "motor_2_rate": "モーター2率", "racer_avg_st": "平均ST",
        "win_prob": "1着確率", "finish": "結果"})
    st.dataframe(view.style.format({"1着確率": "{:.1%}", "平均ST": "{:.3f}"}), hide_index=True)
    st.bar_chart(g.set_index("lane")["win_prob"])
with right:
    st.subheader("3連単 上位")
    top = st.slider("表示点数", 5, 30, 10)
    st.dataframe(tri.head(top).style.format({"prob": "{:.2%}"}), hide_index=True)
    r = races[(races["race_date"] == race_date) & (races["venue"] == venue) & (races["race_no"] == race_no)]
    if not r.empty and r.iloc[0]["trifecta"]:
        res = r.iloc[0]
        rank = tri.index[tri["combo"] == res["trifecta"]]
        st.info(f"結果: {res['trifecta']} (払戻 {res['trifecta_payout']:,}円)"
                + (f" / 予想順位 {rank[0] + 1}位" if len(rank) else ""))

st.caption("確率は過去データからの統計的推定であり、的中や利益を保証するものではありません。")
