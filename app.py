"""Streamlit 画面: streamlit run app.py"""
import pandas as pd
import streamlit as st

from boatrace import config, serve
from boatrace.cli import load_features, predict_race
from boatrace.model import WinModel
from boatrace.venues import SURFACE_LABEL, VENUE_INFO, WATER_LABEL, WIND_DEG

st.set_page_config(page_title="競艇 統計予想", layout="wide")
st.title("競艇 統計予想")

MODES = {"morning": "朝の予想（前日までの情報）", "prerace": "直前の予想（直前情報を入力）"}

# デプロイ時は serve/ (GitHub Actions が毎朝更新) を、ローカルでは data/ の DB とモデルを使う
USE_SERVE = serve.available(config.SERVE_DIR)
MODEL_DIR = config.SERVE_DIR if USE_SERVE else config.MODEL_DIR


def model_file(mode: str):
    return MODEL_DIR / config.model_path(mode).name


if not (USE_SERVE or config.DB_PATH.exists()) or not all(model_file(m).exists() for m in MODES):
    st.warning("データまたはモデルがありません。README の手順で ingest → train を実行してください。")
    st.stop()


@st.cache_data
def cached_features(version: float):
    """version (ファイルの更新時刻) が変わると読み直す。"""
    if USE_SERVE:
        return serve.load(config.SERVE_DIR)
    feat, races = load_features()
    return feat, races, {}


@st.cache_resource
def cached_model(mode: str, version: float):
    return WinModel.load(model_file(mode))


def _mtime(path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


data_version = _mtime(config.SERVE_DIR / "features.parquet" if USE_SERVE else config.DB_PATH)
feat, races, meta = cached_features(data_version)
if meta.get("generated_at"):
    st.caption(f"データ更新: {meta['generated_at'].replace('T', ' ')}（{meta['first_date']}〜{meta['last_date']}）")

dates = sorted(feat["race_date"].unique(), reverse=True)
c1, c2, c3 = st.columns(3)
race_date = c1.selectbox("日付", dates)
venues = sorted(feat.loc[feat["race_date"] == race_date, "venue"].unique())
venue = c2.selectbox("場", venues, format_func=lambda v: f"{v:02d} {config.VENUES.get(v, '')}")
race_nos = sorted(feat.loc[(feat["race_date"] == race_date) & (feat["venue"] == venue), "race_no"].unique())
race_no = c3.selectbox("レース", race_nos, format_func=lambda r: f"{r}R")

race_rows = feat[(feat["race_date"] == race_date) & (feat["venue"] == venue) & (feat["race_no"] == race_no)]
first = race_rows.iloc[0]
info = VENUE_INFO.get(venue, {})
temp = first["temperature"]
st.caption(
    f"水面: {SURFACE_LABEL.get(info.get('surface'), '-')}・{WATER_LABEL.get(info.get('water'), '-')}"
    f" ／ 締切予定: {first.get('deadline') or '-'}"
    f" ／ 気温: {'-' if pd.isna(temp) else f'{temp:.1f}℃'}")

mode = st.radio("予想の種類", list(MODES), format_func=MODES.get, horizontal=True)
prerace = None
if mode == "prerace":
    st.markdown("**直前情報**（公式サイトの直前情報を見て入力。確定済みのレースは実際の値が入っています）")
    d1, d2, d3, d4 = st.columns(4)
    dirs = ["無風"] + list(WIND_DEG)
    cur_dir = first["wind_dir"] if first["wind_dir"] in dirs else "無風"
    wind_dir = d1.selectbox("風向", dirs, index=dirs.index(cur_dir))
    wind_speed = d2.number_input("風速 (m)", 0, 20, int(0 if pd.isna(first["wind_speed"]) else first["wind_speed"]))
    wave = d3.number_input("波高 (cm)", 0, 50, int(0 if pd.isna(first["wave"]) else first["wave"]))
    default_courses = ",".join(str(int(c)) for c in race_rows.sort_values("lane")["course"])
    courses_text = d4.text_input("進入コース（枠1〜6の順）", default_courses)
    try:
        courses = [int(c) for c in courses_text.split(",")]
    except ValueError:
        st.error("進入コースは「1,2,3,4,5,6」のようにカンマ区切りの数字で入力してください。")
        st.stop()
    prerace = {"wind_dir": None if wind_dir == "無風" else wind_dir,
               "wind_speed": 0 if wind_dir == "無風" else wind_speed, "wave": wave, "courses": courses}

try:
    g, tri = predict_race(feat, cached_model(mode, _mtime(model_file(mode))), race_date, venue, race_no, prerace)
except ValueError as e:
    st.error(str(e))
    st.stop()

left, right = st.columns([3, 2])
with left:
    st.subheader("1着確率")
    view = g[["lane", "course", "racer_name", "racer_class", "nat_win_rate", "loc_win_rate",
              "motor_2_rate", "racer_avg_st", "venue_course_win_actual", "win_prob", "finish"]].rename(columns={
        "lane": "枠", "course": "進入", "racer_name": "選手", "racer_class": "級", "nat_win_rate": "全国勝率",
        "loc_win_rate": "当地勝率", "motor_2_rate": "モーター2率", "racer_avg_st": "平均ST",
        "venue_course_win_actual": "場のコース1着率", "win_prob": "1着確率", "finish": "結果"})
    st.dataframe(view.style.format({
        "進入": "{:.0f}", "全国勝率": "{:.2f}", "当地勝率": "{:.2f}", "モーター2率": "{:.1f}",
        "平均ST": "{:.3f}", "場のコース1着率": "{:.1%}", "1着確率": "{:.1%}", "結果": "{:.0f}"}, na_rep="-"),
        hide_index=True)
    st.bar_chart(g.set_index("lane")["win_prob"])
with right:
    st.subheader("3連単 上位")
    top = st.slider("表示点数", 5, 30, 10)
    st.dataframe(tri.head(top).rename(columns={"combo": "組番", "prob": "確率"})
                 .style.format({"確率": "{:.2%}"}), hide_index=True)
    r = races[(races["race_date"] == race_date) & (races["venue"] == venue) & (races["race_no"] == race_no)]
    if not r.empty and r.iloc[0]["trifecta"]:
        res = r.iloc[0]
        rank = tri.index[tri["combo"] == res["trifecta"]]
        st.info(f"結果: {res['trifecta']} (払戻 {int(res['trifecta_payout']):,}円)"
                + (f" / 予想順位 {rank[0] + 1}位" if len(rank) else ""))

st.caption("確率は過去データからの統計的推定であり、的中や利益を保証するものではありません。")
