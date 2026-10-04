"""予想の画面 (app.py から st.navigation で呼ばれる)。"""
import pandas as pd
import streamlit as st

from boatrace import config, serve
from boatrace.cli import KIMARITE_MODEL, load_features, predict_race
from boatrace.features import KIMARITE_HISTORY
from boatrace.kimarite import DECIDED_AT_TURN1, KimariteModel, race_outlook
from boatrace.model import WinModel, exacta_probs, trifecta_probs
from boatrace.venues import SURFACE_LABEL, VENUE_INFO, WATER_LABEL, WIND_DEG

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


@st.cache_data(ttl=60, show_spinner="公式サイトから直前情報を取得中…")
def cached_beforeinfo(race_date: str, venue: int, race_no: int) -> dict:
    """同じレースは60秒以内なら取り直さない (公式サイトへのアクセスを減らすため)。"""
    from boatrace.beforeinfo import fetch_beforeinfo
    return fetch_beforeinfo(race_date, venue, race_no)


if mode == "prerace":
    st.markdown("**直前情報**（「直前情報を取得」で公式サイトの値を入れるか、手で入力。確定済みのレースは実際の値が入っています）")
    dirs = ["無風"] + list(WIND_DEG)
    k = f"{race_date}_{venue}_{race_no}"  # レースを変えたら入力欄も初期値に戻す
    if f"wind_dir_{k}" not in st.session_state:
        st.session_state[f"wind_dir_{k}"] = first["wind_dir"] if first["wind_dir"] in dirs else "無風"
        st.session_state[f"wind_speed_{k}"] = int(0 if pd.isna(first["wind_speed"]) else first["wind_speed"])
        st.session_state[f"wave_{k}"] = int(0 if pd.isna(first["wave"]) else first["wave"])
        st.session_state[f"courses_{k}"] = ",".join(str(int(c)) for c in race_rows.sort_values("lane")["course"])
        exh = race_rows.sort_values("lane")["exh_time"] if "exh_time" in race_rows else pd.Series(dtype=float)
        st.session_state[f"exh_{k}"] = ",".join(f"{t:.2f}" for t in exh) if len(exh) and exh.notna().all() else ""
    if st.button("直前情報を取得（公式サイト）", help="このレースの直前情報ページを1回だけ取得し、風・波・スタート展示の進入を入れます"):
        try:
            bi = cached_beforeinfo(race_date, int(venue), int(race_no))
        except Exception as e:
            st.error(f"直前情報を取得できませんでした: {e}")
        else:
            if bi["wind_speed"] is None and bi["courses"] is None:
                st.warning("直前情報がまだ発表されていません（展示航走の後に出ます）。")
            else:
                st.session_state[f"wind_dir_{k}"] = bi["wind_dir"] or "無風"
                if bi["wind_speed"] is not None:
                    st.session_state[f"wind_speed_{k}"] = int(round(bi["wind_speed"]))
                if bi["wave"] is not None:
                    st.session_state[f"wave_{k}"] = int(round(bi["wave"]))
                if bi["courses"]:
                    st.session_state[f"courses_{k}"] = ",".join(map(str, bi["courses"]))
                if bi["exhibition_times"] and len(bi["exhibition_times"]) == 6:
                    st.session_state[f"exh_{k}"] = ",".join(f"{bi['exhibition_times'][lane]:.2f}" for lane in range(1, 7))
                st.session_state[f"before_{k}"] = bi
    bi = st.session_state.get(f"before_{k}")
    if bi:
        sts = " ".join(f"{lane}号艇 {v or '-'}" for lane, v in sorted((bi["exhibition_st"] or {}).items()))
        st.caption(f"公式の直前情報（{bi['updated'] or '-'}）: 気温 {bi['temperature'] or '-'}℃ ／ "
                   f"水温 {bi['water_temperature'] or '-'}℃ ／ 展示のST: {sts or '-'}"
                   + ("" if bi["courses"] else " ／ 展示の進入が6艇そろっていないため、進入コースは入れていません"))
    d1, d2, d3, d4 = st.columns(4)
    wind_dir = d1.selectbox("風向", dirs, key=f"wind_dir_{k}")
    wind_speed = d2.number_input("風速 (m)", 0, 20, key=f"wind_speed_{k}")
    wave = d3.number_input("波高 (cm)", 0, 50, key=f"wave_{k}")
    courses_text = d4.text_input("進入コース（枠1〜6の順）", key=f"courses_{k}")
    exh_text = st.text_input("展示タイム（枠1〜6の順、例: 6.82,6.85,6.83,6.74,6.82,6.77。空欄なら使わない）", key=f"exh_{k}")
    try:
        courses = [int(c) for c in courses_text.split(",")]
    except ValueError:
        st.error("進入コースは「1,2,3,4,5,6」のようにカンマ区切りの数字で入力してください。")
        st.stop()
    exhibition_times = None
    if exh_text.strip():
        try:
            values = [float(x) for x in exh_text.split(",")]
            if len(values) != 6:
                raise ValueError
            exhibition_times = dict(zip(range(1, 7), values))
        except ValueError:
            st.error("展示タイムは「6.82,6.85,…」のように6艇分をカンマ区切りで入力してください。")
            st.stop()
    prerace = {"wind_dir": None if wind_dir == "無風" else wind_dir,
               "wind_speed": 0 if wind_dir == "無風" else wind_speed, "wave": wave, "courses": courses,
               "exhibition_times": exhibition_times}

try:
    # 展示タイムがあり、展示タイムありのモデルが学習済みならそちらを使う
    model_name = mode
    if prerace and prerace.get("exhibition_times") and model_file("prerace_exh").exists():
        model_name = "prerace_exh"
    if mode == "prerace":
        st.caption("使ったモデル: " + ("直前の予想 ＋ 展示タイム" if model_name == "prerace_exh" else
                                    "直前の予想（展示タイムなし）"))
    g, tri = predict_race(feat, cached_model(model_name, _mtime(model_file(model_name))), race_date, venue, race_no, prerace)
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



@st.cache_resource
def cached_kimarite_model(version: float):
    return KimariteModel.load(MODEL_DIR / KIMARITE_MODEL)


kim_path = MODEL_DIR / KIMARITE_MODEL
# 決まり手の列を書き出す前の serve/ (デプロイ直後) では表示しない
if kim_path.exists() and all(c in g for c, *_ in KIMARITE_HISTORY):
    st.subheader("1マークの展開予想（決まり手）")
    joint, by_type = race_outlook(g, g["win_prob"], cached_kimarite_model(_mtime(kim_path)))
    k1, k2 = st.columns([2, 3])
    with k1:
        st.dataframe(by_type.rename("確率").to_frame().rename_axis("決まり手")
                     .style.format({"確率": "{:.1%}"}), width="stretch")
    with k2:
        tbl = joint[DECIDED_AT_TURN1].copy()
        tbl["抜き・恵まれ"] = joint["抜き"] + joint["恵まれ"]
        tbl.insert(0, "1マークで先頭", joint[DECIDED_AT_TURN1].sum(axis=1))
        tbl.index.name = "枠"
        st.dataframe(tbl.style.format("{:.1%}"), width="stretch")
    st.caption("公式データには1マーク通過時の順位が無いため、決まり手で1マークの展開を表しています。"
               "「1マークで先頭」は、逃げ・差し・まくり・まくり差しのいずれかでその艇が勝つ確率です。"
               "抜き・恵まれは1マークの後で先頭が入れ替わる展開です。")
    actual = r.iloc[0].get("kimarite") if not r.empty else None
    if isinstance(actual, str):
        st.info(f"結果の決まり手: {actual}")


@st.cache_data(ttl=60, show_spinner="公式サイトからオッズを取得中…")
def cached_win_odds(race_date: str, venue: int, race_no: int) -> dict:
    """同じレースは60秒以内なら取り直さない (公式サイトへのアクセスを減らすため)。"""
    from boatrace.odds import fetch_win_odds
    return fetch_win_odds(race_date, venue, race_no)


with st.expander("単勝オッズと期待値（公式サイトから取得）"):
    st.caption("ボタンを押したときだけ、このレースの単勝オッズを公式サイトから1回取得します。"
               "期待値 = 1着確率 × オッズ（1を超えると、確率が正しければ長期的に得）。"
               "公式サイトの情報は私的使用に限られるため、取得したオッズを他の人に配ったり転載したりしないでください。")
    if st.button("単勝オッズを取得"):
        try:
            rec = cached_win_odds(race_date, int(venue), int(race_no))
        except Exception as e:
            st.error(f"オッズを取得できませんでした: {e}")
        else:
            win = {int(k): v for k, v in rec["win"].items()}
            if not win:
                st.warning("オッズがまだ発表されていないか、このレースのオッズがありません。")
            else:
                ev = g[["lane", "racer_name", "win_prob"]].copy()
                ev["odds"] = ev["lane"].map(win)
                inv = 1 / ev["odds"]
                ev["market"] = inv / inv.sum()
                ev["ev"] = ev["win_prob"] * ev["odds"]
                st.write("締切時オッズ（確定）" if rec["final"] else "締切前のオッズ（締切まで変わります）")
                st.dataframe(ev.rename(columns={"lane": "枠", "racer_name": "選手", "win_prob": "1着確率",
                                                "odds": "単勝オッズ", "market": "オッズから見た確率", "ev": "期待値"})
                             .style.format({"1着確率": "{:.1%}", "単勝オッズ": "{:.1f}", "オッズから見た確率": "{:.1%}",
                                            "期待値": "{:.2f}"}, na_rep="-"), hide_index=True)
                st.caption("バックテストでは、締切時オッズでもこの期待値で買って控除率（約25%）を上回れるかは確かめられていません。"
                           "1を超えた艇があっても、利益を保証するものではありません。")

@st.cache_data(ttl=60, show_spinner="公式サイトからオッズを取得中…")
def cached_combo_odds(kind: str, race_date: str, venue: int, race_no: int) -> dict:
    """同じレース・賭け式は60秒以内なら取り直さない。"""
    from boatrace.odds import fetch_combo_odds
    rec = fetch_combo_odds(kind, race_date, venue, race_no)
    return {"final": rec["final"], "odds": {"-".join(map(str, k)): v for k, v in rec["odds"].items()}}


with st.expander("2連単・3連単のオッズと期待値（公式サイトから取得）"):
    st.warning("**この期待値は参考にしないでください。** 3連単の締切時オッズ 1,307 レースで検証したところ、"
               "この画面の3連単の確率（1着確率から計算式 Harville で組み立てたもの）はオッズより当たらず"
               "（対数損失 +0.34）、期待値 1 以上の組を買った場合の回収率は 77%（95%信頼区間 59〜100%）でした。"
               "2着・3着を直接学習するモデルで改善を検証中です。")
    st.caption("ボタンを押した賭け式のオッズページを1回だけ取得します。期待値 = 確率 × オッズ。"
               "私的使用の範囲で、取得したオッズは他の人に配らないでください。")
    b1, b2, b3 = st.columns(3)
    kind = None
    if b1.button("2連単オッズを取得"):
        kind = "exacta"
    if b2.button("3連単オッズを取得"):
        kind = "trifecta"
    min_ev = b3.number_input("表示する期待値の下限", 0.0, 5.0, 0.0, 0.1)
    if kind:
        try:
            rec = cached_combo_odds(kind, race_date, int(venue), int(race_no))
        except Exception as e:
            st.error(f"オッズを取得できませんでした: {e}")
        else:
            if not rec["odds"]:
                st.warning("オッズがまだ発表されていないか、このレースのオッズがありません。")
            else:
                probs = (exacta_probs if kind == "exacta" else trifecta_probs)(g["lane"].tolist(), g["win_prob"].tolist())
                probs["odds"] = probs["combo"].map(rec["odds"])
                probs["ev"] = probs["prob"] * probs["odds"]
                view = probs[probs["ev"].fillna(0) >= min_ev].sort_values("ev", ascending=False)
                st.write(("締切時オッズ（確定）" if rec["final"] else "締切前のオッズ（締切まで変わります）")
                         + f" ／ {len(view)}通り（期待値の高い順）")
                st.dataframe(view.rename(columns={"combo": "組番", "prob": "確率", "odds": "オッズ", "ev": "期待値"})
                             .style.format({"確率": "{:.2%}", "オッズ": "{:.1f}", "期待値": "{:.2f}"}, na_rep="-"),
                             hide_index=True, height=420)

st.caption("確率は過去データからの統計的推定であり、的中や利益を保証するものではありません。")
