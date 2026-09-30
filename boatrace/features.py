"""特徴量の作成。

履歴系の特徴量は「その日より前の日付」のデータだけで計算する (当日以降の結果を
使うとリーク = 未来の情報の混入 になり、バックテストが実力以上に良く見えるため)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .venues import SURFACE_CODE, VENUE_INFO, WATER_CODE, WIND_DEG

CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
RACE_KEYS = ["race_date", "venue", "race_no"]

# 初版の特徴量 (比較用に残す)
FEATURES_V1 = [
    "lane", "class_rank", "nat_win_rate", "nat_2_rate", "loc_win_rate", "loc_2_rate",
    "motor_2_rate", "boat_2_rate", "weight", "age",
    "nat_win_rate_diff", "nat_win_rate_rank", "motor_2_rate_rank", "class_rank_diff",
    "racer_avg_st", "racer_lane_win", "racer_win", "venue_lane_win",
]
# 朝の予想: 前日までに分かる情報 + 気温 (当日の予報で代用できる)
FEATURES_MORNING = FEATURES_V1 + [
    "venue_course_win", "surface", "water", "weight_diff", "weight_diff_fresh", "temperature",
]
# 直前の予想: 直前情報 (風向・風速・波高・進入コース) を加える
FEATURES_PRERACE = FEATURES_MORNING + [
    "venue", "course", "venue_course_win_actual", "wind_speed", "wind_sin", "wind_cos", "wave",
]
FEATURE_SETS = {"morning": FEATURES_MORNING, "prerace": FEATURES_PRERACE}
FEATURES = FEATURES_MORNING


def _prior_rate(df: pd.DataFrame, keys: list[str], value: str, prior: float, k: float) -> pd.Series:
    """keys ごとに、前日までの value の平均をベイズ平滑化 (件数 k 分の prior を加える) して返す。"""
    daily = (df.groupby(keys + ["race_date"])[value].agg(["sum", "count"]).reset_index()
             .sort_values("race_date"))
    g = daily.groupby(keys)
    daily["cum_sum"] = g["sum"].cumsum() - daily["sum"]
    daily["cum_cnt"] = g["count"].cumsum() - daily["count"]
    daily["rate"] = (daily["cum_sum"] + prior * k) / (daily["cum_cnt"] + k)
    merged = df[keys + ["race_date"]].merge(
        daily[keys + ["race_date", "rate"]], on=keys + ["race_date"], how="left")
    return merged["rate"].to_numpy()


def _prior_lookup(df: pd.DataFrame, src_key: str, lookup_key: str, value: str,
                  prior: float, k: float) -> np.ndarray:
    """(venue, src_key) ごとの前日までの平滑化率を、(venue, lookup_key) で引く。

    例: 実際の進入コース (course) で集計した1着率を、枠番 (lane) で引く。
    """
    daily = (df.dropna(subset=[src_key]).groupby(["venue", src_key, "race_date"])[value]
             .agg(["sum", "count"]).reset_index().sort_values("race_date"))
    g = daily.groupby(["venue", src_key])
    daily["rate"] = (g["sum"].cumsum() + prior * k) / (g["count"].cumsum() + k)  # その日を含む累計
    daily = daily.rename(columns={src_key: "_key"})
    daily["_key"] = daily["_key"].astype(float)
    left = df[["venue", "race_date"]].copy()
    left["_key"] = df[lookup_key].astype(float)
    left["_date"] = pd.to_datetime(left["race_date"])
    left["_pos"] = np.arange(len(left))
    daily["_date"] = pd.to_datetime(daily["race_date"])
    # 当日を含まない (allow_exact_matches=False) = 前日までの累計
    merged = pd.merge_asof(
        left.dropna(subset=["_key"]).sort_values("_date"), daily[["venue", "_key", "_date", "rate"]],
        on="_date", by=["venue", "_key"], allow_exact_matches=False)
    out = np.full(len(left), prior)
    out[merged["_pos"].to_numpy()] = merged["rate"].fillna(prior).to_numpy()
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """db.load_frame() の結果に特徴量列を追加して返す。"""
    df = df.sort_values(RACE_KEYS + ["lane"]).reset_index(drop=True).copy()
    for col in ("course", "wind_dir", "wind_speed", "wave", "temperature"):
        if col not in df:
            df[col] = np.nan
    for col in ("course", "wind_speed", "wave", "temperature"):  # 全件欠損だと object 型になるため
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["class_rank"] = df["racer_class"].map(CLASS_RANK).fillna(1)
    df["win"] = (df["finish"] == 1).astype(float).where(df["finish_code"].notna())

    race = df.groupby(RACE_KEYS)
    df["nat_win_rate_diff"] = df["nat_win_rate"] - race["nat_win_rate"].transform("mean")
    df["nat_win_rate_rank"] = race["nat_win_rate"].rank(ascending=False, method="min")
    df["motor_2_rate_rank"] = race["motor_2_rate"].rank(ascending=False, method="min")
    df["class_rank_diff"] = df["class_rank"] - race["class_rank"].transform("mean")

    df["racer_avg_st"] = _prior_rate(df, ["racer_id"], "start_timing", prior=0.16, k=5)
    df["racer_lane_win"] = _prior_rate(df, ["racer_id", "lane"], "win", prior=1 / 6, k=5)
    df["racer_win"] = _prior_rate(df, ["racer_id"], "win", prior=1 / 6, k=10)
    df["venue_lane_win"] = _prior_rate(df, ["venue", "lane"], "win", prior=1 / 6, k=50)

    # 場×進入コースの1着率。朝は枠なり進入 (コース=枠) と仮定し、直前は実際の進入コースで引く
    df["venue_course_win"] = _prior_lookup(df, "course", "lane", "win", prior=1 / 6, k=50)
    df["course"] = df["course"].fillna(df["lane"])
    df["venue_course_win_actual"] = _prior_lookup(df, "course", "course", "win", prior=1 / 6, k=50)

    # 水面: 淡水は浮力が小さく体重の影響が出やすいので、体重差との組み合わせも作る
    info = df["venue"].map(VENUE_INFO)
    df["surface"] = info.map(lambda v: SURFACE_CODE[v["surface"]] if isinstance(v, dict) else np.nan)
    df["water"] = info.map(lambda v: WATER_CODE[v["water"]] if isinstance(v, dict) else np.nan)
    df["weight_diff"] = df["weight"] - race["weight"].transform("mean")
    df["weight_diff_fresh"] = df["weight_diff"] * (df["water"] == WATER_CODE["fresh"])

    # 風: 向きは角度の sin/cos に分解 (北と北北西が近いことをモデルが扱えるように)
    rad = np.deg2rad(df["wind_dir"].map(WIND_DEG))
    calm = df["wind_speed"].fillna(0).eq(0)
    df["wind_sin"] = np.where(calm, 0.0, np.sin(rad))
    df["wind_cos"] = np.where(calm, 0.0, np.cos(rad))
    return df


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """結果が確定し、6艇そろって勝者が1艇だけのレースに絞る。"""
    done = df[df["win"].notna()]
    stats = done.groupby(RACE_KEYS)["win"].agg(["count", "sum"])
    ok = stats[(stats["count"] == 6) & (stats["sum"] == 1)].index
    return done.set_index(RACE_KEYS).loc[ok].reset_index()


def apply_prerace(g: pd.DataFrame, wind_dir: str | None = None, wind_speed: float | None = None,
                  wave: float | None = None, courses: list[int] | None = None,
                  temperature: float | None = None) -> pd.DataFrame:
    """1レース分の特徴量に、直前情報 (風・波・進入コース・気温) を上書きする。

    courses は枠1〜6の順に並べた進入コース (例: 前付けで [1, 2, 4, 3, 5, 6])。
    """
    g = g.sort_values("lane").copy()
    if wind_speed is not None:
        g["wind_speed"] = wind_speed
    if wind_dir is not None:
        g["wind_dir"] = wind_dir
    if wave is not None:
        g["wave"] = wave
    if temperature is not None:
        g["temperature"] = temperature
    if courses is not None:
        if sorted(courses) != list(range(1, len(g) + 1)):
            raise ValueError(f"進入コースは 1〜{len(g)} を1回ずつ指定してください: {courses}")
        g["course"] = courses
    # venue_course_win は「枠番 = コース」で引いた値なので、コース c の率は lane == c の行にある
    rate_by_course = dict(zip(g["lane"], g["venue_course_win"]))
    g["venue_course_win_actual"] = g["course"].map(rate_by_course)
    rad = np.deg2rad(g["wind_dir"].map(WIND_DEG).astype(float))
    calm = g["wind_speed"].fillna(0).eq(0)
    g["wind_sin"] = np.where(calm, 0.0, np.sin(rad))
    g["wind_cos"] = np.where(calm, 0.0, np.cos(rad))
    return g
