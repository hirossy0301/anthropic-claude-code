"""特徴量の作成。

履歴系の特徴量は「その日より前の日付」のデータだけで計算する (当日以降の結果を
使うとリーク = 未来の情報の混入 になり、バックテストが実力以上に良く見えるため)。
"""
from __future__ import annotations

import pandas as pd

CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
RACE_KEYS = ["race_date", "venue", "race_no"]

FEATURES = [
    "lane", "class_rank", "nat_win_rate", "nat_2_rate", "loc_win_rate", "loc_2_rate",
    "motor_2_rate", "boat_2_rate", "weight", "age",
    "nat_win_rate_diff", "nat_win_rate_rank", "motor_2_rate_rank", "class_rank_diff",
    "racer_avg_st", "racer_lane_win", "racer_win", "venue_lane_win",
]


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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """db.load_frame() の結果に特徴量列を追加して返す。"""
    df = df.sort_values(RACE_KEYS + ["lane"]).reset_index(drop=True).copy()
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
    return df


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """結果が確定し、6艇そろって勝者が1艇だけのレースに絞る。"""
    done = df[df["win"].notna()]
    stats = done.groupby(RACE_KEYS)["win"].agg(["count", "sum"])
    ok = stats[(stats["count"] == 6) & (stats["sum"] == 1)].index
    return done.set_index(RACE_KEYS).loc[ok].reset_index()
