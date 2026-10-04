"""2着・3着の専用モデル (着順モデル)。

Harville (1着確率だけから2着・3着を組み立てる) は、2着・3着の確率がずれやすい。
3連単の締切時オッズ 1,307 レースのバックテストでは、Harville の3連単の確率は市場 (オッズ) より
大きく劣った (対数損失 +0.342 [+0.289, +0.395])。そこで2着・3着を直接学習する。

  P(a-b-c) = P(1着 a) × P(2着 b | 1着 a) × P(3着 c | 1着 a, 2着 b)

1着確率は既存のモデル。2着は「1着の艇を除いた5艇」、3着は「1・2着を除いた4艇」の中で、
各艇がその着になるかを二値分類で学習し、残りの艇の中で合計1になるように割る。
1着・2着の艇の枠と強さ (勝率・級別) を特徴量に加えるので、「強い艇が勝ったときの2着」などを学べる。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES_MORNING, RACE_KEYS

BASE = list(FEATURES_MORNING)
_W = ["lane", "nat_win_rate", "class_rank", "racer_avg_st"]
CTX2 = ["w_lane", "w_nat_win_rate", "w_class_rank", "w_racer_avg_st", "rel_w"]
CTX3 = CTX2 + ["s_lane", "s_nat_win_rate", "s_class_rank", "s_racer_avg_st", "rel_s"]


def _ctx(df: pd.DataFrame, src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """src (1レース1艇) の枠・強さを prefix 付きで df に付ける。"""
    s = src[RACE_KEYS + _W].rename(columns={c: f"{prefix}{c}" for c in _W})
    return df.merge(s, on=RACE_KEYS, how="inner")


def _second_candidates(df: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    m = _ctx(df, winners, "w_")
    m = m[m["lane"] != m["w_lane"]].copy()
    m["rel_w"] = m["lane"] - m["w_lane"]
    return m


def _third_candidates(df: pd.DataFrame, winners: pd.DataFrame, seconds: pd.DataFrame) -> pd.DataFrame:
    m = _ctx(_ctx(df, winners, "w_"), seconds, "s_")
    m = m[(m["lane"] != m["w_lane"]) & (m["lane"] != m["s_lane"]) & (m["w_lane"] != m["s_lane"])].copy()
    m["rel_w"] = m["lane"] - m["w_lane"]
    m["rel_s"] = m["lane"] - m["s_lane"]
    return m


def _normalize(raw: np.ndarray, groups: list[pd.Series]) -> np.ndarray:
    s = pd.Series(np.clip(raw, 1e-9, 1))
    return (s / s.groupby([g.to_numpy() for g in groups]).transform("sum")).to_numpy()


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


@dataclass
class PlaceModel:
    second_features: list[str] = field(default_factory=lambda: BASE + CTX2)
    third_features: list[str] = field(default_factory=lambda: BASE + CTX3)
    second: object = None
    third: object = None

    def fit(self, rows: pd.DataFrame) -> "PlaceModel":
        """rows: 結果が確定したレースの各艇 (training_rows)。1〜3着が1艇ずつ決まっているレースだけ使う。"""
        fin = rows["finish"]
        ok = rows.assign(_1=fin == 1, _2=fin == 2, _3=fin == 3).groupby(RACE_KEYS)[["_1", "_2", "_3"]].sum()
        ok = ok[(ok == 1).all(axis=1)].index
        r = rows.set_index(RACE_KEYS).loc[ok].reset_index()
        w, s = r[r["finish"] == 1], r[r["finish"] == 2]
        c2 = _second_candidates(r, w)
        self.second = _lgbm().fit(c2[self.second_features], (c2["finish"] == 2).astype(int))
        c3 = _third_candidates(r, w, s)
        self.third = _lgbm().fit(c3[self.third_features], (c3["finish"] == 3).astype(int))
        return self

    def combo_probs(self, df: pd.DataFrame, win_col: str) -> pd.DataFrame:
        """各レース (6艇) の 3連単 120通りと 2連単 30通りの確率。

        戻り値: (3連単: RACE_KEYS, combo, prob), (2連単: RACE_KEYS, combo, prob)
        """
        df = df[df.groupby(RACE_KEYS)["lane"].transform("size") == 6]
        # 2着: 1着の候補 a ごとに、残り5艇の確率
        c2 = _second_candidates(df, df)
        c2["p2"] = _normalize(self.second.predict_proba(c2[self.second_features])[:, 1],
                              [c2[k] for k in RACE_KEYS] + [c2["w_lane"]])
        # 3着: (a, b) ごとに残り4艇の確率
        c3 = _third_candidates(df, df, df)
        c3["p3"] = _normalize(self.third.predict_proba(c3[self.third_features])[:, 1],
                              [c3[k] for k in RACE_KEYS] + [c3["w_lane"], c3["s_lane"]])
        p1 = df[RACE_KEYS + ["lane", win_col]].rename(columns={"lane": "a", win_col: "p1"})
        ex = c2[RACE_KEYS + ["w_lane", "lane", "p2"]].rename(columns={"w_lane": "a", "lane": "b"}).merge(
            p1, on=RACE_KEYS + ["a"])
        ex["prob"] = ex["p1"] * ex["p2"]
        tri = c3[RACE_KEYS + ["w_lane", "s_lane", "lane", "p3"]].rename(
            columns={"w_lane": "a", "s_lane": "b", "lane": "c"}).merge(
            ex[RACE_KEYS + ["a", "b", "prob"]].rename(columns={"prob": "p12"}), on=RACE_KEYS + ["a", "b"])
        tri["prob"] = tri["p12"] * tri["p3"]
        tri["combo"] = tri["a"].astype(str) + "-" + tri["b"].astype(str) + "-" + tri["c"].astype(str)
        ex["combo"] = ex["a"].astype(str) + "-" + ex["b"].astype(str)
        return tri[RACE_KEYS + ["combo", "prob"]], ex[RACE_KEYS + ["combo", "prob"]]

    def race_combos(self, g: pd.DataFrame, win_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """1レース分の (3連単, 2連単) を確率の高い順に (列: combo, prob)。アプリ用。"""
        tri, ex = self.combo_probs(g, win_col)
        return (tri[["combo", "prob"]].sort_values("prob", ascending=False).reset_index(drop=True),
                ex[["combo", "prob"]].sort_values("prob", ascending=False).reset_index(drop=True))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "PlaceModel":
        return pickle.loads(path.read_bytes())


def harville_combos(df: pd.DataFrame, win_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比較用: Harville の 3連単・2連単 (combo_probs と同じ形)。"""
    from .model import exacta_probs, trifecta_probs
    tris, exs = [], []
    for key, g in df.groupby(RACE_KEYS):
        if len(g) != 6:
            continue
        lanes, p = g["lane"].tolist(), g[win_col].tolist()
        tris.append(trifecta_probs(lanes, p).assign(race_date=key[0], venue=key[1], race_no=key[2]))
        exs.append(exacta_probs(lanes, p).assign(race_date=key[0], venue=key[1], race_no=key[2]))
    cols = RACE_KEYS + ["combo", "prob"]
    return pd.concat(tris)[cols], pd.concat(exs)[cols]


def compare(test: pd.DataFrame, races: pd.DataFrame, win_col: str, model: PlaceModel,
            n_races: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Harville と着順モデルの、実際の3連単・2連単に付けた確率の対数損失を同じレースで比べる。"""
    keys = test[RACE_KEYS].drop_duplicates()
    keys = keys.sample(min(n_races, len(keys)), random_state=seed)
    t = test.merge(keys, on=RACE_KEYS)
    res = races[RACE_KEYS + ["trifecta"]].dropna()
    res = res.assign(exacta=res["trifecta"].str.rsplit("-", n=1).str[0])
    out = []
    for (h_tri, h_ex), (m_tri, m_ex) in [(harville_combos(t, win_col), model.combo_probs(t, win_col))]:
        for name, h, m, col in [("3連単", h_tri, m_tri, "trifecta"), ("2連単", h_ex, m_ex, "exacta")]:
            hit = res[RACE_KEYS + [col]].rename(columns={col: "combo"})
            hh = h.merge(hit, on=RACE_KEYS + ["combo"]).set_index(RACE_KEYS)["prob"]
            mm = m.merge(hit, on=RACE_KEYS + ["combo"]).set_index(RACE_KEYS)["prob"]
            both = pd.concat([hh.rename("h"), mm.rename("m")], axis=1).dropna()
            d = -np.log(both["m"].clip(1e-9)) + np.log(both["h"].clip(1e-9))
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append({"賭け式": name, "レース数": len(both),
                        "Harville": float(-np.log(both["h"].clip(1e-9)).mean()),
                        "着順モデル": float(-np.log(both["m"].clip(1e-9)).mean()),
                        "差 [95%CI]": f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]"})
    return pd.DataFrame(out).set_index("賭け式")
