"""1マークの展開 (決まり手) の予測。

公式データには1マーク通過時の順位が無いため、レースごとの「決まり手」で1マークの展開を表す。
逃げ・差し・まくり・まくり差しは1マークで勝負が決まった (勝った艇が1マークで先頭に立った) ことを、
抜き・恵まれは1マークの後で先頭が変わったことを表す。

確率は2段階に分ける:
  P(艇 i が決まり手 k で勝つ) = P(艇 i が1着) × P(決まり手 k | 艇 i が1着)
1着確率は既存のモデル、後ろの条件付き確率をこのモジュールのモデル (LightGBM の多クラス分類) で出す。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import KIMARITE_HISTORY, RACE_KEYS

TYPES = ["逃げ", "差し", "まくり", "まくり差し", "抜き", "恵まれ"]
# 1マークで勝負が決まった決まり手 (勝った艇が1マークで先頭)
DECIDED_AT_TURN1 = ["逃げ", "差し", "まくり", "まくり差し"]

_SELF = ["racer_avg_st", "nat_win_rate", "class_rank", "motor_2_rate"]
FEATURES = (
    ["course", "venue", "surface", "water", "wind_speed", "wind_sin", "wind_cos", "wave"]
    + _SELF + [c for c, *_ in KIMARITE_HISTORY]
    + [f"in_{c}" for c in _SELF + ["racer_nige_rate"]]
    + ["inner_racer_avg_st", "inner_nat_win_rate", "outer_racer_avg_st", "st_gap_inner", "st_gap_in"]
)


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """各艇に、1コースの艇・1つ内側の艇・1つ外側の艇の情報を付ける (進入コースで引く)。

    直前情報で進入コースを変えたあとにも呼び直せるよう、特徴量の計算とは分けている。
    """
    df = df.copy()
    base = df[RACE_KEYS + ["course"] + _SELF + ["racer_nige_rate"]].drop_duplicates(RACE_KEYS + ["course"])
    inn = base[base["course"] == 1].drop(columns="course").rename(
        columns={c: f"in_{c}" for c in _SELF + ["racer_nige_rate"]})
    df = df.drop(columns=[c for c in inn.columns if c not in RACE_KEYS and c in df], errors="ignore")
    df = df.merge(inn, on=RACE_KEYS, how="left")
    for name, shift in (("inner", -1), ("outer", 1)):
        nb = base[RACE_KEYS + ["course", "racer_avg_st", "nat_win_rate"]].copy()
        nb["course"] = nb["course"] - shift  # コース c+shift の艇を、コース c の行に付ける
        nb = nb.rename(columns={"racer_avg_st": f"{name}_racer_avg_st", "nat_win_rate": f"{name}_nat_win_rate"})
        df = df.drop(columns=[f"{name}_racer_avg_st", f"{name}_nat_win_rate"], errors="ignore")
        df = df.merge(nb, on=RACE_KEYS + ["course"], how="left")
    df["st_gap_inner"] = df["inner_racer_avg_st"] - df["racer_avg_st"]  # 正なら内側の艇よりスタートが速い
    df["st_gap_in"] = df["in_racer_avg_st"] - df["racer_avg_st"]
    df = df.drop(columns=["outer_nat_win_rate"])
    return df


@dataclass
class KimariteModel:
    """勝った艇の特徴量から、その勝ち方 (決まり手) の確率を出す。"""
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    estimator: object = None

    def fit(self, rows: pd.DataFrame) -> "KimariteModel":
        import lightgbm as lgb
        w = add_context(rows)
        w = w[(w["win"] == 1) & w["kimarite"].isin(TYPES)]
        y = w["kimarite"].map({k: i for i, k in enumerate(TYPES)}).astype(int)
        self.estimator = lgb.LGBMClassifier(
            # 「場×コースの割合」に勝つには強めの正則化が要った (葉31・最小50件では基準より悪化)
            objective="multiclass", num_class=len(TYPES), n_estimators=200, learning_rate=0.03,
            num_leaves=15, min_child_samples=200, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, verbose=-1)
        self.estimator.fit(w[self.features], y)
        return self

    def predict_conditional(self, df: pd.DataFrame) -> pd.DataFrame:
        """各艇について P(決まり手 | その艇が1着)。列は TYPES、行は df と同じ順。

        前後の艇の情報を使うので、df にはレースの全艇を含めること。
        """
        w = add_context(df)
        proba = np.zeros((len(w), len(TYPES)))
        proba[:, self.estimator.classes_] = self.estimator.predict_proba(w[self.features])
        return pd.DataFrame(proba, columns=TYPES, index=df.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "KimariteModel":
        return pickle.loads(path.read_bytes())


def race_outlook(g: pd.DataFrame, win_prob: pd.Series, model: KimariteModel) -> tuple[pd.DataFrame, pd.Series]:
    """1レース分の (艇×決まり手の確率の表, 決まり手ごとの確率)。g は進入コースを反映済みの1レース分。"""
    cond = model.predict_conditional(g)
    joint = cond.mul(win_prob.to_numpy(), axis=0)
    joint.index = g["lane"].to_numpy()
    return joint, joint.sum()


# ---- バックテスト ----

def _baseline(train_w: pd.DataFrame, test_w: pd.DataFrame, k: float = 30) -> np.ndarray:
    """場×勝った艇の進入コースごとの決まり手の割合 (コース全体の割合で平滑化)。"""
    onehot = pd.get_dummies(train_w["kimarite"]).reindex(columns=TYPES, fill_value=0).astype(float)
    t = pd.concat([train_w[["venue", "course"]].reset_index(drop=True), onehot.reset_index(drop=True)], axis=1)
    by_course = t.groupby("course")[TYPES].mean()
    vc = t.groupby(["venue", "course"])[TYPES].agg(["sum", "count"])
    out = []
    for v, c in zip(test_w["venue"], test_w["course"]):
        prior = by_course.loc[c].to_numpy() if c in by_course.index else np.full(len(TYPES), 1 / len(TYPES))
        if (v, c) in vc.index:
            row = vc.loc[(v, c)]
            sums = np.array([row[(k_, "sum")] for k_ in TYPES])
            n = row[(TYPES[0], "count")]
            out.append((sums + k * prior) / (n + k))
        else:
            out.append(prior)
    return np.vstack(out)


def report(feat: pd.DataFrame, cutoff: str) -> str:
    """決まり手モデルのバックテスト (Markdown)。勝った艇は実際の進入コースを使う (直前の予想と同じ条件)。"""
    from . import config
    from .backtest import _md
    from .features import FEATURE_SETS, training_rows
    from .model import WinModel

    rows = training_rows(feat)
    rows = rows[rows.groupby(RACE_KEYS)["kimarite"].transform(lambda s: s.isin(TYPES).all())]
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    if train.empty or test.empty:
        return f"# 決まり手のバックテスト\n\ncutoff {cutoff} で学習/検証データが空です。"
    model = KimariteModel().fit(train)
    # 前後の艇の情報を使うので、レースの6艇そろった状態で計算してから勝った艇の行を取り出す
    cond_all = model.predict_conditional(test)
    is_w = (test["win"] == 1).to_numpy()
    test_w = test[is_w]
    p_model = cond_all.to_numpy()[is_w]
    train_w = add_context(train)
    train_w = train_w[train_w["win"] == 1]
    p_base = _baseline(train_w, test_w)
    y = test_w["kimarite"].map({k: i for i, k in enumerate(TYPES)}).to_numpy()
    ll_m = -np.log(np.clip(p_model[np.arange(len(y)), y], 1e-9, 1))
    ll_b = -np.log(np.clip(p_base[np.arange(len(y)), y], 1e-9, 1))
    d = ll_m - ll_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    acc_m, acc_b = (p_model.argmax(1) == y).mean(), (p_base.argmax(1) == y).mean()

    out = [
        f"# 決まり手（1マークの展開）のバックテスト（cutoff {cutoff}）",
        f"- 学習: {train['race_date'].min()}〜{train['race_date'].max()}（{train_w.shape[0]:,} レース）"
        f" / 検証: {test['race_date'].min()}〜{test['race_date'].max()}（{len(test_w):,} レース）",
        "- 進入コースは実際の値を使う（直前の予想と同じ条件）。",
        "",
        "## 勝った艇が分かっているときの、決まり手の当たり具合",
        "基準 = 学習期間の「場×勝った艇の進入コース」ごとの決まり手の割合。差はマイナスならモデルが良い。",
        "",
        _md(pd.DataFrame([
            {"予測": "基準（場×コースの割合）", "対数損失": f"{ll_b.mean():.4f}", "的中率": f"{acc_b:.1%}", "モデル − 基準": "-"},
            {"予測": "モデル", "対数損失": f"{ll_m.mean():.4f}", "的中率": f"{acc_m:.1%}",
             "モデル − 基準": f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]"},
        ]).set_index("予測")),
        "",
    ]

    # 1着確率と組み合わせた「レース全体の決まり手」の確率 (勝った艇が分からない状態での予測)
    win_model = WinModel(features=list(FEATURE_SETS["prerace"]), calibrate=config.CALIBRATE).fit(train)
    test["p_win"] = win_model.predict_win_prob(test).to_numpy()
    joint = cond_all.mul(test["p_win"].to_numpy(), axis=0)
    race_p = joint.groupby([test[k] for k in RACE_KEYS]).sum()
    actual = test_w.set_index(RACE_KEYS)["kimarite"].reindex(race_p.index)
    share = pd.DataFrame({"予測の平均": race_p.mean(), "実際の割合": [(actual == k).mean() for k in TYPES]})
    share.index.name = "決まり手"
    rp = race_p.to_numpy()
    ya = actual.map({k: i for i, k in enumerate(TYPES)}).to_numpy()
    ll_race = -np.log(np.clip(rp[np.arange(len(ya)), ya], 1e-9, 1)).mean()
    out += ["## レース全体の決まり手（1着確率 × 条件付き確率）",
            f"勝った艇が分からない状態で「このレースは何で決まるか」を予測したときの平均。対数損失 {ll_race:.4f}"
            f"（全レースに実際の割合をそのまま付けた場合 "
            f"{-np.log(share['実際の割合'].clip(1e-9).to_numpy()[ya]).mean():.4f}）。", "",
            _md(share, {"予測の平均": "{:.1%}", "実際の割合": "{:.1%}"}), ""]

    # 「逃げ」の確率の較正
    b = pd.cut(race_p["逃げ"], [0, .2, .3, .4, .5, .6, .7, .8, 1.0], include_lowest=True)
    cal = pd.DataFrame({"レース数": race_p.groupby(b, observed=True).size(),
                        "予測": race_p["逃げ"].groupby(b, observed=True).mean(),
                        "実際": (actual == "逃げ").groupby(b, observed=True).mean()})
    cal.index = cal.index.astype(str)
    cal.index.name = "逃げの予測確率"
    out += ["## 「逃げ」の確率の較正", "予測確率の帯ごとに、実際に逃げで決まった割合と比べる。", "",
            _md(cal, {"レース数": "{:,.0f}", "予測": "{:.1%}", "実際": "{:.1%}"}), ""]

    imp = pd.Series(model.estimator.booster_.feature_importance("gain"), index=model.features)
    imp = (imp / imp.sum()).sort_values(ascending=False).head(12).to_frame("寄与")
    imp.index.name = "特徴量"
    out += ["## 特徴量の寄与 上位12（gain の割合）", "", _md(imp, {"寄与": "{:.1%}"}), ""]
    return "\n".join(out)
