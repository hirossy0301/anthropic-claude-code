"""1着確率モデルと 3連単確率の計算。"""
from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, RACE_KEYS


CALIB_HOLDOUT = 0.2  # 較正用に取り分ける、学習期間の最後の割合 (日付順)


def _race_normalize(raw: np.ndarray, df: pd.DataFrame, alpha) -> pd.Series:
    """各艇の値を alpha 乗してから、レース内で合計1になるように割る。

    alpha は1つの数 (全艇共通) か、枠1〜6ごとの6つの数。
    """
    a = np.asarray(alpha, dtype=float)
    if a.ndim:
        a = a[df["lane"].to_numpy() - 1]
    s = pd.Series(np.power(np.clip(raw, 1e-12, 1), a), index=df.index)
    return s / s.groupby([df[k] for k in RACE_KEYS]).transform("sum")


def fit_alpha(raw: np.ndarray, df: pd.DataFrame) -> list[float]:
    """勝者に付く確率の対数損失が最小になる、枠ごとの alpha (各 0.3〜4) を探す。

    枠ごとにするのは、1号艇だけ強弱の幅が予測より大きい、といった偏りを直すため。
    """
    from scipy.optimize import minimize

    win = df["win"].to_numpy() == 1

    def loss(alpha) -> float:
        p = _race_normalize(raw, df, alpha).to_numpy()
        return float(-np.log(np.clip(p[win], 1e-12, 1)).mean())

    res = minimize(loss, x0=np.ones(6), bounds=[(0.3, 4.0)] * 6, method="L-BFGS-B")
    return [float(x) for x in res.x]


@dataclass
class WinModel:
    """各艇の「1着になるか」を二値分類で学習し、レース内で合計1に正規化する。

    calibrate=True のときは、学習期間の最後の CALIB_HOLDOUT を使って較正の係数 alpha を枠ごとに決める。
    正規化の前に各艇の値を alpha 乗する (alpha > 1 でその枠の強弱の差を広げる)。
    係数は6つだけなので過学習しにくく、正規化後もレース内の合計は1のまま。
    """
    kind: str = "lgbm"  # "lgbm" or "logreg"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    estimator: object = None
    medians: dict | None = None  # logreg の欠損補完用 (学習データの中央値。pickle の版依存を避けるため dict)
    calibrate: bool = False
    calib_alpha: float | list[float] = 1.0

    def fit(self, df: pd.DataFrame) -> "WinModel":
        self.calib_alpha = 1.0
        if self.calibrate:
            # 学習期間を日付で前後に分け、前半で学習したモデルの出力を後半で較正する
            dates = np.sort(df["race_date"].unique())
            split = dates[int(len(dates) * (1 - CALIB_HOLDOUT))]
            early, late = df[df["race_date"] < split], df[df["race_date"] >= split]
            probe = WinModel(kind=self.kind, features=self.features)._fit_estimator(early)
            self.calib_alpha = fit_alpha(probe._raw(late), late)
        return self._fit_estimator(df)

    def _fit_estimator(self, df: pd.DataFrame) -> "WinModel":
        X, y = df[self.features], df["win"].astype(int)
        self.medians = X.median().fillna(0).to_dict()  # 全欠損の列 (気温未取得など) は 0
        if self.kind == "lgbm":
            import lightgbm as lgb
            self.estimator = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                min_child_samples=50, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, verbose=-1)
            self.estimator.fit(X, y)
        else:
            self.estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            self.estimator.fit(X.fillna(self.medians), y)
        return self

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features]
        if self.kind != "lgbm":
            X = X.fillna(self.medians)
        return self.estimator.predict_proba(X)[:, 1]

    def predict_win_prob(self, df: pd.DataFrame) -> pd.Series:
        # 較正の導入前に保存したモデルには calib_alpha が無いので 1 (補正なし) とみなす
        return _race_normalize(self._raw(df), df, getattr(self, "calib_alpha", 1.0))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "WinModel":
        return pickle.loads(path.read_bytes())


def exacta_probs(lanes: list[int], win_probs: list[float]) -> pd.DataFrame:
    """1着確率から 2連単 30通りの確率を計算する (Harville): P(a-b) = p_a * p_b / (1 - p_a)。"""
    p = dict(zip(lanes, win_probs))
    rows = [{"combo": f"{a}-{b}", "prob": p[a] * (p[b] / (1 - p[a]) if p[a] < 1 else 0)}
            for a, b in itertools.permutations(lanes, 2)]
    return pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)


def trifecta_probs(lanes: list[int], win_probs: list[float]) -> pd.DataFrame:
    """1着確率から 3連単 120通りの確率を Plackett-Luce (Harville) モデルで計算する。

    P(a-b-c) = p_a * p_b / (1 - p_a) * p_c / (1 - p_a - p_b)
    """
    p = dict(zip(lanes, win_probs))
    rows = []
    for a, b, c in itertools.permutations(lanes, 3):
        denom1, denom2 = 1 - p[a], 1 - p[a] - p[b]
        prob = p[a] * (p[b] / denom1 if denom1 > 0 else 0) * (p[c] / denom2 if denom2 > 0 else 0)
        rows.append({"combo": f"{a}-{b}-{c}", "prob": prob})
    return pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)


def multiclass_log_loss(df: pd.DataFrame, prob_col: str) -> float:
    """勝者に付けた確率の -log の平均 (レース単位)。"""
    winners = df[df["win"] == 1][prob_col].clip(1e-9, 1)
    return float(-np.log(winners).mean())
