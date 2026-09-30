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


@dataclass
class WinModel:
    """各艇の「1着になるか」を二値分類で学習し、レース内で合計1に正規化する。"""
    kind: str = "lgbm"  # "lgbm" or "logreg"
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    estimator: object = None
    medians: pd.Series | None = None  # logreg の欠損補完用 (学習データの中央値)

    def fit(self, df: pd.DataFrame) -> "WinModel":
        X, y = df[self.features], df["win"].astype(int)
        self.medians = X.median().fillna(0)  # 全欠損の列 (気温未取得など) は 0
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

    def predict_win_prob(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.features]
        if self.kind != "lgbm":
            X = X.fillna(self.medians)
        raw = pd.Series(self.estimator.predict_proba(X)[:, 1], index=df.index)
        return raw / raw.groupby([df[k] for k in RACE_KEYS]).transform("sum")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "WinModel":
        return pickle.loads(path.read_bytes())


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
