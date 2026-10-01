"""時系列分割のバックテスト。

学習期間 (cutoff より前) で学習し、検証期間 (cutoff 以降) で評価する。
ランダム分割にすると未来のレースで学習してしまうため使わない。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_SETS, FEATURES_V1, RACE_KEYS, training_rows
from .model import WinModel, trifecta_probs


def race_records(test: pd.DataFrame, races: pd.DataFrame, prob_col: str, top_n: int) -> pd.DataFrame:
    """1レース1行: 1着予想の当否・勝者に付けた確率・3連単上位 top_n 点 (各100円) の当否と払戻。"""
    races = races.set_index(RACE_KEYS)
    rows = []
    for key, g in test.groupby(RACE_KEYS):
        rec = {"race_date": key[0], "venue": key[1], "race_no": key[2],
               "top1_hit": int(g.loc[g[prob_col].idxmax(), "win"] == 1),
               "winner_prob": float(g.loc[g["win"] == 1, prob_col].iloc[0]),
               "bet": 0, "tri_hit": 0, "payout": 0}
        if key in races.index and not pd.isna(races.loc[key, "trifecta"]):
            picks = set(trifecta_probs(g["lane"].tolist(), g[prob_col].tolist())["combo"].head(top_n))
            rec["bet"] = top_n * 100
            if races.loc[key, "trifecta"] in picks:
                rec["tri_hit"], rec["payout"] = 1, int(races.loc[key, "trifecta_payout"])
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(rec: pd.DataFrame, top_n: int) -> dict:
    bet_races = rec[rec["bet"] > 0]
    return {
        "races": len(rec),
        "win_accuracy": rec["top1_hit"].mean(),
        "log_loss": float(-np.log(rec["winner_prob"].clip(1e-9, 1)).mean()),
        f"trifecta_top{top_n}_hit": bet_races["tri_hit"].mean() if len(bet_races) else float("nan"),
        "roi": bet_races["payout"].sum() / bet_races["bet"].sum() if len(bet_races) else float("nan"),
    }


def evaluate(test: pd.DataFrame, races: pd.DataFrame, prob_col: str, top_n: int) -> dict:
    """1着的中率・対数損失、3連単上位 top_n 点買い (各100円) の的中率と回収率。"""
    return summarize(race_records(test, races, prob_col, top_n), top_n)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """同じレースでの b − a の差と 95% 信頼区間 (レース単位の対応のある比較)。

    対数損失は小さいほど良いので、b が良ければ log_loss の差は負になる。
    """
    m = a.merge(b, on=RACE_KEYS, suffixes=("_a", "_b"))
    out = {}
    for name, da, db in [("win_accuracy", "top1_hit_a", "top1_hit_b"), ("log_loss", "winner_prob_a", "winner_prob_b")]:
        if name == "log_loss":
            d = -np.log(m[db].clip(1e-9, 1)) + np.log(m[da].clip(1e-9, 1))
        else:
            d = m[db] - m[da]
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[name] = (float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se))
    return out


def calibration(test: pd.DataFrame, prob_col: str,
                bins=(0, .1, .2, .3, .4, .5, .6, .7, .8, 1.0)) -> pd.DataFrame:
    """予測確率の帯ごとに、予測の平均と実際の1着率を比べる。"""
    b = pd.cut(test[prob_col], list(bins), include_lowest=True)
    return (test.groupby(b, observed=True)
            .agg(艇数=("win", "size"), 予測=(prob_col, "mean"), 実際=("win", "mean")))


def quarterly(rec: pd.DataFrame) -> pd.DataFrame:
    q = pd.PeriodIndex(pd.to_datetime(rec["race_date"]), freq="Q").astype(str)
    g = rec.assign(四半期=q).groupby("四半期")
    return pd.DataFrame({
        "レース数": g.size(),
        "1着的中率": g["top1_hit"].mean(),
        "3連単的中率": g["tri_hit"].mean(),
        "回収率": g["payout"].sum() / g["bet"].sum(),
    })


# 名前: (手法, 特徴量, 較正するか)
SPECS = {
    "logreg 朝": ("logreg", FEATURE_SETS["morning"], False),
    "lgbm 初版(v1)": ("lgbm", FEATURES_V1, False),
    "lgbm 朝": ("lgbm", FEATURE_SETS["morning"], False),
    "lgbm 直前": ("lgbm", FEATURE_SETS["prerace"], False),
    "lgbm 朝 較正": ("lgbm", FEATURE_SETS["morning"], True),
    "lgbm 直前 較正": ("lgbm", FEATURE_SETS["prerace"], True),
}


def run_backtest(feat: pd.DataFrame, races: pd.DataFrame, cutoff: str, top_n: int = 5,
                 detail: bool = False):
    """各手法の指標の表を返す。detail=True なら (表, 検証データ, レース単位の記録, モデル) を返す。"""
    rows = training_rows(feat)
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError(f"cutoff {cutoff} で学習/検証データが空になります")

    # ベースライン: 学習期間の「場×枠の1着率」をそのまま確率にする
    lane_rate = train.groupby(["venue", "lane"])["win"].mean()
    test["p_base"] = [lane_rate.get((v, l), 1 / 6) for v, l in zip(test["venue"], test["lane"])]
    test["p_base"] /= test.groupby(RACE_KEYS)["p_base"].transform("sum")

    records = {"baseline(場×枠)": race_records(test, races, "p_base", top_n)}
    models = {}
    # 要因を追加する前 (v1) と後 (朝 / 直前) を同じ期間で比べる
    for name, (kind, features, calibrate) in SPECS.items():
        models[name] = WinModel(kind=kind, features=list(features), calibrate=calibrate).fit(train)
        test[f"p_{name}"] = models[name].predict_win_prob(test)
        records[name] = race_records(test, races, f"p_{name}", top_n)
    table = pd.DataFrame({k: summarize(v, top_n) for k, v in records.items()}).T
    if detail:
        return table, test, records, models
    return table


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    df = df.copy()
    for col, f in (fmt or {}).items():
        if col in df:
            df[col] = df[col].map(lambda x, f=f: f.format(x) if pd.notna(x) else "-")
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep, *body])


def report(feat: pd.DataFrame, races: pd.DataFrame, cutoff: str, top_n: int = 5) -> str:
    """バックテスト結果を Markdown で返す (GitHub Actions の Summary や README 用)。"""
    table, test, records, models = run_backtest(feat, races, cutoff, top_n, detail=True)
    train_days = feat.loc[feat["race_date"] < cutoff, "race_date"]
    test_days = test["race_date"]
    hit = f"trifecta_top{top_n}_hit"
    t = table.rename(columns={"races": "レース数", "win_accuracy": "1着的中率", "log_loss": "対数損失",
                              hit: f"3連単上位{top_n}点", "roi": "回収率"})
    t.index.name = "手法"
    out = [
        f"# バックテスト（cutoff {cutoff}）",
        f"- 学習: {train_days.min()}〜{train_days.max()} / 検証: {test_days.min()}〜{test_days.max()}",
        "",
        "## 全体",
        _md(t, {"レース数": "{:,.0f}", "1着的中率": "{:.2%}", "対数損失": "{:.4f}",
                f"3連単上位{top_n}点": "{:.2%}", "回収率": "{:.1%}"}),
        "",
        "## 追加した要因の効果（同じレースでの差と95%信頼区間）",
        "区間が0をまたがなければ、差は誤差では説明しにくい。対数損失はマイナスが改善。",
        "",
    ]
    rows = []
    for a, b in [("lgbm 初版(v1)", "lgbm 朝"), ("lgbm 朝", "lgbm 直前"), ("logreg 朝", "lgbm 朝"),
                 ("lgbm 朝", "lgbm 朝 較正"), ("lgbm 直前", "lgbm 直前 較正")]:
        d = paired_difference(records[a], records[b])
        rows.append({"比較": f"{b} − {a}",
                     "1着的中率の差": "{:+.2%} [{:+.2%}, {:+.2%}]".format(*d["win_accuracy"]),
                     "対数損失の差": "{:+.4f} [{:+.4f}, {:+.4f}]".format(*d["log_loss"])})
    out += [_md(pd.DataFrame(rows).set_index("比較")), ""]

    before = calibration(test, "p_lgbm 朝")
    after = calibration(test, "p_lgbm 朝 較正")
    cal = pd.DataFrame({"補正前 艇数": before["艇数"], "補正前 予測": before["予測"], "補正前 実際": before["実際"],
                        "補正後 艇数": after["艇数"], "補正後 予測": after["予測"], "補正後 実際": after["実際"]})
    alphas = "; ".join(f"{n}: " + ", ".join(f"{x:.2f}" for x in np.atleast_1d(models[n].calib_alpha))
                       for n in ("lgbm 朝 較正", "lgbm 直前 較正"))
    out += ["## 較正（lgbm 朝、補正前と後）",
            f"予測確率の帯ごとの、予測の平均と実際の1着率。補正の係数 alpha（枠1〜6）= {alphas}（1より大きいとその枠の強弱の差を広げる）。", "",
            _md(cal.rename_axis("予測確率"),
                {c: ("{:,.0f}" if "艇数" in c else "{:.1%}") for c in cal.columns}), ""]
    out += ["## 四半期ごと（lgbm 朝 較正）", "",
            _md(quarterly(records["lgbm 朝 較正"]),
                {"レース数": "{:,.0f}", "1着的中率": "{:.1%}", "3連単的中率": "{:.1%}", "回収率": "{:.1%}"}), ""]

    for name in ("lgbm 朝", "lgbm 直前"):
        est = models[name].estimator
        imp = (pd.Series(est.booster_.feature_importance("gain"), index=models[name].features)
               .sort_values(ascending=False))
        imp = (imp / imp.sum()).head(15).to_frame("寄与")
        imp.index.name = "特徴量"
        out += [f"## 特徴量の寄与 上位15（{name}、gain の割合）", "", _md(imp, {"寄与": "{:.1%}"}), ""]
    return "\n".join(out)
