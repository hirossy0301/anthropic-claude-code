"""単勝の期待値 (EV) で買うかを決めた場合のバックテスト。

EV = モデルの1着確率 × 単勝オッズ。EV が閾値以上の艇だけを各100円買い、回収率を見る。
オッズは締切時のもの (実際に買う時点のオッズとは違うので、結果は実際より良く出やすい)。

あわせて、モデルの確率と「オッズから逆算した市場の確率」のどちらが当たっているかを比べる。
市場の確率 = (1/オッズ) をレース内で合計1にしたもの (控除率の分を均等に除いたもの)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import _md
from .features import FEATURE_SETS, RACE_KEYS, training_rows
from .model import WinModel

THRESHOLDS = (0.8, 1.0, 1.1, 1.2, 1.5, 2.0)
MODELS = {"朝": "morning", "直前": "prerace"}


def attach_odds(test: pd.DataFrame, odds: pd.DataFrame, races: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """検証データにオッズと単勝払戻を付ける。6艇すべてにオッズがあるレースだけ残す。"""
    m = test.merge(odds, on=RACE_KEYS + ["lane"], how="inner")
    ok = m.groupby(RACE_KEYS)["odds"].agg(lambda s: s.notna().sum() == 6 and len(s) == 6)
    keep = ok[ok].index
    m = m.set_index(RACE_KEYS).loc[keep].reset_index()
    m = m.merge(races[RACE_KEYS + ["win_payout"]], on=RACE_KEYS, how="left")
    m["market"] = 1 / m["odds"]
    m["overround"] = m.groupby(RACE_KEYS)["market"].transform("sum")
    m["market"] /= m["overround"]
    # 勝った艇の払戻は成績ファイルの単勝払戻 (確定値) を使う。オッズ×100 と一致するかも確かめる
    winners = m[m["win"] == 1]
    agree = float((np.round(winners["odds"] * 100) == winners["win_payout"]).mean()) if len(winners) else float("nan")
    info = {"odds_races": odds.groupby(RACE_KEYS).ngroups, "races": len(keep), "payout_agree": agree,
            "takeout": float(1 - (1 / m.groupby(RACE_KEYS)["overround"].first()).mean()) if len(m) else float("nan")}
    return m, info


def _log_loss(m: pd.DataFrame, col: str) -> np.ndarray:
    """レースごとの、勝った艇に付けた確率の -log。"""
    return -np.log(m.loc[m["win"] == 1, col].clip(1e-9, 1).to_numpy())


def fit_blend(m: pd.DataFrame, model_col: str) -> tuple[float, float]:
    """p ∝ モデル^a × 市場^b の a, b を対数損失が最小になるように決める。

    a が 0 より明らかに大きければ、モデルに市場 (オッズ) が織り込んでいない情報がある。
    """
    from scipy.optimize import minimize

    lp, lq = np.log(m[model_col].clip(1e-9)).to_numpy(), np.log(m["market"].clip(1e-9)).to_numpy()
    race = m.groupby(RACE_KEYS).ngroup().to_numpy()
    win = m["win"].to_numpy() == 1

    def loss(ab) -> float:
        z = ab[0] * lp + ab[1] * lq
        z = z - pd.Series(z).groupby(race).transform("max").to_numpy()
        e = np.exp(z)
        p = e / pd.Series(e).groupby(race).transform("sum").to_numpy()
        return float(-np.log(np.clip(p[win], 1e-12, 1)).mean())

    res = minimize(loss, x0=[0.0, 1.0], method="L-BFGS-B", bounds=[(-2, 4), (-2, 4)])
    return float(res.x[0]), float(res.x[1])


def apply_blend(m: pd.DataFrame, model_col: str, a: float, b: float) -> pd.Series:
    z = a * np.log(m[model_col].clip(1e-9)) + b * np.log(m["market"].clip(1e-9))
    e = np.exp(z - z.groupby([m[k] for k in RACE_KEYS]).transform("max"))
    return e / e.groupby([m[k] for k in RACE_KEYS]).transform("sum")


def bets(m: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """mask の艇を各100円買ったときの、レースごとの購入額と払戻。"""
    pay = np.where((m["win"] == 1) & mask, m["win_payout"].fillna(m["odds"] * 100), 0)
    df = m[RACE_KEYS].assign(bet=mask.astype(int) * 100, payout=pay, hit=((m["win"] == 1) & mask).astype(int))
    return df.groupby(RACE_KEYS)[["bet", "payout", "hit"]].sum().reset_index()


def roi_ci(per_race: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """回収率と、レースを単位にしたブートストラップの 95% 信頼区間。"""
    bet, pay = per_race["bet"].to_numpy(float), per_race["payout"].to_numpy(float)
    if bet.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(bet), size=(n_boot, len(bet)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = pay[idx].sum(1) / bet[idx].sum(1)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(pay.sum() / bet.sum()), float(lo), float(hi)


def strategy_table(m: pd.DataFrame, prob_cols: dict[str, str], thresholds=THRESHOLDS) -> pd.DataFrame:
    rows = []

    def add(name: str, mask: pd.Series) -> None:
        r = bets(m, mask)
        roi, lo, hi = roi_ci(r)
        n = int(mask.sum())
        rows.append({"買い方": name, "点数": n, "的中率": r["hit"].sum() / n if n else float("nan"),
                     "平均オッズ": float(m.loc[mask, "odds"].mean()) if n else float("nan"),
                     "回収率": roi, "95%CI": f"[{lo:.1%}, {hi:.1%}]" if n else "-"})

    add("全艇を買う", pd.Series(True, index=m.index))
    fav = m.groupby(RACE_KEYS)["odds"].transform("min") == m["odds"]
    add("1番人気を買う", fav)
    for label, col in prob_cols.items():
        top = m.groupby(RACE_KEYS)[col].transform("max") == m[col]
        add(f"{label}: 確率1位を買う", top)
        for t in thresholds:
            add(f"{label}: EV≥{t:g}", m[col] * m["odds"] >= t)
    return pd.DataFrame(rows).set_index("買い方")


def report(feat: pd.DataFrame, races: pd.DataFrame, odds: pd.DataFrame, cutoff: str) -> str:
    rows = training_rows(feat)
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    m, info = attach_odds(test, odds, races)
    if m.empty:
        return f"# 単勝の期待値バックテスト\n\ncutoff {cutoff} 以降で、オッズを取得済みのレースがありません。"
    for label, mode in MODELS.items():
        model = WinModel(features=list(FEATURE_SETS[mode]), calibrate=config.CALIBRATE).fit(train)
        m[f"p_{label}"] = model.predict_win_prob(m).to_numpy()

    out = [
        f"# 単勝の期待値バックテスト（cutoff {cutoff}）",
        f"- 学習: {train['race_date'].min()}〜{train['race_date'].max()}（{train.groupby(RACE_KEYS).ngroups:,} レース）",
        f"- 検証: 締切時オッズを取得済みの {info['odds_races']:,} レースのうち、6艇とも結果とオッズがそろう "
        f"{info['races']:,} レース（{m['race_date'].min()}〜{m['race_date'].max()}）",
        f"- 確認: 勝った艇の オッズ×100 と成績ファイルの単勝払戻の一致率 {info['payout_agree']:.1%}"
        f" ／ オッズから逆算した控除率 {info['takeout']:.1%}",
        "",
        "## 確率の当たり具合（モデル と 市場=オッズから逆算）",
        "対数損失は小さいほど良い。差はレースごとの差の平均と95%信頼区間（マイナスならモデルの方が良い）。",
        "",
    ]
    mk = _log_loss(m, "market")
    acc_mk = float((m.groupby(RACE_KEYS)["market"].transform("max") == m["market"])[m["win"] == 1].mean())
    rows_ll = [{"確率": "市場（オッズ）", "対数損失": f"{mk.mean():.4f}", "1着的中率": f"{acc_mk:.1%}", "モデル − 市場": "-"}]
    for label in MODELS:
        col = f"p_{label}"
        ll = _log_loss(m, col)
        d = ll - mk
        se = d.std(ddof=1) / np.sqrt(len(d))
        acc = float((m.groupby(RACE_KEYS)[col].transform("max") == m[col])[m["win"] == 1].mean())
        rows_ll.append({"確率": f"モデル {label}", "対数損失": f"{ll.mean():.4f}", "1着的中率": f"{acc:.1%}",
                        "モデル − 市場": f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]"})
    out += [_md(pd.DataFrame(rows_ll).set_index("確率")), ""]

    # モデルと市場の組み合わせ: 日付の前半で a, b を決め、後半で評価する (同じデータで決めて評価しない)
    dates = np.sort(m["race_date"].unique())
    split = dates[len(dates) // 2]
    first, second = m[m["race_date"] < split], m[m["race_date"] >= split].copy()
    blend_rows = []
    prob_cols = {f"モデル {k}": f"p_{k}" for k in MODELS}
    if len(first) and len(second):
        for label in MODELS:
            a, b = fit_blend(first, f"p_{label}")
            second[f"blend_{label}"] = apply_blend(second, f"p_{label}", a, b).to_numpy()
            d = _log_loss(second, f"blend_{label}") - _log_loss(second, "market")
            se = d.std(ddof=1) / np.sqrt(len(d))
            blend_rows.append({"組み合わせ": f"モデル {label} ^{a:.2f} × 市場 ^{b:.2f}",
                               "後半の対数損失 − 市場": f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]"})
        out += ["## モデルに市場が織り込んでいない情報があるか",
                f"前半（〜{split} より前）で p ∝ モデル^a × 市場^b の a, b を決め、後半（{split}〜）で市場だけの確率と比べる。"
                "a が0に近い、または差の区間が0をまたぐなら、モデルはオッズ以上の情報をほぼ持っていない。", "",
                _md(pd.DataFrame(blend_rows).set_index("組み合わせ")), ""]

    out += ["## 買い方ごとの回収率（各100円、締切時オッズ）",
            "EV = 確率 × オッズ。95%CI はレースを単位にしたブートストラップ。区間が100%をまたぐなら、儲かるとも損するとも言えない。", "",
            _md(strategy_table(m, prob_cols), {"点数": "{:,.0f}", "的中率": "{:.1%}", "平均オッズ": "{:.1f}", "回収率": "{:.1%}"}), ""]
    if blend_rows:
        out += [f"### 後半（{split}〜）だけ: 組み合わせた確率で買う", "",
                _md(strategy_table(second, {f"組み合わせ {k}": f"blend_{k}" for k in MODELS}),
                    {"点数": "{:,.0f}", "的中率": "{:.1%}", "平均オッズ": "{:.1f}", "回収率": "{:.1%}"}), ""]
    out += ["## 注意",
            "- 「直前」のモデルは、成績に記録された実際の進入コースと風・波を使う。進入はスタート時に決まり、"
            "締切時オッズには織り込まれていないため、直前の結果は実力より良く見える。締切前に分かる情報だけで比べるなら「朝」を見る。",
            "- 締切時オッズは、実際に買う時点（締切前）のオッズと違う。締切直前に人気が動くため、実際の回収率はこれより悪くなりやすい。",
            "- 閾値をこの表から選ぶと、選んだ閾値の成績は実力より良く見える（同じデータで選んで評価しているため）。",
            "- 控除率（約25%）を上回る優位がなければ、長く続けるほど回収率は控除後の水準に近づく。"]
    return "\n".join(out)


# ---- 3連単 ----

TRIFECTA_THRESHOLDS = (1.0, 1.2, 1.5, 2.0, 3.0)


def trifecta_frame(combos: pd.DataFrame, tri_odds: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """1組1行: モデルの3連単確率 (combos: RACE_KEYS, combo, prob)、締切時オッズ、市場の確率、的中、払戻。

    120通りすべてにオッズがあり、3連単が確定したレースだけ。
    """
    ok = tri_odds.groupby(RACE_KEYS)["odds"].agg(lambda s: s.notna().sum() == 120)
    tri_odds = tri_odds.set_index(RACE_KEYS).loc[ok[ok].index].reset_index()
    m = combos.merge(tri_odds, on=RACE_KEYS + ["combo"], how="inner")
    m = m.merge(races[RACE_KEYS + ["trifecta", "trifecta_payout"]], on=RACE_KEYS, how="inner")
    m = m[m.groupby(RACE_KEYS)["combo"].transform("size") == 120]
    m["win"] = (m["combo"] == m["trifecta"]).astype(int)
    m = m[m.groupby(RACE_KEYS)["win"].transform("sum") == 1]
    m["win_payout"] = m["trifecta_payout"]
    m["market"] = 1 / m["odds"]
    m["market"] /= m.groupby(RACE_KEYS)["market"].transform("sum")
    return m.rename(columns={"prob": "p_model"}).reset_index(drop=True)


def trifecta_report(feat: pd.DataFrame, races: pd.DataFrame, tri_odds: pd.DataFrame, cutoff: str) -> str:
    """3連単の期待値バックテスト。確率は朝のモデル (較正あり) の1着確率から Harville で組み立てる。"""
    rows = training_rows(feat)
    train, test = rows[rows["race_date"] < cutoff], rows[rows["race_date"] >= cutoff].copy()
    test = test[test.set_index(RACE_KEYS).index.isin(tri_odds.set_index(RACE_KEYS).index.unique())]
    if test.empty:
        return f"# 3連単の期待値バックテスト\n\ncutoff {cutoff} 以降で、3連単オッズを取得済みのレースがありません。"
    from .place import PlaceModel, compare, harville_combos

    model = WinModel(features=list(FEATURE_SETS["morning"]), calibrate=config.CALIBRATE).fit(train)
    place = PlaceModel().fit(train)
    test["p_win"] = model.predict_win_prob(test).to_numpy()
    h_tri, _ = harville_combos(test, "p_win")
    p_tri, _ = place.combo_probs(test, "p_win")
    m = trifecta_frame(h_tri, tri_odds, races)
    if m.empty:
        return "# 3連単の期待値バックテスト\n\n120通りのオッズと結果がそろうレースがありません。"
    m = m.merge(p_tri.rename(columns={"prob": "p_place"}), on=RACE_KEYS + ["combo"], how="left")
    n_races = m.groupby(RACE_KEYS).ngroups
    winners = m[m["win"] == 1]
    agree = float((np.round(winners["odds"] * 100) == winners["trifecta_payout"]).mean())

    ll_market = _log_loss(m, "market")

    def vs_market(col: str) -> str:
        d = _log_loss(m, col) - ll_market
        se = d.std(ddof=1) / np.sqrt(len(d))
        return f"{d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]"
    out = [
        f"# 3連単の期待値バックテスト（cutoff {cutoff}）",
        f"- 学習: {train['race_date'].min()}〜{train['race_date'].max()} ／ 検証: 締切時の3連単オッズが120通りそろい、"
        f"結果が確定した {n_races:,} レース（{m['race_date'].min()}〜{m['race_date'].max()}）",
        "- 確率: 朝のモデル（較正あり）の1着確率から、Harville（計算式）と着順モデル（2着・3着を学習）で120通りを計算",
        f"- 確認: 的中した組の オッズ×100 と3連単払戻の一致率 {agree:.1%}",
        "",
        "## 確率の当たり具合（実際の3連単に付けた確率）",
        "",
        _md(pd.DataFrame([
            {"確率": "市場（オッズから逆算）", "対数損失": f"{ll_market.mean():.4f}", "モデル − 市場": "-"},
            {"確率": "Harville", "対数損失": f"{_log_loss(m, 'p_model').mean():.4f}", "モデル − 市場": vs_market("p_model")},
            {"確率": "着順モデル", "対数損失": f"{_log_loss(m, 'p_place').mean():.4f}", "モデル − 市場": vs_market("p_place")},
        ]).set_index("確率")),
        "",
        "## Harville と着順モデル（同じレースでの比較）",
        "実際の3連単・2連単に付けた確率の対数損失。差はマイナスなら着順モデルが良い。",
        "",
        _md(compare(test, races, "p_win", place)),
        "",
    ]
    dates = np.sort(m["race_date"].unique())
    split = dates[len(dates) // 2]
    first, second = m[m["race_date"] < split], m[m["race_date"] >= split].copy()
    blend_cols = {}
    if len(first) and len(second):
        a, b = fit_blend(first, "p_place")
        second["blend"] = apply_blend(second, "p_place", a, b).to_numpy()
        d2 = _log_loss(second, "blend") - _log_loss(second, "market")
        se2 = d2.std(ddof=1) / np.sqrt(len(d2))
        out += ["## モデルに市場が織り込んでいない情報があるか",
                f"前半で p ∝ 着順モデル^a × 市場^b を決め（a = {a:.2f}, b = {b:.2f}）、後半（{split}〜）で市場と比べる: "
                f"対数損失の差 {d2.mean():+.4f} [{d2.mean() - 1.96 * se2:+.4f}, {d2.mean() + 1.96 * se2:+.4f}]", ""]
        blend_cols = {"組み合わせ": "blend"}

    fmt = {"点数": "{:,.0f}", "的中率": "{:.2%}", "平均オッズ": "{:.1f}", "回収率": "{:.1%}"}
    table = strategy_table(m, {"Harville": "p_model", "着順モデル": "p_place"}, TRIFECTA_THRESHOLDS)
    table.index = [i.replace("全艇を買う", "全組を買う") for i in table.index]
    table.index.name = "買い方"
    out += ["## 買い方ごとの回収率（各100円、締切時オッズ）",
            "EV = 確率 × オッズ。95%CI はレースを単位にしたブートストラップ。", "", _md(table, fmt), ""]
    if blend_cols:
        t2 = strategy_table(second, blend_cols, TRIFECTA_THRESHOLDS)
        t2.index = [i.replace("全艇を買う", "全組を買う") for i in t2.index]
        t2.index.name = "買い方"
        out += [f"### 後半（{split}〜）だけ: 組み合わせた確率で買う", "", _md(t2, fmt), ""]
    out += ["## 注意",
            "- 2着・3着の確率は Harville の計算式によるもので、偏りがあることが知られている。",
            "- 3連単は120通りあるため、確率のわずかなずれでも EV の高い組が多く出る。閾値を表から選ぶと実力より良く見える。",
            "- 締切時オッズでの結果。実際に買う締切前のオッズとは違う。"]
    return "\n".join(out)
