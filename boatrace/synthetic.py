"""動作確認用の合成データ生成 (実データではない)。

公式ファイルと同じ体裁 (全角数字・全角空白を含む) の b*.txt / k*.txt を書き出すので、
ダウンロードできない環境でもパーサー〜学習〜画面まで一通り動かせる。
着順は「枠の有利さ + 級別 + 勝率 + モーター + 乱数」の強さから決める。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

FW = str.maketrans("0123456789R", "０１２３４５６７８９Ｒ")
CLASSES = ["A1", "A2", "B1", "B2"]
LANE_ADV = np.array([2.2, 0.9, 0.7, 0.4, 0.1, -0.2])
NAMES = ["山田太郎", "佐藤一郎", "鈴木次郎", "高橋三郎", "田中四郎", "伊藤五郎", "渡辺六郎", "中村七子"]
BRANCHES = ["群馬", "埼玉", "東京", "静岡", "愛知", "大阪", "福岡", "山口"]


def _racers(rng: np.random.Generator, n: int = 400) -> list[dict]:
    racers = []
    for i in range(n):
        cls = rng.choice(CLASSES, p=[0.2, 0.2, 0.45, 0.15])
        skill = {"A1": 1.2, "A2": 0.6, "B1": 0.0, "B2": -0.6}[cls] + rng.normal(0, 0.3)
        racers.append({"id": 3000 + i, "name": NAMES[i % len(NAMES)], "cls": cls, "skill": skill,
                       "branch": BRANCHES[i % len(BRANCHES)], "age": int(rng.integers(20, 55)),
                       "weight": int(rng.integers(47, 56)),
                       "win_rate": float(np.clip(5.0 + 1.5 * skill + rng.normal(0, 0.3), 1, 9))})
    return racers


def generate(out_dir: Path, start: date, days: int, venues=(1, 2, 12, 24), seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    racers = _racers(rng)
    motors = {v: rng.normal(0, 0.4, 80) for v in venues}
    out_dir.mkdir(parents=True, exist_ok=True)
    for day in range(days):
        d = start + timedelta(days=day)
        b, k = ["STARTB"], ["STARTK"]
        for v in venues:
            b += [f"{v:02d}BBGN", "", "                     ***  番組表  ***", ""]
            k += [f"{v:02d}KBGN", "", "                     ***  競走成績  ***", ""]
            for r in range(1, 13):
                picks = rng.choice(len(racers), 6, replace=False)
                header = f"　{str(r).translate(FW)}Ｒ  予選　　　　　　　 Ｈ１８００ｍ  電話投票締切予定１０：３５"
                b += [header, "-" * 79]
                entries = []
                for lane, idx in enumerate(picks, start=1):
                    rc = racers[idx]
                    mno = int(rng.integers(1, 80))
                    m2 = float(np.clip(33 + 10 * motors[v][mno] + rng.normal(0, 3), 10, 70))
                    b2 = float(np.clip(33 + rng.normal(0, 5), 10, 70))
                    st = float(np.clip(0.17 - 0.03 * rc["skill"] + rng.normal(0, 0.04), 0.01, 0.4))
                    strength = LANE_ADV[lane - 1] + rc["skill"] + motors[v][mno] + 4 * (0.16 - st)
                    entries.append((lane, rc, mno, st, strength))
                    b.append(f"{lane} {rc['id']}{rc['name']}{rc['age']:02d}{rc['branch']}"
                             f"{rc['weight']:02d}{rc['cls']} {rc['win_rate']:4.2f} {rc['win_rate'] * 7:5.2f}"
                             f" {rc['win_rate']:4.2f} {rc['win_rate'] * 7:5.2f} {mno:3d} {m2:5.2f}"
                             f" {int(rng.integers(1, 80)):3d} {b2:5.2f}")
                b.append("")
                # Plackett-Luce に従う着順: 強さ + ガンベル乱数 の降順
                noise = rng.gumbel(size=6)
                order = np.argsort(-(np.array([e[4] for e in entries]) + noise))
                k += [f"   {r}R       予選                 H1800m  晴　  風  北西　 3m  波　  2cm",
                      "  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ", "-" * 79]
                for pos, i in enumerate(order, start=1):
                    lane, rc, mno, st, _ = entries[i]
                    name = "　".join(rc["name"])
                    k.append(f"  {pos:02d}  {lane} {rc['id']} {name} {mno:3d}  {int(rng.integers(1, 80)):3d}"
                             f"  6.{int(rng.integers(60, 90))}   {lane}    {st:4.2f}     1.{49 + pos}.{pos}")
                tri = "-".join(str(entries[i][0]) for i in order[:3])
                # 払戻 = 真の確率に対する公正オッズ × 還元率75% (控除率25%)
                p = np.exp([e[4] for e in entries])
                p /= p.sum()
                a, b_, c = order[:3]
                true_prob = p[a] * p[b_] / (1 - p[a]) * p[c] / (1 - p[a] - p[b_])
                payout = max(100, int(0.75 * 100 / true_prob / 10) * 10)
                k += ["", f"        単勝     {entries[order[0]][0]}          {int(rng.integers(110, 900))}",
                      f"        ３連単   {tri}      {payout}  人気     {int(rng.integers(1, 120))}", ""]
            b.append(f"{v:02d}BEND")
            k.append(f"{v:02d}KEND")
        b.append("FINALB")
        k.append("FINALK")
        (out_dir / f"b{d.strftime('%y%m%d')}.txt").write_text("\n".join(b), encoding="utf-8")
        (out_dir / f"k{d.strftime('%y%m%d')}.txt").write_text("\n".join(k), encoding="utf-8")
