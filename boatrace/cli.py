"""コマンドライン: python -m boatrace.cli <command>

  download --start 2023-10-01 --end 2026-09-28   公式データを取得 (要ネットワーク許可)
  ingest                                         data/raw を DB に取り込み
  demo-data --days 120                           合成データを data/raw に生成 (動作確認用)
  weather --start 2023-10-01 --end 2026-09-30    場ごとの気温を取得 (Open-Meteo)
  backtest --cutoff 2026-01-01                   時系列バックテスト
  train                                          全期間で学習 (朝用・直前用の2モデル)
  predict --date 2026-09-29 --venue 12 --race 1  朝の予想
  predict ... --mode prerace --wind-dir 北西 --wind-speed 3 --wave 2 --courses 1,2,3,4,5,6
                                                 直前の予想 (直前情報を入力)
"""
from __future__ import annotations

import argparse
from datetime import date

from . import config, db
from .backtest import run_backtest
from .features import FEATURE_SETS, RACE_KEYS, apply_prerace, build_features, training_rows
from .model import WinModel, trifecta_probs


def load_features():
    conn = db.connect()
    return build_features(db.load_frame(conn)), db.load_races(conn)


def predict_race(feat, model: WinModel, race_date: str, venue: int, race_no: int, prerace=None):
    """1レースを予想する。prerace は apply_prerace() に渡す直前情報の dict (直前モデル用)。"""
    g = feat[(feat["race_date"] == race_date) & (feat["venue"] == venue) & (feat["race_no"] == race_no)]
    if g.empty:
        raise ValueError(f"{race_date} {config.VENUES.get(venue, venue)} {race_no}R の出走表がありません")
    g = apply_prerace(g, **(prerace or {}))
    g["win_prob"] = model.predict_win_prob(g)
    return g, trifecta_probs(g["lane"].tolist(), g["win_prob"].tolist())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="boatrace")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download")
    s.add_argument("--start", type=date.fromisoformat, required=True)
    s.add_argument("--end", type=date.fromisoformat, required=True)
    sub.add_parser("ingest")
    s = sub.add_parser("weather")
    s.add_argument("--start", type=date.fromisoformat, required=True)
    s.add_argument("--end", type=date.fromisoformat, required=True)
    s = sub.add_parser("demo-data")
    s.add_argument("--days", type=int, default=120)
    s.add_argument("--start", type=date.fromisoformat, default=date(2026, 1, 1))
    s = sub.add_parser("backtest")
    s.add_argument("--cutoff", required=True)
    s.add_argument("--top-n", type=int, default=5)
    sub.add_parser("train")
    s = sub.add_parser("predict")
    s.add_argument("--date", required=True)
    s.add_argument("--venue", type=int, required=True)
    s.add_argument("--race", type=int, required=True)
    s.add_argument("--mode", choices=list(FEATURE_SETS), default="morning")
    s.add_argument("--wind-dir", help="16方位 (例: 北西)")
    s.add_argument("--wind-speed", type=float, help="風速 m")
    s.add_argument("--wave", type=float, help="波高 cm")
    s.add_argument("--courses", help="枠1〜6の進入コース (例: 1,2,4,3,5,6)")
    s.add_argument("--temperature", type=float, help="気温 (未取得のときの上書き)")
    a = p.parse_args(argv)

    if a.cmd == "download":
        from .download import download_range
        download_range(a.start, a.end)
    elif a.cmd == "ingest":
        db.ingest_dir(db.connect())
    elif a.cmd == "weather":
        from .weather import fetch_temperatures
        fetch_temperatures(db.connect(), a.start, a.end)
    elif a.cmd == "demo-data":
        from .synthetic import generate
        generate(config.RAW_DIR, a.start, a.days)
        print(f"合成データを {config.RAW_DIR} に生成しました (実データではありません)")
    elif a.cmd == "backtest":
        feat, races = load_features()
        print(run_backtest(feat, races, a.cutoff, a.top_n).round(4).to_string())
    elif a.cmd == "train":
        feat, _ = load_features()
        rows = training_rows(feat)
        for mode, features in FEATURE_SETS.items():
            WinModel(features=list(features)).fit(rows).save(config.model_path(mode))
            print(f"{mode}: {rows.groupby(RACE_KEYS).ngroups} レースで学習 -> {config.model_path(mode)}")
    elif a.cmd == "predict":
        feat, _ = load_features()
        prerace = None
        if a.mode == "prerace":
            prerace = {"wind_dir": a.wind_dir, "wind_speed": a.wind_speed, "wave": a.wave,
                       "temperature": a.temperature,
                       "courses": [int(c) for c in a.courses.split(",")] if a.courses else None}
        elif a.temperature is not None:
            prerace = {"temperature": a.temperature}
        model = WinModel.load(config.model_path(a.mode))
        g, tri = predict_race(feat, model, a.date, a.venue, a.race, prerace)
        print(g[["lane", "course", "racer_name", "racer_class", "nat_win_rate", "motor_2_rate", "win_prob"]]
              .round(3).to_string(index=False))
        print("\n3連単 上位10点")
        print(tri.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
