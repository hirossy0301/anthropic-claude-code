"""レース場ごとの固定情報 (水面の種類・水質・位置)。

水面の種類と水質は公式サイトの「場の特徴」に基づく想定だが、開発環境から公式サイトに
接続できなかったため未検証。緯度経度は気温取得用の概算 (気象データの格子より十分細かい)。
"""
from __future__ import annotations

# 水面の種類: lake=湖・池, sea=海, river=川・河口
# 水質: fresh=淡水, brackish=汽水, salt=海水
VENUE_INFO = {
    1: {"name": "桐生", "surface": "lake", "water": "fresh", "lat": 36.40, "lon": 139.28},
    2: {"name": "戸田", "surface": "lake", "water": "fresh", "lat": 35.81, "lon": 139.68},
    3: {"name": "江戸川", "surface": "river", "water": "brackish", "lat": 35.69, "lon": 139.86},
    4: {"name": "平和島", "surface": "sea", "water": "salt", "lat": 35.58, "lon": 139.74},
    5: {"name": "多摩川", "surface": "lake", "water": "fresh", "lat": 35.65, "lon": 139.49},
    6: {"name": "浜名湖", "surface": "lake", "water": "brackish", "lat": 34.69, "lon": 137.57},
    7: {"name": "蒲郡", "surface": "sea", "water": "brackish", "lat": 34.82, "lon": 137.22},
    8: {"name": "常滑", "surface": "sea", "water": "salt", "lat": 34.88, "lon": 136.83},
    9: {"name": "津", "surface": "sea", "water": "brackish", "lat": 34.69, "lon": 136.52},
    10: {"name": "三国", "surface": "lake", "water": "fresh", "lat": 36.22, "lon": 136.16},
    11: {"name": "びわこ", "surface": "lake", "water": "fresh", "lat": 35.02, "lon": 135.87},
    12: {"name": "住之江", "surface": "lake", "water": "fresh", "lat": 34.61, "lon": 135.48},
    13: {"name": "尼崎", "surface": "lake", "water": "fresh", "lat": 34.72, "lon": 135.43},
    14: {"name": "鳴門", "surface": "sea", "water": "salt", "lat": 34.19, "lon": 134.61},
    15: {"name": "丸亀", "surface": "sea", "water": "salt", "lat": 34.30, "lon": 133.79},
    16: {"name": "児島", "surface": "sea", "water": "salt", "lat": 34.46, "lon": 133.80},
    17: {"name": "宮島", "surface": "sea", "water": "salt", "lat": 34.30, "lon": 132.30},
    18: {"name": "徳山", "surface": "sea", "water": "salt", "lat": 34.03, "lon": 131.77},
    19: {"name": "下関", "surface": "sea", "water": "salt", "lat": 33.99, "lon": 130.99},
    20: {"name": "若松", "surface": "sea", "water": "salt", "lat": 33.90, "lon": 130.79},
    21: {"name": "芦屋", "surface": "lake", "water": "fresh", "lat": 33.89, "lon": 130.66},
    22: {"name": "福岡", "surface": "river", "water": "brackish", "lat": 33.59, "lon": 130.40},
    23: {"name": "唐津", "surface": "lake", "water": "fresh", "lat": 33.43, "lon": 129.98},
    24: {"name": "大村", "surface": "sea", "water": "salt", "lat": 32.91, "lon": 129.93},
}

SURFACE_LABEL = {"lake": "湖・池", "sea": "海", "river": "川・河口"}
WATER_LABEL = {"fresh": "淡水", "brackish": "汽水", "salt": "海水"}
SURFACE_CODE = {"lake": 0, "river": 1, "sea": 2}
WATER_CODE = {"fresh": 0, "brackish": 1, "salt": 2}  # 塩分が濃いほど大きい (浮力も大きい)

# 16方位 -> 角度 (度, 北=0 時計回り)
WIND_DEG = {d: i * 22.5 for i, d in enumerate(
    ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
     "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"])}
