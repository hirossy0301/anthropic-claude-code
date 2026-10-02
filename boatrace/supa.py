"""Supabase (PostgREST) への読み書き。追加のライブラリを使わず requests で呼ぶ。

接続先は環境変数 (GitHub Actions) か引数で渡す:
  SUPABASE_URL         https://xxxx.supabase.co
  SUPABASE_SECRET_KEY  書き込み用 (secret / service_role)。GitHub の Secrets にだけ置く
アプリは読み取り用の publishable (anon) key を Streamlit の secrets から渡す。
"""
from __future__ import annotations

import os

import requests


def env_config() -> tuple[str, str]:
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SECRET_KEY", "")
    if not url or not key:
        raise SystemExit("SUPABASE_URL と SUPABASE_SECRET_KEY を設定してください")
    return url.rstrip("/"), key


def _headers(key: str, extra: dict | None = None) -> dict:
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):  # 旧形式 (JWT) の鍵は Authorization にも入れる
        h["Authorization"] = f"Bearer {key}"
    return {**h, **(extra or {})}


def upsert(url: str, key: str, table: str, rows: list[dict], chunk: int = 500) -> int:
    """主キーが同じ行は、渡した列だけ上書きする。"""
    for i in range(0, len(rows), chunk):
        r = requests.post(f"{url}/rest/v1/{table}", json=rows[i:i + chunk], timeout=60,
                          headers=_headers(key, {"Prefer": "resolution=merge-duplicates,return=minimal"}))
        r.raise_for_status()
    return len(rows)


def select(url: str, key: str, table: str, params: dict | None = None, page: int = 1000) -> list[dict]:
    """全件を page 件ずつ取得する (PostgREST は1回の上限が1,000件)。"""
    out: list[dict] = []
    while True:
        r = requests.get(f"{url}/rest/v1/{table}", params={**(params or {}), "limit": page, "offset": len(out)},
                         headers=_headers(key), timeout=60)
        r.raise_for_status()
        rows = r.json()
        out += rows
        if len(rows) < page:
            return out
