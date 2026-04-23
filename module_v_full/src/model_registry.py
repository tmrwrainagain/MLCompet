from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg

from src.config import DATABASE_URL


MODEL_VERSION_LOG_TABLE = "module_v_model_versions"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_model_version_table() -> None:
    query = f"""
        CREATE TABLE IF NOT EXISTS public.{MODEL_VERSION_LOG_TABLE} (
            version TEXT PRIMARY KEY,
            trained_at TIMESTAMPTZ,
            update_type TEXT NOT NULL DEFAULT 'initial',
            drift_pct DOUBLE PRECISION,
            n_materials INTEGER,
            model_dir TEXT,
            meta_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
        conn.commit()


def log_model_version(meta: dict[str, Any], model_dir: str) -> None:
    ensure_model_version_table()

    trained_at = meta.get("trained_at")
    if isinstance(trained_at, str):
        try:
            trained_at = datetime.fromisoformat(trained_at)
        except ValueError:
            trained_at = None

    payload = json.dumps(meta, ensure_ascii=False, default=_json_default)

    query = f"""
        INSERT INTO public.{MODEL_VERSION_LOG_TABLE} (
            version,
            trained_at,
            update_type,
            drift_pct,
            n_materials,
            model_dir,
            meta_json
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (version) DO UPDATE SET
            trained_at = EXCLUDED.trained_at,
            update_type = EXCLUDED.update_type,
            drift_pct = EXCLUDED.drift_pct,
            n_materials = EXCLUDED.n_materials,
            model_dir = EXCLUDED.model_dir,
            meta_json = EXCLUDED.meta_json,
            created_at = NOW()
    """

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    meta.get("version"),
                    trained_at,
                    meta.get("update_type", "initial"),
                    meta.get("drift_pct"),
                    meta.get("n_materials"),
                    model_dir,
                    payload,
                ),
            )
        conn.commit()


def fetch_recent_model_versions(limit: int = 10) -> list[dict[str, Any]]:
    ensure_model_version_table()
    query = f"""
        SELECT
            version,
            trained_at,
            update_type,
            drift_pct,
            n_materials,
            model_dir,
            meta_json,
            created_at
        FROM public.{MODEL_VERSION_LOG_TABLE}
        ORDER BY COALESCE(trained_at, created_at) DESC
        LIMIT %s
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
    return [dict(row) for row in rows]
