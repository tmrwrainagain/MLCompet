"""
PostgreSQL database manager.
All upsert operations prevent duplicate rows on re-run.
Extended with:
  - lesson_type column on materials
  - methodology_requirements table
  - methodology_compliance table  (per-material, per-requirement scores)
  - cluster columns on materials (parallel_cluster, sequential_cluster, complexity_cluster)
"""

from contextlib import contextmanager
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
import psycopg2.extras
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATABASE_URL, METHODOLOGY_REQUIREMENTS


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """Create all tables if they don't exist and seed static data."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS materials (
                id                    SERIAL PRIMARY KEY,
                url                   TEXT NOT NULL,
                url_hash              TEXT UNIQUE NOT NULL,
                subject               TEXT,
                topic                 TEXT,
                text_content          TEXT,
                annotation            TEXT,
                file_type             TEXT,
                language              TEXT,
                lesson_type           TEXT DEFAULT 'other',
                moderation_status     TEXT DEFAULT 'pending',
                moderation_notes      TEXT,
                compliance_score      REAL,
                is_compliant          BOOLEAN,
                is_generated          BOOLEAN DEFAULT FALSE,
                generation_source_id  INTEGER REFERENCES materials(id),
                has_previous          BOOLEAN DEFAULT FALSE,
                previous_material_id  INTEGER REFERENCES materials(id),
                has_next              BOOLEAN DEFAULT FALSE,
                next_material_id      INTEGER REFERENCES materials(id),
                parallel_cluster      INTEGER,
                sequential_cluster    INTEGER,
                complexity_cluster    INTEGER,
                difficulty_label      TEXT,
                created_at            TIMESTAMPTZ DEFAULT NOW(),
                updated_at            TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS media_items (
                id                   SERIAL PRIMARY KEY,
                material_id          INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
                media_type           TEXT NOT NULL,
                source_url           TEXT,
                local_path           TEXT,
                description          TEXT,
                position_in_material INTEGER,
                created_at           TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS material_features (
                id                   SERIAL PRIMARY KEY,
                material_id          INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
                feature_name         TEXT NOT NULL,
                feature_value        REAL,
                feature_value_text   TEXT,
                UNIQUE (material_id, feature_name)
            );

            CREATE TABLE IF NOT EXISTS feature_importance (
                id               SERIAL PRIMARY KEY,
                feature_name     TEXT NOT NULL,
                importance_score REAL,
                method           TEXT,
                computed_at      TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS methodology_requirements (
                id          SERIAL PRIMARY KEY,
                category    TEXT NOT NULL,
                requirement TEXT NOT NULL,
                description TEXT,
                UNIQUE (category, requirement)
            );

            CREATE TABLE IF NOT EXISTS methodology_compliance (
                id             SERIAL PRIMARY KEY,
                material_id    INTEGER NOT NULL REFERENCES materials(id) ON DELETE CASCADE,
                requirement_id INTEGER NOT NULL REFERENCES methodology_requirements(id) ON DELETE CASCADE,
                is_met         BOOLEAN,
                score          REAL,
                notes          TEXT,
                UNIQUE (material_id, requirement_id)
            );

            CREATE INDEX IF NOT EXISTS idx_materials_url_hash  ON materials(url_hash);
            CREATE INDEX IF NOT EXISTS idx_media_material      ON media_items(material_id);
            CREATE INDEX IF NOT EXISTS idx_features_material   ON material_features(material_id);
            CREATE INDEX IF NOT EXISTS idx_compliance_material ON methodology_compliance(material_id);
            """
        )

    _seed_methodology_requirements()


def _seed_methodology_requirements():
    with get_conn() as conn:
        cur = conn.cursor()
        for req in METHODOLOGY_REQUIREMENTS:
            cur.execute(
                """
                INSERT INTO methodology_requirements (category, requirement, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (category, requirement) DO UPDATE
                    SET description = EXCLUDED.description
                """,
                (req["category"], req["requirement"], req["description"]),
            )


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def upsert_material(data: Dict[str, Any]) -> int:
    url = data.get("url", "")
    if not url:
        raise ValueError("url is required")

    h = _url_hash(url)
    content_hash = data.get("file_content_hash")

    with get_conn() as conn:
        cur = conn.cursor()

        # Primary lookup: by url_hash
        cur.execute("SELECT id FROM materials WHERE url_hash = %s", (h,))
        row = cur.fetchone()

        # Secondary lookup: by file_content_hash (catches same file loaded from different paths)
        if not row and content_hash:
            cur.execute(
                "SELECT id FROM materials WHERE file_content_hash = %s",
                (content_hash,),
            )
            row = cur.fetchone()

        if row:
            material_id = row[0]
            update = {
                k: v
                for k, v in data.items()
                if k not in ("url", "url_hash", "id", "created_at") and v is not None
            }
            update["updated_at"] = datetime.now()
            if update:
                set_sql = ", ".join(f"{k} = %s" for k in update)
                cur.execute(
                    f"UPDATE materials SET {set_sql} WHERE id = %s",
                    list(update.values()) + [material_id],
                )
        else:
            insert = {"url": url, "url_hash": h}
            insert.update({k: v for k, v in data.items() if k not in ("url", "url_hash")})
            cols = ", ".join(insert.keys())
            placeholders = ", ".join(["%s"] * len(insert))
            cur.execute(
                f"INSERT INTO materials ({cols}) VALUES ({placeholders}) RETURNING id",
                list(insert.values()),
            )
            material_id = cur.fetchone()[0]

    return material_id


def get_all_materials() -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM materials ORDER BY id")
        return [dict(r) for r in cur.fetchall()]


def get_material_by_id(mid: int) -> Optional[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM materials WHERE id = %s", (mid,))
        row = cur.fetchone()
        return dict(row) if row else None


def upsert_media_item(data: Dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM media_items WHERE material_id = %s AND source_url = %s",
            (data.get("material_id"), data.get("source_url")),
        )
        row = cur.fetchone()

        if row:
            item_id = row[0]
            update = {k: v for k, v in data.items() if k != "id"}
            set_sql = ", ".join(f"{k} = %s" for k in update)
            cur.execute(
                f"UPDATE media_items SET {set_sql} WHERE id = %s",
                list(update.values()) + [item_id],
            )
        else:
            cols = ", ".join(data.keys())
            placeholders = ", ".join(["%s"] * len(data))
            cur.execute(
                f"INSERT INTO media_items ({cols}) VALUES ({placeholders}) RETURNING id",
                list(data.values()),
            )
            item_id = cur.fetchone()[0]

    return item_id


def get_media_items(material_id: int) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM media_items WHERE material_id = %s ORDER BY id", (material_id,))
        return [dict(r) for r in cur.fetchall()]


def upsert_feature(material_id: int, name: str, value: float = None, value_text: str = None):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO material_features (material_id, feature_name, feature_value, feature_value_text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (material_id, feature_name) DO UPDATE
                SET feature_value      = EXCLUDED.feature_value,
                    feature_value_text = EXCLUDED.feature_value_text
            """,
            (material_id, name, value, value_text),
        )


def get_material_features(material_id: int) -> Dict:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            "SELECT feature_name, feature_value, feature_value_text "
            "FROM material_features WHERE material_id = %s",
            (material_id,),
        )
        result = {}
        for r in cur.fetchall():
            result[r["feature_name"]] = (
                r["feature_value"] if r["feature_value"] is not None else r["feature_value_text"]
            )
        return result


def get_materials_with_selected_features(feature_names: Optional[Iterable[str]] = None) -> List[Dict]:
    if feature_names is None:
        feature_names = [
            "word_count",
            "avg_sentence_length",
            "media_count",
            "has_images",
            "has_videos",
            "has_questions",
            "compliance_score",
        ]

    feature_names = list(feature_names)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM materials ORDER BY id")
        materials = [dict(r) for r in cur.fetchall()]

    rows: List[Dict] = []
    for material in materials:
        feats = get_material_features(material["id"])
        row = dict(material)
        for name in feature_names:
            target_name = "compliance_score_feat" if name == "compliance_score" else name
            value = feats.get(name)
            row[target_name] = value if value is not None else 0
        rows.append(row)
    return rows


def save_feature_importance(rows: List[Dict]):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM feature_importance")
        for r in rows:
            cur.execute(
                "INSERT INTO feature_importance (feature_name, importance_score, method) VALUES (%s, %s, %s)",
                (r["feature_name"], r["importance_score"], r["method"]),
            )


def get_feature_importance(limit: Optional[int] = None) -> List[Dict]:
    sql = "SELECT feature_name, importance_score, method FROM feature_importance ORDER BY importance_score DESC"
    params: List[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def get_all_requirements() -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT * FROM methodology_requirements ORDER BY category, id")
        return [dict(r) for r in cur.fetchall()]


def upsert_compliance(material_id: int, requirement_id: int, is_met: bool, score: float, notes: str = ""):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO methodology_compliance (material_id, requirement_id, is_met, score, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (material_id, requirement_id) DO UPDATE
                SET is_met = EXCLUDED.is_met,
                    score  = EXCLUDED.score,
                    notes  = EXCLUDED.notes
            """,
            (material_id, requirement_id, is_met, score, notes),
        )


def get_compliance_for_material(material_id: int) -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT mc.*, mr.category, mr.requirement, mr.description
            FROM methodology_compliance mc
            JOIN methodology_requirements mr ON mr.id = mc.requirement_id
            WHERE mc.material_id = %s
            ORDER BY mr.category, mr.requirement
            """,
            (material_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_compliance_summary() -> List[Dict]:
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT
                m.subject,
                m.is_generated,
                mr.category,
                mr.requirement,
                AVG(mc.score)  AS avg_score,
                COUNT(mc.id)   AS sample_count,
                SUM(CASE WHEN mc.is_met THEN 1 ELSE 0 END)::REAL / COUNT(mc.id) AS met_ratio
            FROM methodology_compliance mc
            JOIN materials m ON m.id = mc.material_id
            JOIN methodology_requirements mr ON mr.id = mc.requirement_id
            GROUP BY m.subject, m.is_generated, mr.category, mr.requirement
            ORDER BY m.subject, mr.category, mr.requirement
            """
        )
        return [dict(r) for r in cur.fetchall()]


def save_cluster_labels(labels: List[Dict]):
    with get_conn() as conn:
        cur = conn.cursor()
        for row in labels:
            mid = row["material_id"]
            fields = {k: v for k, v in row.items() if k != "material_id" and v is not None}
            if not fields:
                continue
            set_sql = ", ".join(f"{k} = %s" for k in fields)
            cur.execute(
                f"UPDATE materials SET {set_sql}, updated_at = NOW() WHERE id = %s",
                list(fields.values()) + [mid],
            )
