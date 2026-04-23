"""PostgreSQL helpers for Module V — reads labelled materials from Module A/B."""

from __future__ import annotations

import pandas as pd
import psycopg

from src.config import DATABASE_URL


FEATURE_CTE = """
    WITH feature_pivot AS (
        SELECT
            mf.material_id,
            MAX(CASE WHEN mf.feature_name = 'subject' THEN mf.feature_value_text END) AS subject_feat,
            MAX(CASE WHEN mf.feature_name = 'topic' THEN mf.feature_value_text END) AS topic_feat,
            MAX(CASE WHEN mf.feature_name = 'word_count' THEN mf.feature_value END) AS word_count,
            MAX(CASE WHEN mf.feature_name = 'avg_sentence_length' THEN mf.feature_value END) AS avg_sentence_length,
            MAX(CASE WHEN mf.feature_name = 'media_count' THEN mf.feature_value END) AS media_count,
            MAX(CASE WHEN mf.feature_name = 'has_images' THEN mf.feature_value END) AS has_images,
            MAX(CASE WHEN mf.feature_name = 'has_videos' THEN mf.feature_value END) AS has_videos,
            MAX(CASE WHEN mf.feature_name = 'has_questions' THEN mf.feature_value END) AS has_questions,
            MAX(CASE WHEN mf.feature_name = 'compliance_score' THEN mf.feature_value END) AS compliance_score_feat,
            MAX(CASE WHEN mf.feature_name = 'is_generated' THEN mf.feature_value END) AS is_generated
        FROM material_features mf
        GROUP BY mf.material_id
    )
"""


def get_connection():
    return psycopg.connect(DATABASE_URL)


def load_labelled_materials() -> pd.DataFrame:
    """
    Load all materials that have cluster labels assigned by Module B.
    Returns a DataFrame ready for ML feature engineering.
    """
    with get_connection() as conn:
        schema_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'materials'
        """
        material_columns = {
            row["column_name"]
            for _, row in pd.read_sql(schema_query, conn).iterrows()
        }

        labels_in_materials = {
            "parallel_cluster",
            "sequential_cluster",
            "complexity_cluster",
            "difficulty_label",
        }.issubset(material_columns)

        direct_feature_columns = {
            "word_count",
            "avg_sentence_length",
            "media_count",
            "has_images",
            "has_videos",
            "has_questions",
            "compliance_score",
            "is_generated",
        }.issubset(material_columns)

        if labels_in_materials and direct_feature_columns:
            query = """
                SELECT
                    m.id,
                    COALESCE(m.topic, m.annotation, m.subject, m.url) AS title,
                    m.subject,
                    m.topic,
                    m.lesson_type,
                    m.word_count,
                    m.avg_sentence_length,
                    m.media_count,
                    m.has_images,
                    m.has_videos,
                    m.has_questions,
                    m.compliance_score        AS compliance_score_feat,
                    m.is_generated,
                    m.text_content,
                    m.parallel_cluster,
                    m.sequential_cluster,
                    m.complexity_cluster,
                    m.difficulty_label
                FROM materials m
                WHERE m.word_count IS NOT NULL
                ORDER BY m.id
            """
            return pd.read_sql(query, conn)

        query = FEATURE_CTE + """
            SELECT
                m.id,
                COALESCE(fp.topic_feat, m.topic, m.annotation, m.subject, m.url) AS title,
                COALESCE(fp.subject_feat, m.subject) AS subject,
                COALESCE(fp.topic_feat, m.topic) AS topic,
                m.lesson_type,
                fp.word_count,
                fp.avg_sentence_length,
                fp.media_count,
                fp.has_images,
                fp.has_videos,
                fp.has_questions,
                fp.compliance_score_feat,
                fp.is_generated,
                m.text_content,
                {parallel_col} AS parallel_cluster,
                {sequential_col} AS sequential_cluster,
                {complexity_col} AS complexity_cluster,
                {difficulty_col} AS difficulty_label
            FROM materials m
            JOIN feature_pivot fp ON fp.material_id = m.id
            {labels_join}
            WHERE fp.word_count IS NOT NULL
            ORDER BY m.id
        """.format(
            parallel_col="m.parallel_cluster" if labels_in_materials else "cl.parallel_cluster",
            sequential_col="m.sequential_cluster" if labels_in_materials else "cl.sequential_cluster",
            complexity_col="m.complexity_cluster" if labels_in_materials else "cl.complexity_cluster",
            difficulty_col="m.difficulty_label" if labels_in_materials else "cl.difficulty_label",
            labels_join="" if labels_in_materials else "JOIN cluster_labels cl ON cl.material_id = m.id",
        )
        return pd.read_sql(query, conn)


def load_all_materials() -> pd.DataFrame:
    """Load all materials (including unlabelled) for inference."""
    with get_connection() as conn:
        schema_query = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'materials'
        """
        material_columns = {
            row["column_name"]
            for _, row in pd.read_sql(schema_query, conn).iterrows()
        }

        direct_feature_columns = {
            "word_count",
            "avg_sentence_length",
            "media_count",
            "has_images",
            "has_videos",
            "has_questions",
            "compliance_score",
            "is_generated",
        }.issubset(material_columns)

        if direct_feature_columns:
            query = """
                SELECT
                    m.id,
                    COALESCE(m.topic, m.annotation, m.subject, m.url) AS title,
                    m.subject,
                    m.topic,
                    m.lesson_type,
                    m.word_count,
                    m.avg_sentence_length,
                    m.media_count,
                    m.has_images,
                    m.has_videos,
                    m.has_questions,
                    m.compliance_score AS compliance_score_feat,
                    m.is_generated,
                    m.text_content
                FROM materials m
                WHERE m.word_count IS NOT NULL
                ORDER BY m.id
            """
            return pd.read_sql(query, conn)

        query = FEATURE_CTE + """
            SELECT
                m.id,
                COALESCE(fp.topic_feat, m.topic, m.annotation, m.subject, m.url) AS title,
                COALESCE(fp.subject_feat, m.subject) AS subject,
                COALESCE(fp.topic_feat, m.topic) AS topic,
                m.lesson_type,
                fp.word_count,
                fp.avg_sentence_length,
                fp.media_count,
                fp.has_images,
                fp.has_videos,
                fp.has_questions,
                fp.compliance_score_feat,
                fp.is_generated,
                m.text_content
            FROM materials m
            JOIN feature_pivot fp ON fp.material_id = m.id
            WHERE fp.word_count IS NOT NULL
            ORDER BY m.id
        """
        return pd.read_sql(query, conn)
