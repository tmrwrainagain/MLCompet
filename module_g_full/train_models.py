"""
Train and save sklearn models for Module G API.

Run ONCE after Module B has populated the database with cluster labels.
The API (api.py) only LOADS these models — it never retrains them.

Usage:
    python train_models.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from module_G.config import DATABASE_URL, MODELS_DIR


# ── Synthetic fallback data (used if DB is empty or unavailable) ──────────────

SYNTHETIC_TEXTS = [
    "Введение в программирование. Основные понятия алгоритма и программы. Переменные и типы данных.",
    "Циклы и условные операторы. Практические задания по программированию на Python.",
    "Объектно-ориентированное программирование. Классы, объекты, наследование, инкапсуляция.",
    "Базы данных. SQL-запросы. Нормализация таблиц. Транзакции и индексы.",
    "Математический анализ. Пределы функций. Производные. Формула Лейбница.",
    "Интегральное исчисление. Методы интегрирования. Несобственные интегралы.",
    "Линейная алгебра. Матрицы и определители. Системы линейных уравнений.",
    "Теория вероятностей. Случайные события и их вероятности. Теорема Байеса.",
    "Физика. Механика. Законы Ньютона. Работа и энергия. Закон сохранения.",
    "Химия. Строение атома. Периодическая система. Химические реакции.",
    "Английский язык. Грамматика. Времена глагола. Условные предложения.",
    "История России. Ключевые события XX века. Революция и советский период.",
]

SYNTHETIC_NUMERIC = [
    [300, 12, 0, 0, 0, 0, 0.6],
    [800, 15, 1, 0, 0, 1, 0.7],
    [1500, 20, 0, 1, 0, 1, 0.8],
    [1200, 18, 1, 0, 1, 1, 0.75],
    [600, 25, 0, 0, 0, 0, 0.5],
    [900, 22, 1, 0, 0, 1, 0.65],
    [700, 16, 0, 1, 0, 0, 0.55],
    [500, 14, 1, 0, 0, 1, 0.7],
    [1100, 19, 0, 1, 1, 1, 0.8],
    [400, 11, 0, 0, 0, 0, 0.45],
    [850, 17, 1, 0, 0, 1, 0.72],
    [650, 13, 0, 0, 0, 1, 0.6],
]


def _load_from_db():
    import psycopg2
    import psycopg2.extras
    conn = psycopg2.connect(DATABASE_URL)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT m.id, m.text_content, m.sequential_cluster, m.complexity_cluster,
               m.difficulty_label,
               COALESCE(f_wc.feature_value, 500)  AS word_count,
               COALESCE(f_as.feature_value, 15)   AS avg_sentence_length,
               COALESCE(f_mc.feature_value, 0)    AS media_count,
               COALESCE(f_hi.feature_value, 0)    AS has_images,
               COALESCE(f_hv.feature_value, 0)    AS has_videos,
               COALESCE(f_hq.feature_value, 0)    AS has_questions,
               COALESCE(f_cs.feature_value, 0.5)  AS compliance_score
        FROM materials m
        LEFT JOIN material_features f_wc ON f_wc.material_id = m.id AND f_wc.feature_name = 'word_count'
        LEFT JOIN material_features f_as ON f_as.material_id = m.id AND f_as.feature_name = 'avg_sentence_length'
        LEFT JOIN material_features f_mc ON f_mc.material_id = m.id AND f_mc.feature_name = 'media_count'
        LEFT JOIN material_features f_hi ON f_hi.material_id = m.id AND f_hi.feature_name = 'has_images'
        LEFT JOIN material_features f_hv ON f_hv.material_id = m.id AND f_hv.feature_name = 'has_videos'
        LEFT JOIN material_features f_hq ON f_hq.material_id = m.id AND f_hq.feature_name = 'has_questions'
        LEFT JOIN material_features f_cs ON f_cs.material_id = m.id AND f_cs.feature_name = 'compliance_score'
        WHERE m.text_content IS NOT NULL AND LENGTH(m.text_content) > 20
        ORDER BY m.id
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def _build_matrices(rows):
    texts   = [r["text_content"] or "" for r in rows]
    numeric = [
        [
            float(r.get("word_count", 500)),
            float(r.get("avg_sentence_length", 15)),
            float(r.get("media_count", 0)),
            float(r.get("has_images", 0)),
            float(r.get("has_videos", 0)),
            float(r.get("has_questions", 0)),
            float(r.get("compliance_score", 0.5)),
        ]
        for r in rows
    ]
    return texts, np.array(numeric, dtype=float)


def train_and_save():
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestCentroid
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # ── 1. Load data ──────────────────────────────────────────────────────────
    rows = []
    try:
        rows = _load_from_db()
        print(f"  Loaded {len(rows)} materials from database.")
    except Exception as exc:
        print(f"  DB unavailable ({exc}), using synthetic data.")

    if len(rows) < 4:
        print("  Not enough DB data — padding with synthetic samples.")
        texts   = SYNTHETIC_TEXTS
        numeric = np.array(SYNTHETIC_NUMERIC, dtype=float)
        seq_labels = [i % 4 for i in range(len(texts))]
    else:
        texts, numeric = _build_matrices(rows)
        seq_labels = [int(r.get("sequential_cluster") or 0) for r in rows]

    n = len(texts)

    # ── 2. TF-IDF vectorizer ──────────────────────────────────────────────────
    tfidf = TfidfVectorizer(max_features=500, sublinear_tf=True, min_df=1)
    X_tfidf = tfidf.fit_transform(texts)
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    print("  Saved tfidf_vectorizer.joblib")

    # ── 3. Numeric scaler (for parallel) ──────────────────────────────────────
    scaler_num = StandardScaler()
    X_scaled   = scaler_num.fit_transform(numeric)
    joblib.dump(scaler_num, MODELS_DIR / "scaler_numeric.joblib")
    print("  Saved scaler_numeric.joblib")

    # ── 4. KMeans — parallel clustering (TF-IDF features) ────────────────────
    n_par = min(4, n)
    km_parallel = KMeans(n_clusters=n_par, random_state=42, n_init=10)
    km_parallel.fit(X_tfidf)
    joblib.dump(km_parallel, MODELS_DIR / "kmeans_parallel.joblib")
    print("  Saved kmeans_parallel.joblib")

    # ── 5. NearestCentroid — sequential proxy (AgglomerativeClustering labels) ─
    n_seq = min(4, n)
    # Use existing sequential_cluster labels if available; else cluster fresh
    unique_seq = list(set(seq_labels))
    if len(unique_seq) < 2:
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=n_seq, linkage="ward")
        seq_labels = ac.fit_predict(X_tfidf.toarray()).tolist()

    nc_seq = NearestCentroid()
    nc_seq.fit(X_tfidf.toarray(), seq_labels)
    joblib.dump(nc_seq, MODELS_DIR / "nearest_centroid_sequential.joblib")
    print("  Saved nearest_centroid_sequential.joblib")

    # ── 6. KMeans — complexity (numeric: word_count, avg_sent, has_q, score) ──
    complexity_features = numeric[:, [0, 1, 5, 6]]  # wc, avg_sent, has_q, compliance
    mm_scaler = MinMaxScaler()
    X_cmp = mm_scaler.fit_transform(complexity_features)
    # Reuse mm_scaler for complexity
    joblib.dump(mm_scaler, MODELS_DIR / "scaler_complexity.joblib")

    n_cmp = min(3, n)
    km_cmp = KMeans(n_clusters=n_cmp, random_state=42, n_init=10)
    km_cmp.fit(X_cmp)

    # Build label map: sort clusters by mean complexity score (ascending)
    centers = km_cmp.cluster_centers_
    order   = sorted(range(n_cmp), key=lambda i: float(np.mean(centers[i])))
    names   = ["Базовый", "Средний", "Продвинутый"]
    label_map = {str(cluster_id): names[rank] for rank, cluster_id in enumerate(order)}

    joblib.dump(km_cmp, MODELS_DIR / "kmeans_complexity.joblib")
    (MODELS_DIR / "complexity_label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  Saved kmeans_complexity.joblib + complexity_label_map.json")

    print(f"\n  All models saved to: {MODELS_DIR}")


if __name__ == "__main__":
    print("Training Module G models...")
    train_and_save()
    print("Done.")
