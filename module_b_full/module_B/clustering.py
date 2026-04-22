"""
Clustering module for Module B.

Implements three types of labelling:
  1. Parallel learning clusters  - materials from independent subject sequences
                                   that can be studied simultaneously.
  2. Sequential learning clusters - materials forming a logical progression
                                   (within one or across multiple subjects).
  3. Complexity clusters          - grouping by difficulty of mastery.
"""

import importlib.util as _ilu
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# Load module_B config explicitly to avoid collision with module_A "config" in sys.modules
_b_cfg_path = Path(__file__).parent / "config.py"
_b_cfg_spec = _ilu.spec_from_file_location("_module_b_config", _b_cfg_path)
_b_cfg = _ilu.module_from_spec(_b_cfg_spec)
_b_cfg_spec.loader.exec_module(_b_cfg)
CLUSTER_SETTINGS = _b_cfg.CLUSTER_SETTINGS
MODEL_FAST = _b_cfg.MODEL_FAST

# Use module_A llm — resolve path to avoid import issues
_mod_a_dir = Path(__file__).parent.parent / "module_A"
if str(_mod_a_dir) not in sys.path:
    sys.path.insert(0, str(_mod_a_dir))
from llm import extract_json_object, generate_text


def load_materials_df() -> pd.DataFrame:
    """Load all materials from PostgreSQL into a DataFrame."""
    from database.manager import get_materials_with_selected_features

    rows = get_materials_with_selected_features()
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _tfidf_matrix(texts: List[str], max_features: int = 500):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=1,
    )
    X = vec.fit_transform(texts)
    return X, vec


def _numeric_features(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "word_count",
        "avg_sentence_length",
        "media_count",
        "has_images",
        "has_videos",
        "has_questions",
        "compliance_score_feat",
    ]
    sub = df[cols].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    from sklearn.preprocessing import StandardScaler

    return StandardScaler().fit_transform(sub.values)


def cluster_parallel(df: pd.DataFrame) -> np.ndarray:
    """
    Materials suitable for parallel study come from independent subject sequences.
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=int)

    from scipy.sparse import csr_matrix, hstack
    from sklearn.cluster import KMeans

    texts = df["text_content"].fillna("").astype(str).tolist()
    X_text, _ = _tfidf_matrix(texts, max_features=300)
    X_num = _numeric_features(df)
    X = hstack([X_text, csr_matrix(X_num)])

    n_clusters = min(CLUSTER_SETTINGS["parallel"]["n_clusters"], len(df))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return km.fit_predict(X).astype(int)


def cluster_sequential(df: pd.DataFrame) -> np.ndarray:
    """
    Materials suitable for sequential study form a logical chain within or across subjects.
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=int)

    from sklearn.cluster import AgglomerativeClustering

    texts = df["text_content"].fillna("").astype(str).tolist()
    X, _ = _tfidf_matrix(texts, max_features=400)
    X_dense = X.toarray()

    n_clusters = min(CLUSTER_SETTINGS["sequential"]["n_clusters"], len(df))
    ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    return ac.fit_predict(X_dense).astype(int)


def cluster_complexity(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Group materials by difficulty of mastery.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler

    feat_cols = ["word_count", "avg_sentence_length", "has_questions", "compliance_score_feat"]
    sub = df[feat_cols].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    X = MinMaxScaler().fit_transform(sub.values)

    n_clusters = min(CLUSTER_SETTINGS["complexity"]["n_clusters"], max(len(df), 1))
    if len(df) < 2:
        return np.zeros(len(df), dtype=int), {0: "Базовый"}

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X).astype(int)

    difficulty_scores = {i: float(np.mean(km.cluster_centers_[i])) for i in range(n_clusters)}
    sorted_clusters = sorted(difficulty_scores, key=difficulty_scores.get)
    default_labels = ["Базовый", "Средний", "Продвинутый"]
    label_map = {cluster_id: default_labels[i] for i, cluster_id in enumerate(sorted_clusters)}
    return labels, label_map


def compute_2d_projection(df: pd.DataFrame, method: str = "tsne") -> Optional[np.ndarray]:
    """
    Returns an (N, 2) array for scatter-plot visualisation.
    method: 'tsne' or 'umap'
    """
    if len(df) < 3:
        return None

    texts = df["text_content"].fillna("").astype(str).tolist()
    X, _ = _tfidf_matrix(texts, max_features=200)
    X_dense = X.toarray()

    try:
        if method == "umap":
            import umap

            reducer = umap.UMAP(
                n_components=2,
                random_state=42,
                n_neighbors=min(15, len(df) - 1),
            )
            return reducer.fit_transform(X_dense)

        from sklearn.manifold import TSNE
        import sklearn

        perplexity = min(30, max(5, len(df) - 1))
        # n_iter renamed to max_iter in sklearn >= 1.4
        tsne_kwargs = {"n_components": 2, "random_state": 42, "perplexity": perplexity}
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_ver >= (1, 4):
            tsne_kwargs["max_iter"] = 1000
        else:
            tsne_kwargs["n_iter"] = 1000
        tsne = TSNE(**tsne_kwargs)
        return tsne.fit_transform(X_dense)
    except Exception as e:
        print(f"  Projection error ({method}): {e}")
        return None


def describe_clusters(df: pd.DataFrame, labels: np.ndarray, cluster_type: str) -> Dict[int, str]:
    """
    Ask Gemini to generate a human-readable label for each cluster
    based on sample topics from the cluster.
    """
    unique_labels = sorted(set(labels.tolist()))
    descriptions: Dict[int, str] = {}

    for lbl in unique_labels:
        mask = labels == lbl
        topics = df.loc[mask, "topic"].dropna().astype(str).head(5).tolist()
        subjects = df.loc[mask, "subject"].dropna().astype(str).unique().tolist()

        prompt = f"""Вы - методист. Дайте краткое название кластера учебных материалов.
Тип кластера: {cluster_type}
Предметы: {', '.join(subjects[:4])}
Темы: {', '.join(topics)}

Верните только JSON:
{{
  "label": "краткое название 3-5 слов"
}}"""

        try:
            data = extract_json_object(generate_text(prompt, model=MODEL_FAST))
            descriptions[lbl] = data.get("label", f"Кластер {lbl}") if data else f"Кластер {lbl}"
        except Exception:
            descriptions[lbl] = f"Кластер {lbl}"

    return descriptions


def run_clustering() -> pd.DataFrame:
    """
    Run all three clusterings, save labels to DB, return enriched DataFrame.
    """
    from database.manager import save_cluster_labels

    df = load_materials_df()
    if df.empty:
        print("  No materials to cluster.")
        return df

    print(f"  Clustering {len(df)} materials...")

    par_labels = cluster_parallel(df)
    par_desc = describe_clusters(df, par_labels, "параллельное изучение")
    df["parallel_cluster"] = par_labels
    df["parallel_cluster_label"] = [par_desc.get(label, f"Кластер {label}") for label in par_labels]

    seq_labels = cluster_sequential(df)
    seq_desc = describe_clusters(df, seq_labels, "последовательное изучение")
    df["sequential_cluster"] = seq_labels
    df["sequential_cluster_label"] = [seq_desc.get(label, f"Кластер {label}") for label in seq_labels]

    cmp_labels, cmp_label_map = cluster_complexity(df)
    df["complexity_cluster"] = cmp_labels
    df["difficulty_label"] = [cmp_label_map.get(label, "Средний") for label in cmp_labels]

    labels_to_save = []
    for _, row in df.iterrows():
        labels_to_save.append(
            {
                "material_id": int(row["id"]),
                "parallel_cluster": int(row["parallel_cluster"]),
                "sequential_cluster": int(row["sequential_cluster"]),
                "complexity_cluster": int(row["complexity_cluster"]),
                "difficulty_label": str(row["difficulty_label"]),
            }
        )
    save_cluster_labels(labels_to_save)

    print("  Cluster labels saved to database.")
    return df
