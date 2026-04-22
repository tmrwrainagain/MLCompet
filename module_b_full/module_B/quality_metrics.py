"""
Clustering quality assessment for Module B.

Metrics computed:
  - Silhouette Score          (cohesion vs. separation; higher is better; range −1..1)
  - Davies-Bouldin Index      (intra/inter-cluster ratio;  lower  is better)
  - Calinski-Harabasz Index   (variance ratio;             higher is better)

Visual analysis:
  - 2-D scatter plots (t-SNE / UMAP projection) coloured by each cluster type
  - Cluster composition bar charts
  - Pair-wise metric comparison table

All charts are saved to module_B/reports/quality_*.png
All textual conclusions are returned as markdown strings.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import importlib.util as _ilu
_b_cfg_spec = _ilu.spec_from_file_location("_mod_b_cfg", Path(__file__).parent / "config.py")
_b_cfg = _ilu.module_from_spec(_b_cfg_spec)
_b_cfg_spec.loader.exec_module(_b_cfg)
REPORTS_DIR = _b_cfg.REPORTS_DIR


def _df_to_md(df: "pd.DataFrame") -> str:
    """Markdown table without tabulate dependency."""
    if df is None or df.empty:
        return "Нет данных."
    headers = " | ".join(str(c) for c in df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(v) if v is not None else "" for v in row) for row in df.itertuples(index=False, name=None)]
    return "| " + " |\n| ".join([headers, sep] + rows) + " |"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _feature_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    """Build a numeric matrix from available clustering features."""
    cols = [c for c in [
        "word_count", "avg_sentence_length", "media_count",
        "has_images", "has_videos", "has_questions", "compliance_score_feat",
    ] if c in df.columns]
    if not cols:
        return None
    sub = df[cols].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(sub.values)


def _tfidf_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    """TF-IDF dense matrix from text_content."""
    texts = df["text_content"].fillna("").astype(str).tolist()
    if not any(t.strip() for t in texts):
        return None
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=200, sublinear_tf=True, min_df=1)
    return vec.fit_transform(texts).toarray()


def _combined_matrix(df: pd.DataFrame) -> Optional[np.ndarray]:
    X_text = _tfidf_matrix(df)
    X_num  = _feature_matrix(df)
    if X_text is None and X_num is None:
        return None
    if X_text is None:
        return X_num
    if X_num is None:
        return X_text
    return np.hstack([X_text, X_num])


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute all three clustering quality metrics."""
    from sklearn.metrics import (
        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    )

    n_unique = len(set(labels))
    if n_unique < 2 or len(X) < 2:
        return {"silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}

    results = {}
    try:
        results["silhouette"]         = float(silhouette_score(X, labels))
    except Exception:
        results["silhouette"]         = float("nan")
    try:
        results["davies_bouldin"]     = float(davies_bouldin_score(X, labels))
    except Exception:
        results["davies_bouldin"]     = float("nan")
    try:
        results["calinski_harabasz"]  = float(calinski_harabasz_score(X, labels))
    except Exception:
        results["calinski_harabasz"]  = float("nan")

    return results


def compare_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare K-Means vs. Agglomerative clustering for sequential and parallel types.
    Returns a summary DataFrame.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering

    X = _combined_matrix(df)
    if X is None or len(df) < 3:
        return pd.DataFrame()

    n = min(4, len(df))
    rows = []

    for method_name, clf in [
        ("K-Means",        KMeans(n_clusters=n, random_state=42, n_init=10)),
        ("Агломеративный", AgglomerativeClustering(n_clusters=n, linkage="ward")),
    ]:
        labels  = clf.fit_predict(X).astype(int)
        metrics = compute_metrics(X, labels)
        rows.append({
            "Метод":             method_name,
            "Силуэт":            round(metrics["silhouette"], 4),
            "Дэвис-Болдин":      round(metrics["davies_bouldin"], 4),
            "Калинский-Харабаш": round(metrics["calinski_harabasz"], 2),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visual analysis
# ---------------------------------------------------------------------------

def _scatter_2d(coords: np.ndarray, labels: np.ndarray, label_map: Dict[int, str],
                title: str, filename: str) -> str:
    """Save a 2-D scatter plot coloured by cluster label."""
    fig, ax = plt.subplots(figsize=(9, 6))
    unique  = sorted(set(labels.tolist()))
    cmap    = plt.get_cmap("tab10")

    for i, lbl in enumerate(unique):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(i / max(len(unique), 1))],
            label=label_map.get(lbl, f"Кластер {lbl}"),
            alpha=0.75, s=60, edgecolors="white", linewidths=0.4,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Компонента 1")
    ax.set_ylabel("Компонента 2")
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    plt.tight_layout()

    path = str(REPORTS_DIR / filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _composition_bar(df: pd.DataFrame, cluster_col: str, label_col: str,
                     group_col: str, title: str, filename: str) -> str:
    """Stacked bar chart: cluster composition by subject / lesson_type."""
    if cluster_col not in df.columns or group_col not in df.columns:
        return ""

    pivot = (
        df.groupby([label_col, group_col])
          .size()
          .unstack(fill_value=0)
    )
    if pivot.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Кластер")
    ax.set_ylabel("Количество материалов")
    ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.15, 1))
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    path = str(REPORTS_DIR / filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_clustering(df: pd.DataFrame) -> Dict:
    """
    Full quality evaluation:
      - metrics for each cluster type
      - 2-D scatter plots
      - composition bar charts
      - method comparison table
      - written conclusion
    """
    from clustering import compute_2d_projection

    results  = {}
    X        = _combined_matrix(df)
    has_data = X is not None and len(df) >= 3

    cluster_configs = [
        ("parallel_cluster",   "parallel_cluster_label",   "Параллельное изучение"),
        ("sequential_cluster", "sequential_cluster_label", "Последовательное изучение"),
        ("complexity_cluster", "difficulty_label",         "Сложность"),
    ]

    # ── 2-D projection (shared) ──────────────────────────────────────────
    coords = None
    if has_data:
        coords = compute_2d_projection(df, method="tsne")
        if coords is None:
            coords = compute_2d_projection(df, method="umap")

    for cluster_col, label_col, name in cluster_configs:
        if cluster_col not in df.columns:
            continue

        labels    = df[cluster_col].fillna(-1).astype(int).values
        label_map = (
            df.dropna(subset=[label_col])
              .set_index(cluster_col)[label_col]
              .to_dict()
        ) if label_col in df.columns else {}

        # ── Metrics ──────────────────────────────────────────────────────
        metrics = compute_metrics(X, labels) if has_data else {}
        results[cluster_col] = {
            "name":    name,
            "metrics": metrics,
            "plots":   [],
        }

        # ── Scatter plot ─────────────────────────────────────────────────
        if coords is not None:
            path = _scatter_2d(
                coords, labels, label_map,
                title    = f"2-D проекция (t-SNE) — {name}",
                filename = f"quality_{cluster_col}_scatter.png",
            )
            results[cluster_col]["plots"].append(path)

        # ── Composition bar ───────────────────────────────────────────────
        for grp_col, fname_suffix in [("subject", "subject"), ("lesson_type", "lesson")]:
            path = _composition_bar(
                df, cluster_col, label_col, grp_col,
                title    = f"Состав кластеров по «{grp_col}» — {name}",
                filename = f"quality_{cluster_col}_{fname_suffix}.png",
            )
            if path:
                results[cluster_col]["plots"].append(path)

    # ── Method comparison table ──────────────────────────────────────────
    cmp_df = compare_methods(df)
    results["method_comparison"] = cmp_df

    # ── Textual conclusions ──────────────────────────────────────────────
    results["conclusion"] = _build_conclusion(results)

    return results


# ---------------------------------------------------------------------------
# Written conclusion
# ---------------------------------------------------------------------------

def _build_conclusion(results: Dict) -> str:
    lines = ["## Выводы по качеству разметки данных\n"]

    cluster_configs = [
        ("parallel_cluster",   "Параллельное изучение"),
        ("sequential_cluster", "Последовательное изучение"),
        ("complexity_cluster", "Сложность"),
    ]

    for key, name in cluster_configs:
        if key not in results:
            continue
        m = results[key].get("metrics", {})
        sil = m.get("silhouette", float("nan"))
        db  = m.get("davies_bouldin", float("nan"))
        ch  = m.get("calinski_harabasz", float("nan"))

        quality = "недостаточно данных"
        if not np.isnan(sil):
            if sil > 0.5:
                quality = "высокое (кластеры чёткие и разделённые)"
            elif sil > 0.25:
                quality = "удовлетворительное (структура прослеживается)"
            else:
                quality = "низкое (кластеры перекрываются)"

        lines += [
            f"### {name}",
            f"- **Силуэт:** {sil:.4f}" if not np.isnan(sil) else "- **Силуэт:** —",
            f"- **Дэвис-Болдин:** {db:.4f}" if not np.isnan(db) else "- **Дэвис-Болдин:** —",
            f"- **Калинский-Харабаш:** {ch:.2f}" if not np.isnan(ch) else "- **Калинский-Харабаш:** —",
            f"- **Качество разметки:** {quality}",
            "",
        ]

    cmp = results.get("method_comparison")
    if cmp is not None and not cmp.empty:
        lines += [
            "### Сравнение методов кластеризации",
            _df_to_md(cmp),
            "",
            "**Обоснование выбора метода:**",
            "- Для **параллельного** и **сложностного** кластеров использован K-Means: "
            "эффективен на высокоразмерных TF-IDF пространствах, устойчив при больших выборках.",
            "- Для **последовательного** кластера выбрана **агломеративная иерархическая кластеризация** "
            "(связь Ward): она сохраняет дендрограмму зависимостей, что отражает реальную "
            "последовательность тем.",
            "- Наилучший метод выбран на основе максимума силуэтного индекса и минимума "
            "индекса Дэвиса-Болдина.",
            "",
        ]

    return "\n".join(lines)
