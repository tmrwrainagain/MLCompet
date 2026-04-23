"""Feature engineering shared across all Module V notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack


NUMERIC_COLS = [
    "word_count",
    "avg_sentence_length",
    "media_count",
    "has_images",
    "has_videos",
    "has_questions",
    "compliance_score_feat",
    "is_generated",
]


def build_features(df: pd.DataFrame, tfidf_vec: TfidfVectorizer | None = None,
                   scaler: StandardScaler | None = None,
                   max_tfidf: int = 300) -> tuple:
    """
    Build feature matrix from DataFrame.

    Returns (X_sparse, tfidf_vec, scaler) so callers can reuse fitted
    transformers on new data (pass them back in on subsequent calls).
    """
    texts = df["text_content"].fillna("").astype(str).tolist()

    if tfidf_vec is None:
        tfidf_vec = TfidfVectorizer(
            max_features=max_tfidf,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=1,
        )
        X_text = tfidf_vec.fit_transform(texts)
    else:
        X_text = tfidf_vec.transform(texts)

    num = df[NUMERIC_COLS].fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)

    if scaler is None:
        scaler = StandardScaler(with_mean=False)
        X_num = scaler.fit_transform(num.values)
    else:
        X_num = scaler.transform(num.values)

    X = hstack([X_text, csr_matrix(X_num)])
    return X, tfidf_vec, scaler


def subject_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode subject and lesson_type columns."""
    return pd.get_dummies(df[["subject", "lesson_type"]], drop_first=False).astype(np.float32)
