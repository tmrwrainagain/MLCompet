"""Model loader and predictor.

Loads pre-trained sklearn models from model_V_full/models/{current_version}/.
Never calls .fit() — only .predict() / .transform().

Feature pipeline mirrors model_V_full/src/features.py:
  X = hstack([tfidf.transform(texts), csr(scaler.transform(numeric_cols))])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    COMPLEXITY_MULTIPLIERS,
    LESSON_TYPE_MULTIPLIERS,
    MODEL_V_FULL_DIR,
    PRACTICE_OVERHEAD,
    READING_SPEED_WPM,
)

# ── Feature columns (must match model_V_full/src/features.py) ─────────────────
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

# Complexity cluster integer → label (from model_V_full/src/config.py)
COMPLEXITY_LABEL_MAP = {0: "Базовый", 1: "Средний", 2: "Продвинутый"}

# Phase names for sequential clusters
PHASE_NAMES = {
    0: "Ориентация (Знание)",
    1: "Понимание (Компрехенсия)",
    2: "Применение",
    3: "Синтез и Оценка",
}


def _resolve_models_dir() -> Optional[Path]:
    """Return path to current model version dir from model_V_full."""
    version_file = MODEL_V_FULL_DIR / "models" / "current_version.txt"
    if not version_file.exists():
        return None
    version = version_file.read_text(encoding="utf-8").strip()
    model_dir = MODEL_V_FULL_DIR / "models" / f"v_{version}"
    return model_dir if model_dir.exists() else None


class ModelPredictor:
    def __init__(self) -> None:
        self._tfidf   = None   # TfidfVectorizer from model_V_full
        self._scaler  = None   # StandardScaler  from model_V_full
        self._clf_par = None   # parallel_cluster_best.joblib  (LogisticRegression / RF)
        self._clf_seq = None   # sequential_cluster_best.joblib
        self._clf_cmp = None   # complexity_cluster_best.joblib
        self._model_version: str = ""
        self.loaded = False

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all pre-trained models from model_V_full. Called once at API startup."""
        try:
            import joblib

            model_dir = _resolve_models_dir()
            if model_dir is None:
                print(f"  [Predictor] WARNING: model_V_full not found at {MODEL_V_FULL_DIR}")
                return

            self._tfidf   = joblib.load(model_dir / "tfidf_vectorizer.joblib")
            self._scaler  = joblib.load(model_dir / "scaler.joblib")
            self._clf_par = joblib.load(model_dir / "parallel_cluster_best.joblib")
            self._clf_seq = joblib.load(model_dir / "sequential_cluster_best.joblib")
            self._clf_cmp = joblib.load(model_dir / "complexity_cluster_best.joblib")

            meta_path = model_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                self._model_version = meta.get("version", "")
                algs = {k: v.get("algorithm") for k, v in meta.get("models", {}).items()}
                print(f"  [Predictor] Loaded version {self._model_version}: {algs}")

            self.loaded = True
        except Exception as exc:
            print(f"  [Predictor] WARNING: failed to load models: {exc}")
            self.loaded = False

    # ── Feature matrix ────────────────────────────────────────────────────────

    def _build_X(self, texts: List[str], numeric: np.ndarray):
        """Build sparse feature matrix identical to model_V_full/src/features.py."""
        from scipy.sparse import csr_matrix, hstack

        X_text = self._tfidf.transform(texts)
        X_num  = self._scaler.transform(numeric)
        return hstack([X_text, csr_matrix(X_num)])

    def _text_to_numeric(self, text: str) -> np.ndarray:
        """Estimate numeric features from raw text (no DB lookup)."""
        words     = text.split()
        sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        wc  = len(words)
        asl = wc / max(len(sentences), 1)
        hq  = float("?" in text)
        return np.array([[wc, asl, 0, 0, 0, hq, 0.5, 0]], dtype=float)

    def _row_to_numeric(self, row: dict) -> np.ndarray:
        """Convert a DB material dict to numeric feature row."""
        return np.array([[
            float(row.get("word_count") or 0),
            float(row.get("avg_sentence_length") or 0),
            float(row.get("media_count") or 0),
            float(row.get("has_images") or 0),
            float(row.get("has_videos") or 0),
            float(row.get("has_questions") or 0),
            float(row.get("compliance_score_feat") or row.get("compliance_score") or 0.5),
            float(row.get("is_generated") or 0),
        ]], dtype=float)

    # ── Parallel ──────────────────────────────────────────────────────────────

    def predict_parallel(self, texts: List[str]) -> List[int]:
        if not self.loaded:
            return [i % 4 for i in range(len(texts))]
        numeric = np.vstack([self._text_to_numeric(t) for t in texts])
        X = self._build_X(texts, numeric)
        return self._clf_par.predict(X).tolist()

    # ── Sequential ────────────────────────────────────────────────────────────

    def predict_sequential(self, texts: List[str]) -> List[int]:
        if not self.loaded:
            return [0] * len(texts)
        numeric = np.vstack([self._text_to_numeric(t) for t in texts])
        X = self._build_X(texts, numeric)
        return self._clf_seq.predict(X).tolist()

    # ── Complexity ────────────────────────────────────────────────────────────

    def predict_complexity_from_text(self, text: str) -> Tuple[str, int, float]:
        words     = text.split()
        sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        wc  = len(words)
        asl = round(wc / max(len(sentences), 1), 1)

        if not self.loaded:
            return self._heuristic_label(wc, asl), wc, asl

        numeric = self._text_to_numeric(text)
        X = self._build_X([text], numeric)
        cluster = int(self._clf_cmp.predict(X)[0])
        return COMPLEXITY_LABEL_MAP.get(cluster, "Средний"), wc, asl

    def predict_complexity_from_row(self, row: dict) -> str:
        """Predict complexity for a DB material row."""
        if not self.loaded:
            wc  = float(row.get("word_count") or 500)
            asl = float(row.get("avg_sentence_length") or 15)
            return self._heuristic_label(int(wc), asl)
        text    = row.get("text_content") or ""
        numeric = self._row_to_numeric(row)
        X = self._build_X([text], numeric)
        cluster = int(self._clf_cmp.predict(X)[0])
        return COMPLEXITY_LABEL_MAP.get(cluster, "Средний")

    @staticmethod
    def _heuristic_label(word_count: int, avg_sent: float) -> str:
        score = word_count / 500 + avg_sent / 20
        if score < 1.5:
            return "Базовый"
        if score < 3.0:
            return "Средний"
        return "Продвинутый"

    # ── Time estimation ───────────────────────────────────────────────────────

    def estimate_minutes(self, word_count: int, lesson_type: str,
                         difficulty_label: str, has_questions: bool = False,
                         has_videos: bool = False) -> dict:
        lesson_mult = LESSON_TYPE_MULTIPLIERS.get(lesson_type, 1.0)
        diff_mult   = COMPLEXITY_MULTIPLIERS.get(difficulty_label, 1.0)

        reading_min  = word_count / READING_SPEED_WPM * lesson_mult
        complex_oh   = reading_min * (diff_mult - 1.0)
        practice_min = PRACTICE_OVERHEAD.get(difficulty_label, 0) if has_questions else 0
        video_min    = 7 if has_videos else 0
        total        = reading_min + complex_oh + practice_min + video_min

        return {
            "total": max(1, round(total)),
            "breakdown": {
                "reading_minutes":             round(reading_min, 1),
                "complexity_overhead_minutes": round(complex_oh, 1),
                "practice_minutes":            practice_min,
                "video_minutes":               video_min,
            },
        }

    # ── Trajectory ────────────────────────────────────────────────────────────

    def build_trajectory(self, materials: List[dict], difficulty_level: str,
                         available_hours_per_week: float, learning_style: str,
                         target_topics: List[str] = None,
                         exclude_ids: List[int] = None) -> dict:
        exclude_ids = set(exclude_ids or [])
        mats = [m for m in materials if m.get("id") not in exclude_ids]

        if target_topics:
            tl = [t.lower() for t in target_topics]
            filtered = [m for m in mats if any(t in (m.get("topic") or "").lower() for t in tl)]
            mats = filtered or mats

        # Filter by difficulty
        level_order = {"Базовый": 0, "Средний": 1, "Продвинутый": 2}
        target_num  = level_order.get(difficulty_level, 1)

        result = []
        for m in mats:
            dl = m.get("difficulty_label") or self.predict_complexity_from_row(m)
            if level_order.get(dl, 1) <= target_num:
                m["_difficulty_label"] = dl
                result.append(m)

        mats = result or mats
        for m in mats:
            if "_difficulty_label" not in m:
                m["_difficulty_label"] = m.get("difficulty_label") or self.predict_complexity_from_row(m)

        # Sort: sequential_cluster → complexity
        mats.sort(key=lambda m: (m.get("sequential_cluster") or 0, m.get("complexity_cluster") or 0))

        available_minutes = available_hours_per_week * 60
        plan: List[dict] = []
        week, week_min = 1, 0.0
        week_ids: Dict[str, List[int]] = {}

        for mat in mats:
            wc  = int(mat.get("word_count") or mat.get("word_count_f") or 500)
            lt  = mat.get("lesson_type") or "lecture"
            dl  = mat["_difficulty_label"]
            hq  = bool(mat.get("has_questions") or mat.get("has_questions_f"))
            hv  = bool(mat.get("has_videos") or mat.get("has_videos_f"))
            est = self.estimate_minutes(wc, lt, dl, hq, hv)["total"]

            if week_min + est > available_minutes and week_ids.get(str(week)):
                week += 1
                week_min = 0.0

            week_ids.setdefault(str(week), []).append(mat["id"])
            week_min += est

            seq_cl = mat.get("sequential_cluster") or 0
            plan.append({
                "material_id":       mat["id"],
                "topic":             mat.get("topic"),
                "difficulty_label":  dl,
                "estimated_minutes": est,
                "week":              week,
                "rationale":         (
                    f"Стартовый материал фазы {seq_cl} — нет предшественников"
                    if not mat.get("has_previous") else
                    f"Логически следует за предыдущим материалом (фаза {seq_cl})"
                ),
            })

        total_min = sum(s["estimated_minutes"] for s in plan)
        return {
            "trajectory":              plan,
            "total_steps":             len(plan),
            "total_estimated_minutes": total_min,
            "total_weeks":             week,
            "weekly_plan":             week_ids,
        }


predictor = ModelPredictor()
