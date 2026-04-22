"""
Feature extraction and importance analysis.
Computes numerical/categorical features for each material,
determines topic adjacency, ranks features by importance.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import importlib.util as _ilu
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
_a_cfg_path = Path(__file__).parent.parent / "config.py"
_a_cfg_spec = _ilu.spec_from_file_location("_module_a_config", _a_cfg_path)
_a_cfg = _ilu.module_from_spec(_a_cfg_spec)
_a_cfg_spec.loader.exec_module(_a_cfg)
MODEL_FAST = _a_cfg.MODEL_FAST
from database.manager import (
    get_all_materials, get_material_features, get_media_items,
    upsert_feature, upsert_material, save_feature_importance,
)
from llm import extract_json_object, generate_text
from processors.text_extractor import word_count


class FeatureExtractor:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_all_features(self):
        for mat in get_all_materials():
            self._extract(mat)

    def _extract(self, mat: dict):
        mid = mat["id"]
        text = mat.get("text_content") or ""
        media = get_media_items(mid)

        feats = {
            "text_length": len(text),
            "word_count": word_count(text),
            "sentence_count": max(1, len(re.split(r"[.!?]+", text))) if text else 0,
            "has_introduction": int(bool(re.search(r"введени[ея]|вступлени[ея]|introduction", text, re.I))),
            "has_conclusion": int(bool(re.search(r"заключени[ея]|вывод|conclusion", text, re.I))),
            "has_questions": int(bool(re.search(r"вопрос|задани[ея]|упражнени[ея]|question|exercise", text, re.I))),
            "has_images": int(any(m["media_type"] == "image" for m in media)),
            "has_videos": int(any(m["media_type"] == "video" for m in media)),
            "has_audio": int(any(m["media_type"] == "audio" for m in media)),
            "media_count": len(media),
            "compliance_score": float(mat.get("compliance_score") or 0),
            "is_generated": int(bool(mat.get("is_generated"))),
        }
        feats["avg_sentence_length"] = (
            feats["word_count"] / feats["sentence_count"] if feats["sentence_count"] else 0
        )

        for name, val in feats.items():
            upsert_feature(mid, name, value=float(val))

        # Categorical
        upsert_feature(mid, "subject", value_text=mat.get("subject") or "")
        upsert_feature(mid, "topic", value_text=mat.get("topic") or "")
        upsert_feature(mid, "file_type", value_text=mat.get("file_type") or "")
        upsert_feature(mid, "language", value_text=mat.get("language") or "")

    # ------------------------------------------------------------------
    # Topic adjacency
    # ------------------------------------------------------------------

    def compute_topic_adjacency(self):
        """Ask Gemini to order materials by subject and store prev/next links."""
        all_mats = get_all_materials()
        by_subject: Dict[str, list] = {}
        for m in all_mats:
            s = m.get("subject") or "Общее"
            by_subject.setdefault(s, []).append(m)

        for subject, items in by_subject.items():
            if len(items) < 2:
                continue
            topics = "\n".join(f"ID {m['id']}: {m.get('topic', 'N/A')}" for m in items)
            prompt = f"""Определите логическую последовательность тем по предмету «{subject}».

Темы:
{topics}

Верните ТОЛЬКО JSON:
{{
  "ordered_ids": [<id1>, <id2>, ...]
}}"""

            try:
                data = extract_json_object(generate_text(prompt, model=MODEL_FAST))
                ordered = data.get("ordered_ids", [])
                if not ordered:
                    continue
                for i, mid in enumerate(ordered):
                    mat = next((x for x in items if x["id"] == mid), None)
                    if not mat:
                        continue
                    upd: dict = {"url": mat["url"]}
                    upd["has_previous"] = i > 0
                    upd["has_next"] = i < len(ordered) - 1
                    if i > 0:
                        upd["previous_material_id"] = ordered[i - 1]
                    if i < len(ordered) - 1:
                        upd["next_material_id"] = ordered[i + 1]
                    upsert_material(upd)
            except Exception as e:
                print(f"  Adjacency error ({subject}): {e}")

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def analyse_importance(self) -> List[Dict]:
        """
        Rank features by importance using:
          1. Variance
          2. Correlation with compliance_score
          3. SHAP (if sklearn+shap available)
          4. Permutation importance (sklearn fallback)
        """
        mats = get_all_materials()
        if len(mats) < 3:
            print("  Not enough materials for feature importance analysis.")
            return []

        numeric_feats = [
            "text_length", "word_count", "sentence_count", "avg_sentence_length",
            "has_introduction", "has_conclusion", "has_questions",
            "has_images", "has_videos", "has_audio", "media_count",
            "compliance_score", "is_generated",
        ]

        rows = []
        for m in mats:
            feats = get_material_features(m["id"])
            rows.append({f: float(feats.get(f) or 0) for f in numeric_feats})

        df = pd.DataFrame(rows)
        importance: Dict[str, float] = {}

        # --- Variance ---
        variances = df.var()
        max_var = variances.max() or 1
        for feat, v in variances.items():
            importance[feat] = float(v / max_var)

        # --- Correlation with compliance_score (ANOVA proxy) ---
        target = df["compliance_score"]
        if target.std() > 0:
            for feat in df.columns:
                if feat == "compliance_score":
                    continue
                corr = abs(df[feat].corr(target))
                if not np.isnan(corr):
                    importance[feat] = (importance.get(feat, 0) + corr) / 2

        # --- SHAP / permutation importance ---
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance

            X = df.drop(columns=["compliance_score"])
            y = df["compliance_score"]
            if y.std() > 0 and len(X) >= 5:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X, y)

                try:
                    import shap
                    expl = shap.TreeExplainer(rf)
                    shap_vals = np.abs(expl.shap_values(X)).mean(axis=0)
                    max_shap = shap_vals.max() or 1
                    for i, feat in enumerate(X.columns):
                        shap_score = float(shap_vals[i] / max_shap)
                        importance[feat] = (importance.get(feat, 0) + shap_score) / 2
                    print("  SHAP importance computed.")
                except ImportError:
                    # Permutation importance fallback
                    perm = permutation_importance(rf, X, y, n_repeats=5, random_state=42)
                    max_pi = perm.importances_mean.max() or 1
                    for i, feat in enumerate(X.columns):
                        pi_score = float(max(perm.importances_mean[i], 0) / max_pi)
                        importance[feat] = (importance.get(feat, 0) + pi_score) / 2
                    print("  Permutation importance computed.")
        except ImportError:
            print("  sklearn not available; using variance+correlation only.")
        except Exception as e:
            print(f"  Importance error: {e}")

        results = sorted(
            [{"feature_name": k, "importance_score": v, "method": "combined"} for k, v in importance.items()],
            key=lambda x: x["importance_score"],
            reverse=True,
        )
        save_feature_importance(results)
        return results
