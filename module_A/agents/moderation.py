"""
Moderation agent: checks each material against methodological guidelines
using Gemini 2.5 Flash, updates the database with overall and per-requirement results.
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import importlib.util as _ilu
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
_a_cfg_path = Path(__file__).parent.parent / "config.py"
_a_cfg_spec = _ilu.spec_from_file_location("_module_a_config", _a_cfg_path)
_a_cfg = _ilu.module_from_spec(_a_cfg_spec)
_a_cfg_spec.loader.exec_module(_a_cfg)
GEMINI_API_KEY = _a_cfg.GEMINI_API_KEY
MODEL_FAST = _a_cfg.MODEL_FAST
METHODOLOGICAL_GUIDELINES = _a_cfg.METHODOLOGICAL_GUIDELINES
from database.manager import (
    get_all_materials, get_material_by_id, get_media_items,
    get_all_requirements, upsert_material, upsert_compliance,
)
from llm import extract_json_array, extract_json_object, generate_text


class ModerationAgent:
    def __init__(self):
        self._requirements: List[Dict] = []

    def _load_requirements(self):
        if not self._requirements:
            self._requirements = get_all_requirements()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def moderate_all(self) -> Dict[int, dict]:
        """Moderate every material (skip already fully moderated ones)."""
        self._load_requirements()
        results = {}
        for mat in get_all_materials():
            mid = mat["id"]
            if mat.get("moderation_status") in ("approved", "rejected") and mat.get("compliance_score"):
                print(f"  [Moderation] Skip id={mid} (already moderated)")
                continue
            print(f"\n[Moderation] id={mid} — {mat.get('topic', 'N/A')}")
            result = self.moderate_material(mid)
            results[mid] = result
            print(
                f"  status={result.get('moderation_status')}  "
                f"score={result.get('compliance_score')}/10"
            )
        return results

    def moderate_material(self, material_id: int) -> dict:
        self._load_requirements()
        mat = get_material_by_id(material_id)
        if not mat:
            return {"moderation_status": "error", "moderation_notes": "Material not found"}

        media_items = get_media_items(material_id)
        parts: List[str] = []
        if mat.get("text_content"):
            parts.append(f"Текст материала:\n{mat['text_content'][:2500]}")
        if media_items:
            media_desc = "\n".join(
                f"- {it['media_type']}: {(it['description'] or '')[:200]}"
                for it in media_items
            )
            parts.append(f"Медиаконтент:\n{media_desc}")

        if not parts:
            return {"moderation_status": "pending", "moderation_notes": "Нет контента для модерации"}

        # ── Overall moderation ───────────────────────────────────────────
        overall = self._moderate_overall(mat, parts)

        # ── Per-requirement compliance ───────────────────────────────────
        self._moderate_per_requirement(mat, parts, material_id)

        return overall

    # ------------------------------------------------------------------
    # Overall moderation
    # ------------------------------------------------------------------

    def _moderate_overall(self, mat: dict, parts: List[str]) -> dict:
        prompt = f"""{METHODOLOGICAL_GUIDELINES}

МАТЕРИАЛ:
Предмет: {mat.get('subject', 'Не определён')}
Тема: {mat.get('topic', 'Не определена')}
{chr(10).join(parts)}

Оцените соответствие методическим рекомендациям. Верните ТОЛЬКО JSON:
{{
  "compliance_score": <число 0-10>,
  "is_compliant": <true|false>,
  "moderation_status": "<approved|rejected|needs_revision>",
  "moderation_notes": "развёрнутые выводы о соответствии",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "recommendations": "рекомендации по улучшению"
}}"""

        try:
            data = extract_json_object(generate_text(prompt, model=MODEL_FAST))
            if data:
                notes = data.get("moderation_notes", "")
                recs  = data.get("recommendations", "")
                upsert_material({
                    "url":               mat["url"],
                    "compliance_score":  data.get("compliance_score"),
                    "is_compliant":      data.get("is_compliant"),
                    "moderation_status": data.get("moderation_status", "pending"),
                    "moderation_notes":  f"{notes}\n\nРекомендации: {recs}".strip(),
                })
                return data
        except Exception as e:
            print(f"  Overall moderation error: {e}")
        return {"moderation_status": "error", "moderation_notes": "AI error"}

    # ------------------------------------------------------------------
    # Per-requirement compliance
    # ------------------------------------------------------------------

    def _moderate_per_requirement(self, mat: dict, parts: List[str], material_id: int):
        """
        Send a single prompt to Gemini asking it to evaluate every requirement.
        Results are saved to methodology_compliance table.
        """
        req_list = "\n".join(
            f'  {i+1}. [{r["category"]}] {r["requirement"]}: {r["description"]}'
            for i, r in enumerate(self._requirements)
        )

        prompt = f"""Вы — эксперт по оценке учебных материалов.

МАТЕРИАЛ:
Предмет: {mat.get('subject', 'Не определён')}
Тема: {mat.get('topic', 'Не определена')}
{chr(10).join(parts)}

ТРЕБОВАНИЯ (нумерованный список):
{req_list}

Для каждого требования оцените, выполнено ли оно в данном материале.
Верните ТОЛЬКО JSON-массив (индексы совпадают с номерами требований):
[
  {{
    "requirement_index": 1,
    "is_met": true,
    "score": 8.5,
    "notes": "краткое обоснование"
  }},
  ...
]"""

        try:
            results = extract_json_array(generate_text(prompt, model=MODEL_FAST))
            if not results:
                return

            for item in results:
                idx = item.get("requirement_index", 0) - 1
                if 0 <= idx < len(self._requirements):
                    req_id = self._requirements[idx]["id"]
                    upsert_compliance(
                        material_id    = material_id,
                        requirement_id = req_id,
                        is_met         = bool(item.get("is_met", False)),
                        score          = float(item.get("score", 0)),
                        notes          = item.get("notes", ""),
                    )
        except Exception as e:
            print(f"  Per-requirement moderation error: {e}")
