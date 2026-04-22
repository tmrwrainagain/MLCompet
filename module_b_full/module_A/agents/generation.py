"""
Generation agent: analyses curriculum gaps, proposes new materials to the user,
generates them with Gemini, runs auto-moderation, saves to DB.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import importlib.util as _ilu
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
_a_cfg_path = Path(__file__).parent.parent / "config.py"
_a_cfg_spec = _ilu.spec_from_file_location("_module_a_config", _a_cfg_path)
_a_cfg = _ilu.module_from_spec(_a_cfg_spec)
_a_cfg_spec.loader.exec_module(_a_cfg)
MODEL_PRO = _a_cfg.MODEL_PRO
from database.manager import get_all_materials, upsert_material
from agents.moderation import ModerationAgent
from llm import extract_json_object, generate_text


class GenerationAgent:
    def __init__(self):
        self.moderation = ModerationAgent()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_interactive(self):
        """Identify gaps → offer choices → generate → moderate → save."""
        print("\n[Generation] Analysing curriculum for gaps...")
        gaps = self._find_gaps()

        if not gaps:
            print("  No curriculum gaps detected.")
            return

        print(f"\n  Found {len(gaps)} potential gaps:\n")
        for i, g in enumerate(gaps, 1):
            print(f"  {i}. [{g.get('subject')}] {g.get('topic')}")
            print(f"     {g.get('description', '')[:100]}")

        print("\nGenerate materials for which gaps? (numbers, comma-sep / 'all' / 'none'): ", end="")
        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSkipped.")
            return

        if choice.lower() == "none":
            return

        selected = gaps if choice.lower() == "all" else self._parse_selection(choice, gaps)

        for gap in selected:
            self._generate_and_save(gap)

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def _find_gaps(self) -> List[Dict]:
        materials = get_all_materials()
        if not materials:
            return []

        by_subject: Dict[str, list] = {}
        for m in materials:
            s = m.get("subject") or "Общее"
            by_subject.setdefault(s, []).append(m)

        all_gaps = []
        for subject, items in by_subject.items():
            topic_list = "\n".join(
                f"- {it['topic']}: {(it.get('annotation') or '')[:80]}"
                for it in items if it.get("topic")
            )
            if not topic_list:
                continue

            prompt = f"""Вы — методист. Проанализируйте темы по предмету «{subject}» и определите пробелы.

Имеющиеся темы:
{topic_list}

Верните ТОЛЬКО JSON:
{{
  "gaps": [
    {{
      "subject": "{subject}",
      "topic": "название недостающей темы",
      "description": "почему важна для непрерывности программы",
      "position": "before|between|after",
      "reference_topic": "тема, рядом с которой должна идти"
    }}
  ]
}}"""

            try:
                data = extract_json_object(generate_text(prompt, model=MODEL_PRO))
                if data:
                    all_gaps.extend(data.get("gaps", []))
            except Exception as e:
                print(f"  Gap analysis error ({subject}): {e}")

        return all_gaps

    # ------------------------------------------------------------------
    # Material generation
    # ------------------------------------------------------------------

    def _generate_and_save(self, gap: Dict):
        topic = gap.get("topic", "")
        subject = gap.get("subject", "")
        context = gap.get("description", "")
        print(f"\n[Generation] Generating: [{subject}] {topic}")

        prompt = f"""Создайте полноценный учебный материал.

Предмет: {subject}
Тема: {topic}
Контекст: {context}

Структура: Введение с целями → Теоретическая часть → Примеры → Контрольные вопросы → Заключение.

Верните ТОЛЬКО JSON:
{{
  "subject": "{subject}",
  "topic": "{topic}",
  "text_content": "полный текст материала",
  "annotation": "краткая аннотация 2-3 предложения"
}}"""

        try:
            data = extract_json_object(generate_text(prompt, model=MODEL_PRO))
            if not data:
                print("  Failed to parse generated content.")
                return
        except Exception as e:
            print(f"  Generation error: {e}")
            return

        url = f"generated://{subject.replace(' ', '_')}/{topic.replace(' ', '_')}"
        data["url"] = url
        data["is_generated"] = True
        data["moderation_status"] = "pending"

        mid = upsert_material(data)
        print(f"  Saved id={mid}")

        # Auto-moderation
        print(f"  Auto-moderating...")
        res = self.moderation.moderate_material(mid)
        print(f"  Moderation: status={res.get('moderation_status')}  score={res.get('compliance_score')}/10")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_selection(choice: str, gaps: List[Dict]) -> List[Dict]:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            return [gaps[i] for i in indices if 0 <= i < len(gaps)]
        except ValueError:
            return []
