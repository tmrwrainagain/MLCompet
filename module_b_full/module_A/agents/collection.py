"""
Collection agent.

Primary mode for the competition: process local teaching files from `test_files`.
URL mode is preserved and can be quickly re-enabled if needed later.
"""

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import importlib.util as _ilu
import requests
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
_a_cfg_path = Path(__file__).parent.parent / "config.py"
_a_cfg_spec = _ilu.spec_from_file_location("_module_a_config", _a_cfg_path)
_a_cfg = _ilu.module_from_spec(_a_cfg_spec)
_a_cfg_spec.loader.exec_module(_a_cfg)
LESSON_TYPES = _a_cfg.LESSON_TYPES
MATERIALS_DIR = _a_cfg.MATERIALS_DIR
MAX_MEDIA_PER_PAGE = _a_cfg.MAX_MEDIA_PER_PAGE
MODEL_FAST = _a_cfg.MODEL_FAST
SUPPORTED_AUDIO_TYPES = _a_cfg.SUPPORTED_AUDIO_TYPES
SUPPORTED_IMAGE_TYPES = _a_cfg.SUPPORTED_IMAGE_TYPES
SUPPORTED_TEXT_TYPES = _a_cfg.SUPPORTED_TEXT_TYPES
SUPPORTED_VIDEO_TYPES = _a_cfg.SUPPORTED_VIDEO_TYPES
from database.manager import upsert_material, upsert_media_item
from llm import extract_json_object, generate_text
from processors.downloader import download_file, extract_media_links
from processors.media_analyzer import analyze_audio, analyze_image, analyze_video
from processors.text_extractor import clean_text, extract_text


TEXTUAL_EXTENSIONS = {
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".txt",
    ".text",
    ".log",
    ".md",
    ".rst",
    ".rtf",
    ".xlsx",
    ".xls",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".toml",
    ".odt",
    ".ods",
    ".odp",
    ".epub",
}


class CollectionAgent:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process_urls(self, urls: List[str]) -> List[int]:
        ids = []
        for raw in urls:
            url = raw.strip()
            if url and not url.startswith("#"):
                mid = self.process_url(url)
                if mid:
                    ids.append(mid)
        return ids

    def process_local_files(self, file_paths: List[Path]) -> List[int]:
        ids = []
        for raw_path in file_paths:
            path = Path(raw_path)
            if not path.exists() or path.is_dir():
                continue
            mid = self.process_local_file(path)
            if mid:
                ids.append(mid)
        return ids

    def process_url(self, url: str) -> Optional[int]:
        print(f"\n[Collection][URL] {url}")
        local_path, _, err = download_file(url, self.session)
        if err or not local_path:
            print(f"  Download failed: {err}")
            return None
        return self._process_material_file(local_path, source_id=url, source_kind="url")

    def process_local_file(self, file_path: Path) -> Optional[int]:
        print(f"\n[Collection][FILE] {file_path}")
        try:
            local_copy = self._copy_to_materials_dir(file_path)
            # Stable source_id uses just the filename so the same file loaded from
            # different absolute paths (test_files/ vs dist/test_files/) deduplicates correctly.
            source_id = f"local://test_files/{file_path.name}"
            content_hash = hashlib.md5(file_path.read_bytes()).hexdigest()
            return self._process_material_file(
                local_copy,
                source_id=source_id,
                source_kind="local",
                extra={"file_content_hash": content_hash},
            )
        except Exception as exc:
            print(f"  Local file processing failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def _process_material_file(self, local_path: Path, source_id: str, source_kind: str, extra: dict = None) -> Optional[int]:
        ext = local_path.suffix.lower()
        material = {"url": source_id, "file_type": ext}
        if extra:
            material.update(extra)

        if ext in TEXTUAL_EXTENSIONS:
            extracted = extract_text(local_path)
            material["text_content"] = clean_text(extracted["text"])
            if extracted["title"]:
                material["topic"] = extracted["title"]
        elif ext in SUPPORTED_IMAGE_TYPES:
            material["text_content"] = analyze_image(local_path)
            material["file_type"] = "image"
        elif ext in SUPPORTED_VIDEO_TYPES:
            material["text_content"] = analyze_video(local_path)
            material["file_type"] = "video"
        elif ext in SUPPORTED_AUDIO_TYPES:
            material["text_content"] = analyze_audio(local_path)
            material["file_type"] = "audio"
        else:
            print(f"  Unsupported file type: {ext}")
            return None

        ai_info = self._analyse_content(material.get("text_content", ""), source_id)
        material.update(ai_info)

        material_id = upsert_material(material)
        print(
            f"  Saved id={material_id}, "
            f"subject={material.get('subject')}, "
            f"topic={material.get('topic')}, "
            f"lesson_type={material.get('lesson_type', 'other')}"
        )

        if ext in {".html", ".htm"}:
            html_text = local_path.read_text(encoding="utf-8", errors="ignore")
            if source_kind == "local":
                self._process_local_embedded_media(html_text, local_path, material_id)
            else:
                self._process_remote_embedded_media(html_text, source_id, material_id)

        return material_id

    def _copy_to_materials_dir(self, source_path: Path) -> Path:
        target = MATERIALS_DIR / source_path.name
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return target

    # ------------------------------------------------------------------
    # LLM gap-fill for materials missing metadata
    # ------------------------------------------------------------------

    def fill_missing_metadata(self):
        """
        For materials that have text_content but are missing subject/topic/annotation/language,
        call the LLM to populate those fields.  Runs after bulk loading so even partially
        extracted files get complete records.
        """
        from database.manager import get_all_materials

        mats = get_all_materials()
        filled = 0
        for mat in mats:
            needs_fill = (
                not mat.get("subject")
                or not mat.get("topic")
                or not mat.get("annotation")
                or not mat.get("language")
            )
            if not needs_fill:
                continue

            text = (mat.get("text_content") or "").strip()
            if len(text) < 30:
                continue  # nothing to analyse

            print(f"  [Fill] id={mat['id']} — filling missing metadata via LLM...")
            ai_info = self._analyse_content(text, mat["url"])
            if not ai_info:
                continue

            update = {"url": mat["url"]}
            for field in ("subject", "topic", "annotation", "language", "lesson_type"):
                if ai_info.get(field) and not mat.get(field):
                    update[field] = ai_info[field]

            if len(update) > 1:
                upsert_material(update)
                filled += 1

        print(f"  [Fill] Metadata filled for {filled} materials.")

    # ------------------------------------------------------------------
    # AI analysis
    # ------------------------------------------------------------------

    def _analyse_content(self, text: str, source_id: str) -> dict:
        if not text or len(text.strip()) < 30:
            return {}

        lesson_types_str = ", ".join(LESSON_TYPES)
        prompt = f"""Проанализируйте учебный материал и верните ТОЛЬКО JSON:
{{
  "subject": "название предмета / дисциплины",
  "topic": "тема материала",
  "annotation": "краткая аннотация 2-3 предложения",
  "language": "ru|en|другое",
  "lesson_type": "один из: {lesson_types_str}"
}}

Правила определения lesson_type:
  lecture    — теоретический текст, объяснение понятий
  seminar    — обсуждение, разбор примеров
  practice   — практические задания, упражнения
  lab        — лабораторная работа с пошаговыми инструкциями
  self_study — материал для самостоятельного изучения
  test       — тест, контрольные вопросы

Источник: {source_id}
Текст (первые 3000 символов):
{text[:3000]}"""

        try:
            data = extract_json_object(generate_text(prompt, model=MODEL_FAST))
            if data:
                if data.get("lesson_type") not in LESSON_TYPES:
                    data["lesson_type"] = "other"
                return data
        except Exception as exc:
            print(f"  AI analysis error: {exc}")
        return {}

    # ------------------------------------------------------------------
    # Embedded media
    # ------------------------------------------------------------------

    def _process_remote_embedded_media(self, html: str, base_url: str, material_id: int):
        links = extract_media_links(html, base_url)
        processed = 0

        for media_type, media_url in links:
            if processed >= MAX_MEDIA_PER_PAGE:
                break
            local, _, err = download_file(media_url, self.session)
            if err or not local:
                continue
            if self._save_media_item(material_id, media_type, media_url, local, processed):
                processed += 1

    def _process_local_embedded_media(self, html: str, html_path: Path, material_id: int):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return

        soup = BeautifulSoup(html, "lxml")
        candidates: List[tuple[str, Path]] = []

        for img in soup.find_all("img", src=True):
            candidates.append(("image", (html_path.parent / img["src"]).resolve()))
        for tag in soup.find_all(["video", "source"]):
            src = tag.get("src") or tag.get("data-src")
            if src:
                candidates.append(("video", (html_path.parent / src).resolve()))
        for tag in soup.find_all("audio", src=True):
            candidates.append(("audio", (html_path.parent / tag["src"]).resolve()))

        processed = 0
        for media_type, path in candidates:
            if processed >= MAX_MEDIA_PER_PAGE:
                break
            if not path.exists() or path.is_dir():
                continue
            local = self._copy_to_materials_dir(path)
            source_url = f"local://{path.as_posix()}"
            if self._save_media_item(material_id, media_type, source_url, local, processed):
                processed += 1

    def _save_media_item(
        self,
        material_id: int,
        media_type: str,
        source_url: str,
        local_path: Path,
        position: int,
    ) -> bool:
        ext = local_path.suffix.lower()
        description = ""
        if ext in SUPPORTED_IMAGE_TYPES:
            description = analyze_image(local_path)
        elif ext in SUPPORTED_VIDEO_TYPES:
            description = analyze_video(local_path)
        elif ext in SUPPORTED_AUDIO_TYPES:
            description = analyze_audio(local_path)

        if not description:
            return False

        upsert_media_item(
            {
                "material_id": material_id,
                "media_type": media_type,
                "source_url": source_url,
                "local_path": str(local_path),
                "description": description,
                "position_in_material": position,
            }
        )
        print(f"  Media [{media_type}] saved: {source_url[:80]}")
        return True
