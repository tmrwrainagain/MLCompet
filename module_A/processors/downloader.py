"""
Downloads files from URLs, detects formats, extracts embedded media links.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import mimetypes
import requests
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MATERIALS_DIR,
    SUPPORTED_AUDIO_TYPES,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_TEXT_TYPES,
    SUPPORTED_VIDEO_TYPES,
)


def _detect_extension(url: str, content_type: str) -> str:
    parsed_ext = Path(urlparse(url).path).suffix.lower()
    all_types = SUPPORTED_TEXT_TYPES | SUPPORTED_IMAGE_TYPES | SUPPORTED_VIDEO_TYPES | SUPPORTED_AUDIO_TYPES
    if parsed_ext in all_types:
        return parsed_ext

    ct = content_type.split(";")[0].strip()
    ext = mimetypes.guess_extension(ct) or ""
    if ext in (".jpe", ".jpeg"):
        ext = ".jpg"
    if ext:
        return ext

    return ".html"


def _safe_name(url: str) -> str:
    name = urlparse(url).netloc + urlparse(url).path
    name = re.sub(r"[^\w\-_.]", "_", name)
    return name[:100]


def download_file(
    url: str, session: Optional[requests.Session] = None
) -> Tuple[Optional[Path], str, str]:
    """
    Download a URL to local storage.
    Returns (local_path, content_type, error_message).
    """
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )

    try:
        resp = session.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        ext = _detect_extension(url, ct)
        local_path = MATERIALS_DIR / (_safe_name(url) + ext)

        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)

        return local_path, ct, ""
    except Exception as exc:
        return None, "", str(exc)


def extract_media_links(html_content: str, base_url: str) -> List[Tuple[str, str]]:
    """
    Return list of (media_type, absolute_url) found in HTML.
    media_type is one of: 'image', 'video', 'audio', 'document'
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []

    soup = BeautifulSoup(html_content, "lxml")
    links: List[Tuple[str, str]] = []

    for img in soup.find_all("img", src=True):
        links.append(("image", urljoin(base_url, img["src"])))

    for tag in soup.find_all(["video", "source"]):
        src = tag.get("src") or tag.get("data-src")
        if src:
            links.append(("video", urljoin(base_url, src)))

    for tag in soup.find_all(["audio"]):
        src = tag.get("src")
        if src:
            links.append(("audio", urljoin(base_url, src)))

    for a in soup.find_all("a", href=True):
        ext = Path(urlparse(a["href"]).path).suffix.lower()
        if ext in SUPPORTED_TEXT_TYPES - {".html", ".htm"}:
            links.append(("document", urljoin(base_url, a["href"])))

    return links
