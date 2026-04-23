"""Shared configuration for Module V notebooks and CLI/EXE agents."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def _detect_runtime_root() -> Path:
    cwd = Path.cwd()
    if (cwd / ".env").exists() or (cwd / "notebooks").exists():
        return cwd

    if getattr(sys, "frozen", False):
        return cwd

    return Path(__file__).resolve().parent.parent


ROOT = _detect_runtime_root()
load_dotenv(ROOT / ".env", override=False)

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
LOGS_DIR    = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
DATA_DIR    = ROOT / "data"

for _d in [MODELS_DIR, LOGS_DIR, REPORTS_DIR, DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ─────────────────────────────────────────────────────────────────
POSTGRES_HOST     = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.environ.get("POSTGRES_DB", "educational_materials")
POSTGRES_USER     = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── ML targets (cluster column names from Module B) ───────────────────────────
TARGET_PARALLEL   = "parallel_cluster"    # K-Means, 4 classes
TARGET_SEQUENTIAL = "sequential_cluster"  # Agglomerative, 4 classes
TARGET_COMPLEXITY = "complexity_cluster"  # K-Means, 3 classes (0=Basic, 1=Mid, 2=Advanced)

# ── Model versioning ─────────────────────────────────────────────────────────
MODEL_VERSION_FILE = MODELS_DIR / "current_version.txt"

# ── Time estimation constants ─────────────────────────────────────────────────
# Average reading speed (words per minute) for educational text
WORDS_PER_MINUTE = 200
# Complexity multipliers applied to base reading time
COMPLEXITY_MULTIPLIERS = {0: 1.0, 1: 1.4, 2: 2.0}  # Basic / Mid / Advanced
# Extra time (minutes) per media element (image/video/question)
MEDIA_TIME_MIN  = 2
VIDEO_TIME_MIN  = 5
QUESTION_TIME_MIN = 3
