"""
Module B configuration — reuses Module A database settings.
"""

import importlib.util
import os
import sys
from pathlib import Path

# ── Module A path (for shared DB config) ────────────────────────────────────
MODULE_A_DIR = Path(__file__).parent.parent / "module_A"
MODULE_A_CONFIG = MODULE_A_DIR / "config.py"
_spec = importlib.util.spec_from_file_location("module_a_shared_config", MODULE_A_CONFIG)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Module A config from {MODULE_A_CONFIG}")
_module_a_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module_a_config)
sys.path.insert(0, str(MODULE_A_DIR))

DATABASE_URL = _module_a_config.DATABASE_URL
METHODOLOGY_REQUIREMENTS = _module_a_config.METHODOLOGY_REQUIREMENTS
LESSON_TYPES = _module_a_config.LESSON_TYPES
LESSON_TYPE_LABELS = _module_a_config.LESSON_TYPE_LABELS
GEMINI_API_KEY = _module_a_config.GEMINI_API_KEY
MODEL_FAST = _module_a_config.MODEL_FAST

# ── Module B paths ───────────────────────────────────────────────────────────
PROJECT_RUNTIME_DIR = Path(
    os.environ.get("PROJECT_RUNTIME_DIR", str(Path(__file__).resolve().parent.parent))
)
BASE_DIR = PROJECT_RUNTIME_DIR / "module_B"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dashboard access levels ──────────────────────────────────────────────────
# Stored as plaintext here for simplicity; use bcrypt hashes in production.
USERS = {
    "admin": {
        "password": os.environ.get("DASHBOARD_ADMIN_PASS", "admin123"),
        "role":     "admin",
        "name":     "Администратор",
    },
    "teacher": {
        "password": os.environ.get("DASHBOARD_TEACHER_PASS", "teacher123"),
        "role":     "teacher",
        "name":     "Преподаватель",
    },
    "student": {
        "password": os.environ.get("DASHBOARD_STUDENT_PASS", "student123"),
        "role":     "student",
        "name":     "Студент",
    },
}

# Pages accessible to each role
ROLE_PAGES = {
    "admin":   ["overview", "analytics", "clustering", "requirements", "data"],
    "teacher": ["overview", "analytics", "clustering", "requirements"],
    "student": ["overview"],
}

# ── Clustering settings ──────────────────────────────────────────────────────
CLUSTER_SETTINGS = {
    "parallel":   {"n_clusters": 4, "method": "kmeans"},
    "sequential": {"n_clusters": 4, "method": "hierarchical"},
    "complexity": {"n_clusters": 3, "method": "kmeans",
                   "labels": {0: "Базовый", 1: "Средний", 2: "Продвинутый"}},
}

# Refresh interval for live dashboard data (seconds)
DASHBOARD_REFRESH_SECONDS = 60
