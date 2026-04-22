import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional helper for raw environments
    load_dotenv = None


def _autoload_env():
    if load_dotenv is None:
        return

    runtime_dir = Path(
        os.environ.get("PROJECT_RUNTIME_DIR", str(Path(__file__).resolve().parent.parent))
    )
    candidates = [
        runtime_dir / "module_A" / ".env",
        runtime_dir / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_file in candidates:
        if env_file.exists():
            load_dotenv(env_file, override=False)


_autoload_env()

# ---------------------------------------------------------------------------
# API Configuration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Gemini 2.5 Flash as specified in the task requirements
MODEL_FAST = "gemini-2.5-flash"
MODEL_PRO  = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# PostgreSQL configuration
# ---------------------------------------------------------------------------
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "educational_materials")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_RUNTIME_DIR = Path(
    os.environ.get("PROJECT_RUNTIME_DIR", str(Path(__file__).resolve().parent.parent))
)
BASE_DIR = PROJECT_RUNTIME_DIR / "module_A"
MATERIALS_DIR = BASE_DIR / "downloaded_materials"
ANALYSIS_OUTPUT_DIR = BASE_DIR / "analysis_output"
REPORTS_DIR = BASE_DIR / "reports"
TEST_FILES_DIR = PROJECT_RUNTIME_DIR / "test_files"
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

for _d in [MATERIALS_DIR, ANALYSIS_OUTPUT_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
TEST_FILES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Lesson types (типы занятий)
# ---------------------------------------------------------------------------
LESSON_TYPES = [
    "lecture",       # Лекция
    "seminar",       # Семинар
    "practice",      # Практическое занятие
    "lab",           # Лабораторная работа
    "self_study",    # Самостоятельная работа
    "test",          # Контрольная / тест
    "other",         # Иное
]

LESSON_TYPE_LABELS = {
    "lecture":    "Лекция",
    "seminar":    "Семинар",
    "practice":   "Практическое занятие",
    "lab":        "Лабораторная работа",
    "self_study": "Самостоятельная работа",
    "test":       "Контрольная / тест",
    "other":      "Иное",
}

# ---------------------------------------------------------------------------
# Methodological guidelines text (used in moderation prompt)
# ---------------------------------------------------------------------------
METHODOLOGICAL_GUIDELINES = """
Методические рекомендации к учебным материалам:
1. Структура: наличие введения, основной части и заключения
2. Целеполагание: явное указание учебных целей и задач
3. Соответствие аудитории: материал адаптирован под целевую аудиторию
4. Безопасность контента: отсутствие вредоносного или неприемлемого контента
5. Наглядность: наличие иллюстраций, примеров, схем
6. Последовательность: логически выстроенное изложение
7. Язык: доступный и понятный язык изложения
8. Контроль знаний: наличие вопросов, задач или упражнений
9. Актуальность: использование актуальной информации
10. Источники: наличие ссылок на источники и литературу
"""

# ---------------------------------------------------------------------------
# Methodology requirements (per category, for per-requirement compliance)
# ---------------------------------------------------------------------------
METHODOLOGY_REQUIREMENTS = [
    # ── Структура и оформление ──────────────────────────────────────────
    {
        "category":    "Структура и оформление",
        "requirement": "Наличие заголовков и структуры",
        "description": "Материал имеет чёткую структуру с заголовками и разделами",
    },
    {
        "category":    "Структура и оформление",
        "requirement": "Логическая последовательность изложения",
        "description": "Материал изложен в логически последовательном порядке",
    },
    {
        "category":    "Структура и оформление",
        "requirement": "Наличие введения",
        "description": "Материал начинается с введения, задающего контекст",
    },
    {
        "category":    "Структура и оформление",
        "requirement": "Наличие заключения",
        "description": "Материал завершается выводами или заключением",
    },
    {
        "category":    "Структура и оформление",
        "requirement": "Список литературы / источников",
        "description": "Присутствует список использованной литературы или ссылок",
    },
    # ── Содержательная часть ────────────────────────────────────────────
    {
        "category":    "Содержательная часть",
        "requirement": "Соответствие заявленной теме",
        "description": "Содержание соответствует заявленной теме и дисциплине",
    },
    {
        "category":    "Содержательная часть",
        "requirement": "Полнота раскрытия темы",
        "description": "Тема раскрыта достаточно полно и систематически",
    },
    {
        "category":    "Содержательная часть",
        "requirement": "Актуальность информации",
        "description": "Представленная информация является современной и актуальной",
    },
    {
        "category":    "Содержательная часть",
        "requirement": "Научная корректность",
        "description": "Информация научно корректна, достоверна и не содержит ошибок",
    },
    # ── Методическая составляющая ───────────────────────────────────────
    {
        "category":    "Методическая составляющая",
        "requirement": "Явно сформулированные учебные цели",
        "description": "Материал содержит чётко сформулированные учебные цели",
    },
    {
        "category":    "Методическая составляющая",
        "requirement": "Дидактическая ценность",
        "description": "Материал обладает выраженной образовательной ценностью",
    },
    {
        "category":    "Методическая составляющая",
        "requirement": "Соответствие уровню целевой аудитории",
        "description": "Сложность и стиль изложения соответствуют уровню аудитории",
    },
    {
        "category":    "Методическая составляющая",
        "requirement": "Доступность языка изложения",
        "description": "Язык понятен, термины объяснены",
    },
    # ── Практическая часть ──────────────────────────────────────────────
    {
        "category":    "Практическая часть",
        "requirement": "Наличие практических заданий или упражнений",
        "description": "Материал содержит задания для самопроверки или практики",
    },
    {
        "category":    "Практическая часть",
        "requirement": "Наличие примеров и иллюстраций решений",
        "description": "Присутствуют разобранные примеры с решениями типовых задач",
    },
    # ── Медиаконтент ────────────────────────────────────────────────────
    {
        "category":    "Медиаконтент",
        "requirement": "Наличие иллюстраций, схем или диаграмм",
        "description": "Материал содержит наглядные иллюстративные элементы",
    },
    {
        "category":    "Медиаконтент",
        "requirement": "Качество и информативность медиаконтента",
        "description": "Медиаматериалы качественны и несут смысловую нагрузку",
    },
    {
        "category":    "Медиаконтент",
        "requirement": "Соответствие медиаконтента тексту",
        "description": "Иллюстрации, видео и аудио соответствуют тексту материала",
    },
    # ── Актуальность и точность ─────────────────────────────────────────
    {
        "category":    "Актуальность и точность",
        "requirement": "Отсутствие устаревших данных",
        "description": "Материал не содержит устаревших фактов и цифр",
    },
    {
        "category":    "Актуальность и точность",
        "requirement": "Корректность терминологии",
        "description": "Терминология соответствует стандартам дисциплины",
    },
    {
        "category":    "Актуальность и точность",
        "requirement": "Безопасность контента",
        "description": "Материал не содержит вредоносного или неприемлемого контента",
    },
]

# ---------------------------------------------------------------------------
# File type sets
# ---------------------------------------------------------------------------
SUPPORTED_TEXT_TYPES = {
    ".txt", ".text", ".log", ".md", ".rst", ".rtf",
    ".html", ".htm", ".xhtml", ".xml",
    ".json", ".jsonl", ".yaml", ".yml", ".ini", ".cfg", ".toml",
    ".pdf",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv", ".tsv",
    ".odt", ".ods", ".odp", ".epub",
}
SUPPORTED_IMAGE_TYPES = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif",
    ".svg", ".ico",
}
SUPPORTED_VIDEO_TYPES = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
}
SUPPORTED_AUDIO_TYPES = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma",
    ".aiff", ".aif", ".opus", ".amr", ".oga",
}

# ---------------------------------------------------------------------------
# Processing limits
# ---------------------------------------------------------------------------
MAX_MEDIA_PER_PAGE       = 5
VIDEO_FRAME_INTERVAL_SEC = 10
MAX_VIDEO_FRAMES         = 5
MAX_AUDIO_SIZE_MB        = 20
