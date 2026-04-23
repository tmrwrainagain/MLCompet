"""Module G configuration."""
import os
from pathlib import Path

# ── Path to pre-trained models from model_V_full ─────────────────────────────
MODEL_V_FULL_DIR = Path(os.getenv(
    "MODEL_V_FULL_DIR",
    str(Path(__file__).parent.parent.parent / "model_V_full"),
))

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

# ── Database ─────────────────────────────────────────────────────────────────
POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB       = os.getenv("POSTGRES_DB", "educational_materials")
POSTGRES_USER     = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# ── LLM ──────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_FAST     = os.getenv("LLM_MODEL", "gemini-2.5-flash")

# ── API server ────────────────────────────────────────────────────────────────
API_HOST     = os.getenv("API_HOST", "0.0.0.0")
API_PORT     = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://127.0.0.1:{API_PORT}/api/v1")

# ── Telegram ──────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODULE_G_DIR = Path(__file__).parent
MODELS_DIR   = MODULE_G_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Time estimation ───────────────────────────────────────────────────────────
READING_SPEED_WPM = 150  # слов/минуту для учебного текста

LESSON_TYPE_MULTIPLIERS = {
    "lecture":    1.0,
    "seminar":    1.2,
    "practice":   1.5,
    "lab":        2.0,
    "self_study": 0.9,
    "test":       0.5,
    "other":      1.0,
}

COMPLEXITY_MULTIPLIERS = {
    "Базовый":      1.0,
    "Средний":      1.4,
    "Продвинутый":  1.9,
}

PRACTICE_OVERHEAD = {
    "Базовый":      5,
    "Средний":      10,
    "Продвинутый":  20,
}

LESSON_TYPE_LABELS = {
    "lecture":    "Лекция",
    "seminar":    "Семинар",
    "practice":   "Практическое занятие",
    "lab":        "Лабораторная работа",
    "self_study": "Самостоятельная работа",
    "test":       "Контрольная / тест",
    "other":      "Иное",
}

# ── Methodology ───────────────────────────────────────────────────────────────
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

METHODOLOGY_REQUIREMENTS = [
    {"category": "Структура и оформление",    "requirement": "Наличие заголовков и структуры",          "description": "Материал имеет чёткую структуру с заголовками и разделами"},
    {"category": "Структура и оформление",    "requirement": "Логическая последовательность изложения", "description": "Материал изложен в логически последовательном порядке"},
    {"category": "Структура и оформление",    "requirement": "Наличие введения",                        "description": "Материал начинается с введения, задающего контекст"},
    {"category": "Структура и оформление",    "requirement": "Наличие заключения",                      "description": "Материал завершается выводами или заключением"},
    {"category": "Структура и оформление",    "requirement": "Список литературы / источников",          "description": "Присутствует список использованной литературы или ссылок"},
    {"category": "Содержательная часть",      "requirement": "Соответствие заявленной теме",            "description": "Содержание соответствует заявленной теме и дисциплине"},
    {"category": "Содержательная часть",      "requirement": "Полнота раскрытия темы",                  "description": "Тема раскрыта достаточно полно и систематически"},
    {"category": "Содержательная часть",      "requirement": "Актуальность информации",                 "description": "Представленная информация является современной и актуальной"},
    {"category": "Содержательная часть",      "requirement": "Научная корректность",                    "description": "Информация научно корректна, достоверна и не содержит ошибок"},
    {"category": "Методическая составляющая", "requirement": "Явно сформулированные учебные цели",      "description": "Материал содержит чётко сформулированные учебные цели"},
    {"category": "Методическая составляющая", "requirement": "Дидактическая ценность",                 "description": "Материал обладает выраженной образовательной ценностью"},
    {"category": "Методическая составляющая", "requirement": "Соответствие уровню целевой аудитории",  "description": "Сложность и стиль изложения соответствуют уровню аудитории"},
    {"category": "Методическая составляющая", "requirement": "Доступность языка изложения",            "description": "Язык понятен, термины объяснены"},
    {"category": "Практическая часть",        "requirement": "Наличие практических заданий или упражнений", "description": "Материал содержит задания для самопроверки или практики"},
    {"category": "Практическая часть",        "requirement": "Наличие примеров и иллюстраций решений", "description": "Присутствуют разобранные примеры с решениями типовых задач"},
    {"category": "Медиаконтент",              "requirement": "Наличие иллюстраций, схем или диаграмм", "description": "Материал содержит наглядные иллюстративные элементы"},
    {"category": "Медиаконтент",              "requirement": "Качество и информативность медиаконтента","description": "Медиаматериалы качественны и несут смысловую нагрузку"},
    {"category": "Медиаконтент",              "requirement": "Соответствие медиаконтента тексту",       "description": "Иллюстрации, видео и аудио соответствуют тексту материала"},
    {"category": "Актуальность и точность",   "requirement": "Отсутствие устаревших данных",            "description": "Материал не содержит устаревших фактов и цифр"},
    {"category": "Актуальность и точность",   "requirement": "Корректность терминологии",               "description": "Терминология соответствует стандартам дисциплины"},
    {"category": "Актуальность и точность",   "requirement": "Безопасность контента",                   "description": "Материал не содержит вредоносного или неприемлемого контента"},
]
