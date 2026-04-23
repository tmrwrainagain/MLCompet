# Модуль Г — Интеграция и взаимодействие

Агент интеграции, предоставляющий доступ к предобученным моделям кластеризации и аналитики через REST API и Telegram-бота.

---

## Содержание

1. [Архитектура](#архитектура)
2. [Запуск](#запуск)
3. [Telegram-бот — интерфейс пользователя](#telegram-бот--интерфейс-пользователя)
4. [API — эндпоинты](#api--эндпоинты)
5. [Предобученные модели](#предобученные-модели)
6. [Переменные окружения](#переменные-окружения)

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                     PostgreSQL                          │
│  materials / methodology_compliance / material_features │
└──────────────────────┬──────────────────────────────────┘
                       │ read-only
                       ▼
┌─────────────────────────────────────────────────────────┐
│          FastAPI — Module G (порт 8000)                 │
│                                                         │
│  [Startup] predictor.load()                             │
│    ├── tfidf_vectorizer.joblib                          │
│    ├── kmeans_parallel.joblib         (4 кластера)      │
│    ├── nearest_centroid_sequential.joblib (4 фазы)     │
│    ├── kmeans_complexity.joblib       (3 уровня)        │
│    └── complexity_label_map.json                        │
│                                                         │
│  POST /api/v1/assess       ← LLM (Gemini → OpenAI)     │
│  POST /api/v1/parallel     ← kmeans_parallel.predict()  │
│  POST /api/v1/sequential   ← nearest_centroid.predict() │
│  POST /api/v1/complexity   ← kmeans_complexity.predict()│
│  POST /api/v1/time/material  ← формула по параметрам   │
│  POST /api/v1/time/subject   ← запрос в БД + формула   │
│  POST /api/v1/time/subjects  ← запрос в БД + формула   │
│  POST /api/v1/trajectory   ← БД + предобученные модели │
│  GET  /health                                           │
│  GET  /docs                ← Swagger UI                 │
└──────────────────────┬──────────────────────────────────┘
                       ▲ httpx
┌─────────────────────────────────────────────────────────┐
│         Telegram Bot (python-telegram-bot v20+)         │
│                                                         │
│  /start  /help  /assess  /parallel  /sequential         │
│  /complexity  /time  /trajectory  /cancel               │
│                                                         │
│  ConversationHandler-ы + InlineKeyboardMarkup           │
└─────────────────────────────────────────────────────────┘
                       ▲ Telegram API
                   [Пользователь]
```

**Ключевой принцип:** API только **загружает** предобученные joblib-модели при старте (`predictor.load()`). Метод `.fit()` нигде не вызывается. Для обучения — отдельный скрипт `train_models.py`.

---

## Запуск

### 1. Установить зависимости

```bash
pip install -r requirements.txt
```

### 2. Заполнить `.env`

```env
GEMINI_API_KEY=ваш_ключ
OPENAI_API_KEY=ваш_ключ
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=educational_materials
POSTGRES_USER=postgres
POSTGRES_PASSWORD=ваш_пароль
TELEGRAM_BOT_TOKEN=токен_от_BotFather
```

> Токен для Telegram-бота получить у [@BotFather](https://t.me/BotFather): `/newbot` → следовать инструкциям.

### 3. Обучить и сохранить модели (один раз)

```bash
python train_models.py
```

Скрипт подключается к БД, обучает три sklearn-модели и сохраняет их в `module_G/saved_models/`. Если БД недоступна или пустая — использует встроенные синтетические данные.

### 4. Запустить

```bash
# Оба агента сразу:
python run_module_g.py

# Только API:
python agent_api.py

# Только бот:
python agent_bot.py
```

После запуска API документация доступна по адресу `http://localhost:8000/docs`.

---

## Telegram-бот — интерфейс пользователя

### Команды

| Команда | Описание |
|---------|---------|
| `/start` | Главное меню с кнопками |
| `/help` | Справка по командам |
| `/assess` | Оценить материал по методическим критериям |
| `/parallel` | Определить группы параллельного изучения |
| `/sequential` | Определить порядок последовательного изучения |
| `/complexity` | Определить уровень сложности освоения |
| `/time` | Оценить время освоения (материал / предмет / набор) |
| `/trajectory` | Построить индивидуальную траекторию |
| `/cancel` | Отменить текущий диалог |

### Схема диалогов

#### `/assess` — оценка материала
```
/assess
  ├── [По ID материала] → введите ID → результат из БД или LLM
  └── [Ввести текст]   → вставьте текст → LLM-оценка
```

#### `/parallel` и `/sequential` — кластеризация
```
/parallel или /sequential
  └── Отправьте тексты, разделённые ---
      Материал 1
      ---
      Материал 2
      ---
      Материал 3
      → Результат группировки
```

#### `/complexity` — уровень сложности
```
/complexity
  └── Отправьте текст → Базовый / Средний / Продвинутый
```

#### `/time` — время освоения
```
/time
  ├── [Один материал]   → текст → оценка в минутах
  ├── [Предмет]         → название → суммарное время по БД
  └── [Набор предметов] → список через запятую
                          → [Последовательно] / [Параллельно]
```

#### `/trajectory` — индивидуальная траектория
```
/trajectory
  → Введите предмет
  → Выберите уровень: [Базовый] [Средний] [Продвинутый]
  → Часов в неделю: 5
  → Стиль: [Последовательное] [Смешанное]
  → Темы (или /skip для всех)
  → Пошаговый план по неделям
```

### Пример вывода `/trajectory`

```
🗺 Индивидуальная траектория — Информатика
Уровень: Средний | Стиль: sequential
Материалов: 6 | Время: 4 ч 20 мин | Недель: 3

Шаг 1 (неделя 1)
  Введение в алгоритмы
  Сложность: Базовый | Время: 35 мин
  Стартовый материал фазы 0 — нет предшественников

Шаг 2 (неделя 1)
  Условные операторы
  Сложность: Базовый | Время: 28 мин
  Логически следует за предыдущим (фаза 0)
...
```

---

## API — эндпоинты

Подробная документация с форматами запросов/ответов и примерами — в [API_DOCS.md](API_DOCS.md).

| Метод | Эндпоинт | Назначение |
|-------|---------|-----------|
| GET | `/health` | Статус сервера и загрузки моделей |
| POST | `/api/v1/assess` | Оценка по методическим требованиям |
| POST | `/api/v1/parallel` | Группы параллельного изучения |
| POST | `/api/v1/sequential` | Порядок последовательного изучения |
| POST | `/api/v1/complexity` | Уровень сложности |
| POST | `/api/v1/time/material` | Время освоения материала |
| POST | `/api/v1/time/subject` | Время освоения предмета |
| POST | `/api/v1/time/subjects` | Время освоения набора предметов |
| POST | `/api/v1/trajectory` | Индивидуальная траектория |

Интерактивная документация (Swagger UI): `http://localhost:8000/docs`

---

## Предобученные модели

Скрипт `train_models.py` обучает и сохраняет в `module_G/saved_models/`:

| Файл | Алгоритм | Назначение |
|------|---------|-----------|
| `tfidf_vectorizer.joblib` | TF-IDF (sklearn) | Векторизация текстов |
| `kmeans_parallel.joblib` | KMeans (4 кластера) | Группы параллельного изучения |
| `nearest_centroid_sequential.joblib` | NearestCentroid | Прокси для последовательных фаз |
| `kmeans_complexity.joblib` | KMeans (3 кластера) | Уровень сложности |
| `scaler_numeric.joblib` | StandardScaler | Нормализация числовых признаков |
| `complexity_label_map.json` | JSON | Маппинг кластер → Базовый/Средний/Продвинутый |

**API при загрузке вызывает только `joblib.load()` и `.predict()` — никакого переобучения.**

> Если модели отсутствуют (не запускался `train_models.py`), API сообщит об этом в `/health` (`models_loaded: false`) и будет использовать простые эвристики.

---

## Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|-----------|---------|----------------------|
| `GEMINI_API_KEY` | API-ключ Google Gemini | — |
| `OPENAI_API_KEY` | API-ключ OpenAI (fallback) | — |
| `POSTGRES_HOST` | Хост PostgreSQL | `localhost` |
| `POSTGRES_PORT` | Порт PostgreSQL | `5432` |
| `POSTGRES_DB` | Имя базы данных | `educational_materials` |
| `POSTGRES_USER` | Пользователь БД | `postgres` |
| `POSTGRES_PASSWORD` | Пароль БД | — |
| `TELEGRAM_BOT_TOKEN` | Токен Telegram-бота | — |
| `API_HOST` | Хост FastAPI | `0.0.0.0` |
| `API_PORT` | Порт FastAPI | `8000` |
| `API_BASE_URL` | Базовый URL для бота | `http://127.0.0.1:8000/api/v1` |
