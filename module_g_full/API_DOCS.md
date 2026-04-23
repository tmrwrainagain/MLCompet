# API Documentation — Module G

**Base URL:** `http://localhost:8000`  
**Version:** 1.0.0  
**Swagger UI:** `http://localhost:8000/docs`

Все эндпоинты принимают и возвращают `application/json`. Ошибки валидации — `422 Unprocessable Entity` с описанием поля.

---

## Содержание

1. [GET /health](#get-health)
2. [POST /api/v1/assess](#post-apiv1assess)
3. [POST /api/v1/parallel](#post-apiv1parallel)
4. [POST /api/v1/sequential](#post-apiv1sequential)
5. [POST /api/v1/complexity](#post-apiv1complexity)
6. [POST /api/v1/time/material](#post-apiv1timematerial)
7. [POST /api/v1/time/subject](#post-apiv1timesubject)
8. [POST /api/v1/time/subjects](#post-apiv1timesubjects)
9. [POST /api/v1/trajectory](#post-apiv1trajectory)
10. [Коды ошибок](#коды-ошибок)

---

## GET /health

Проверка состояния сервера и загрузки предобученных моделей.

**Запрос:** параметров нет.

**Ответ `200 OK`:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "version": "1.0.0"
}
```

**Пример:**
```bash
curl http://localhost:8000/health
```

---

## POST /api/v1/assess

Оценить учебный материал по методическим требованиям.  
Использует LLM (Gemini → OpenAI fallback). Если материал уже оценён и есть в БД — возвращает кэшированный результат.

### Запрос

Обязательно одно из двух: `material_id` **или** `text`.

```json
{
  "material_id": 42
}
```

```json
{
  "text": "Тема: Введение в алгоритмы...\n\nАлгоритм — это...",
  "subject": "Информатика",
  "topic": "Алгоритмы сортировки",
  "lesson_type": "lecture"
}
```

| Поле | Тип | Обязательно | Описание |
|------|-----|-------------|---------|
| `material_id` | integer | если нет `text` | ID материала в БД |
| `text` | string (≥10 символов) | если нет `material_id` | Текст материала |
| `subject` | string | нет | Предмет/дисциплина |
| `topic` | string | нет | Тема |
| `lesson_type` | enum | нет | `lecture`, `seminar`, `practice`, `lab`, `self_study`, `test`, `other` |

### Ответ `200 OK`

```json
{
  "material_id": 42,
  "subject": "Информатика",
  "topic": "Алгоритмы сортировки",
  "overall_score": 7.5,
  "is_compliant": true,
  "moderation_status": "approved",
  "moderation_notes": "Материал хорошо структурирован, содержит практические задания",
  "strengths": ["Чёткая структура", "Практические задания"],
  "weaknesses": ["Отсутствуют иллюстрации"],
  "recommendations": "Добавить схемы и диаграммы для наглядности",
  "requirements": [
    {
      "category": "Структура и оформление",
      "requirement": "Наличие заголовков и структуры",
      "is_met": true,
      "score": 9.0,
      "notes": "Чёткая иерархия заголовков"
    },
    {
      "category": "Структура и оформление",
      "requirement": "Наличие введения",
      "is_met": true,
      "score": 8.5,
      "notes": "Введение задаёт контекст темы"
    }
  ],
  "cached": false
}
```

**Пример:**
```bash
curl -X POST http://localhost:8000/api/v1/assess \
  -H "Content-Type: application/json" \
  -d '{"text": "Введение в программирование. Алгоритм — это...", "subject": "Информатика"}'
```

---

## POST /api/v1/parallel

Разбить список материалов на группы для **параллельного изучения** (материалы из разных групп независимы и могут изучаться одновременно).

Использует предобученную модель `kmeans_parallel.joblib`.

### Запрос

```json
{
  "materials": [
    {"text": "Математический анализ. Пределы функций...", "title": "Лекция по матанализу", "subject": "Математика"},
    {"text": "Введение в программирование на Python...", "title": "Основы Python"},
    {"text": "История России XX века...", "title": "Лекция по истории"},
    {"text": "Интегральное исчисление...", "title": "Интегралы"}
  ]
}
```

| Поле | Тип | Описание |
|------|-----|---------|
| `materials` | array (≥2) | Список материалов |
| `materials[].text` | string (≥5 символов) | Текст материала |
| `materials[].title` | string | Заголовок (для отображения) |
| `materials[].subject` | string | Предмет |

### Ответ `200 OK`

```json
{
  "groups": [
    {
      "group_id": 0,
      "label": "Тематический поток А",
      "titles": ["Лекция по матанализу", "Интегралы"]
    },
    {
      "group_id": 1,
      "label": "Тематический поток Б",
      "titles": ["Основы Python"]
    },
    {
      "group_id": 2,
      "label": "Тематический поток В",
      "titles": ["Лекция по истории"]
    }
  ],
  "total_groups": 3,
  "recommendation": "Материалы разделены на 3 независимых групп — материалы из разных групп можно изучать одновременно."
}
```

**Пример:**
```bash
curl -X POST http://localhost:8000/api/v1/parallel \
  -H "Content-Type: application/json" \
  -d '{
    "materials": [
      {"text": "Математика. Производные...", "title": "Производные"},
      {"text": "Программирование. Циклы в Python...", "title": "Циклы Python"},
      {"text": "Физика. Механика...", "title": "Механика"}
    ]
  }'
```

---

## POST /api/v1/sequential

Определить **порядок последовательного изучения** материалов (от базового к сложному, по фазам таксономии Блума).

Использует предобученную модель `nearest_centroid_sequential.joblib`.

### Запрос

Формат аналогичен `/parallel`.

```json
{
  "materials": [
    {"text": "Объектно-ориентированное программирование. Наследование...", "title": "ООП"},
    {"text": "Переменные и типы данных в Python...", "title": "Основы Python"},
    {"text": "Создание проекта на Python с использованием классов...", "title": "Проект"}
  ]
}
```

### Ответ `200 OK`

```json
{
  "order": [
    {
      "position": 1,
      "phase": 0,
      "phase_name": "Ориентация (Знание)",
      "title": "Основы Python",
      "subject": null
    },
    {
      "position": 2,
      "phase": 1,
      "phase_name": "Понимание (Компрехенсия)",
      "title": "ООП",
      "subject": null
    },
    {
      "position": 3,
      "phase": 2,
      "phase_name": "Применение",
      "title": "Проект",
      "subject": null
    }
  ],
  "explanation": "Материалы упорядочены по фазам когнитивного освоения согласно таксономии Блума: от базового ознакомления к синтезу и критическому анализу."
}
```

---

## POST /api/v1/complexity

Определить **уровень сложности** учебного материала.

Использует предобученную модель `kmeans_complexity.joblib`.

### Запрос

```json
{
  "text": "Введение в основы программирования. Переменные — это...",
  "title": "Введение в программирование"
}
```

| Поле | Тип | Описание |
|------|-----|---------|
| `text` | string (≥10 символов) | Текст материала |
| `title` | string | Заголовок (опционально) |

### Ответ `200 OK`

```json
{
  "level": "Базовый",
  "score": 0.25,
  "word_count": 340,
  "avg_sentence_length": 12.4,
  "explanation": "Материал доступен без специальной подготовки. Небольшой объём, простые предложения."
}
```

Уровни: `Базовый` (score ≈ 0.25) | `Средний` (score ≈ 0.55) | `Продвинутый` (score ≈ 0.85)

---

## POST /api/v1/time/material

Оценить **время освоения одного учебного материала**.

### Формула

```
reading_min   = word_count / 150 × lesson_type_multiplier
complexity_oh = reading_min × (complexity_multiplier − 1)
practice_min  = 5–20 (если has_questions)
video_min     = 7 (если has_videos)
total         = reading_min + complexity_oh + practice_min + video_min
```

### Запрос (по ID из БД)

```json
{
  "material_id": 15
}
```

### Запрос (по тексту)

```json
{
  "text": "Тема: Алгоритмы сортировки...",
  "lesson_type": "lecture",
  "difficulty_label": "Средний",
  "has_questions": true,
  "has_videos": false
}
```

| Поле | Тип | Описание |
|------|-----|---------|
| `material_id` | integer | ID в БД (если нет `text`) |
| `text` | string | Текст (если нет `material_id`) |
| `lesson_type` | enum | Тип занятия (влияет на множитель) |
| `difficulty_label` | enum | `Базовый`, `Средний`, `Продвинутый` |
| `has_questions` | boolean | Есть ли контрольные вопросы |
| `has_videos` | boolean | Есть ли видеоматериалы |

### Ответ `200 OK`

```json
{
  "title": "Алгоритмы сортировки",
  "lesson_type": "Лекция",
  "difficulty_label": "Средний",
  "word_count": 1200,
  "estimated_minutes": 53,
  "breakdown": {
    "reading_minutes": 8.0,
    "complexity_overhead_minutes": 3.2,
    "practice_minutes": 10,
    "video_minutes": 0
  },
  "human_readable": "~53 мин"
}
```

**Множители типа занятия:**

| Тип | Множитель |
|-----|-----------|
| Лекция | 1.0 |
| Семинар | 1.2 |
| Практика | 1.5 |
| Лабораторная | 2.0 |
| Самостоятельная | 0.9 |
| Тест | 0.5 |

**Множители сложности:** Базовый × 1.0 | Средний × 1.4 | Продвинутый × 1.9

---

## POST /api/v1/time/subject

Оценить **суммарное время освоения всех материалов предмета** (данные из БД).

### Запрос

```json
{
  "subject": "Информатика"
}
```

### Ответ `200 OK`

```json
{
  "subject": "Информатика",
  "material_count": 12,
  "total_minutes": 340,
  "human_readable": "~5 ч 40 мин",
  "by_lesson_type": {
    "Лекция": 120,
    "Практическое занятие": 150,
    "Лабораторная работа": 70
  },
  "by_difficulty": {
    "Базовый": 80,
    "Средний": 160,
    "Продвинутый": 100
  }
}
```

---

## POST /api/v1/time/subjects

Оценить **время освоения набора предметов** с поддержкой параллельного режима.

При `parallel_study: true` итоговое время = максимальному времени одного предмета (дисциплины изучаются параллельно).

### Запрос

```json
{
  "subjects": ["Информатика", "Математика", "Физика"],
  "parallel_study": false
}
```

### Ответ `200 OK`

```json
{
  "subjects": [
    {"subject": "Информатика", "total_minutes": 340, "material_count": 12},
    {"subject": "Математика",  "total_minutes": 280, "material_count": 9},
    {"subject": "Физика",      "total_minutes": 420, "material_count": 14}
  ],
  "grand_total_minutes": 1040,
  "parallel_total_minutes": 420,
  "human_readable_sequential": "~17 ч 20 мин",
  "human_readable_parallel": "~7 ч 0 мин (при параллельном изучении)"
}
```

---

## POST /api/v1/trajectory

Построить **индивидуальную траекторию изучения** материалов по заданным параметрам.

Алгоритм:
1. Загрузить все материалы предмета из БД.
2. Фильтровать по `difficulty_level` (≤ заданного уровня).
3. Фильтровать по `target_topics` (нечёткое совпадение с темами).
4. Исключить `exclude_material_ids`.
5. Сортировать по `sequential_cluster`, затем по `complexity_cluster`.
6. Оценить время каждого материала.
7. Жадно распределить по неделям с учётом `available_hours_per_week`.

### Запрос

```json
{
  "subject": "Информатика",
  "difficulty_level": "Средний",
  "available_hours_per_week": 6.0,
  "learning_style": "sequential",
  "exclude_material_ids": [3, 7],
  "target_topics": ["Алгоритмы", "Структуры данных"]
}
```

| Поле | Тип | Описание |
|------|-----|---------|
| `subject` | string | Предмет для выборки из БД |
| `difficulty_level` | enum | `Базовый`, `Средний`, `Продвинутый` |
| `available_hours_per_week` | float (>0, ≤80) | Доступное время в неделю |
| `learning_style` | enum | `sequential` или `mixed` |
| `exclude_material_ids` | array of int | ID материалов для исключения |
| `target_topics` | array of string | Целевые темы (пустой = все) |

### Ответ `200 OK`

```json
{
  "trajectory": [
    {
      "step": 1,
      "material_id": 2,
      "topic": "Введение в алгоритмы",
      "difficulty_label": "Базовый",
      "estimated_minutes": 35,
      "week": 1,
      "rationale": "Стартовый материал фазы 0 — нет предшественников"
    },
    {
      "step": 2,
      "material_id": 5,
      "topic": "Сортировка пузырьком",
      "difficulty_label": "Базовый",
      "estimated_minutes": 28,
      "week": 1,
      "rationale": "Логически следует за предыдущим материалом (фаза 0)"
    },
    {
      "step": 3,
      "material_id": 8,
      "topic": "Бинарный поиск",
      "difficulty_label": "Средний",
      "estimated_minutes": 47,
      "week": 2,
      "rationale": "Стартовый материал фазы 1 — нет предшественников"
    }
  ],
  "total_steps": 3,
  "total_estimated_minutes": 110,
  "total_weeks": 2,
  "weekly_plan": {
    "1": [2, 5],
    "2": [8]
  },
  "parameters_used": {
    "subject": "Информатика",
    "difficulty_level": "Средний",
    "available_hours_per_week": 6.0,
    "learning_style": "sequential"
  }
}
```

**Пример:**
```bash
curl -X POST http://localhost:8000/api/v1/trajectory \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "Информатика",
    "difficulty_level": "Средний",
    "available_hours_per_week": 5,
    "learning_style": "sequential"
  }'
```

---

## Коды ошибок

| Код | Причина |
|-----|---------|
| `200` | Успех |
| `400` | Некорректный запрос (например, пустой текст) |
| `404` | Материал или предмет не найден в БД |
| `422` | Ошибка валидации Pydantic (неверный тип или отсутствует обязательное поле) |
| `500` | Внутренняя ошибка сервера |

**Пример ошибки 422:**
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body"],
      "msg": "Укажите material_id или text",
      "input": {}
    }
  ]
}
```

**Пример ошибки 404:**
```json
{
  "detail": "Материалы по предмету 'НесуществующийПредмет' не найдены"
}
```
