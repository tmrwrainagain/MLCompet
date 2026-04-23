# Модуль В — Моделирование, прогнозирование и рекомендации

**Аналитическая система учебных материалов**  
Использует данные из PostgreSQL, загруженные и размеченные Модулями А и Б.

---

## Содержание

1. [Быстрый старт](#быстрый-старт)
2. [Структура проекта](#структура-проекта)
3. [Обучение моделей (3.1)](#1-обучение-моделей)
4. [Непрерывное обучение (3.2)](#2-непрерывное-обучение)
5. [Оценка времени (3.3)](#3-оценка-времени-освоения)
6. [Рекомендации и траектории (3.4)](#4-индивидуальные-траектории)
7. [Отчёт (3.5)](#5-итоговый-отчёт)
8. [HTML-экспорт ноутбуков](#html-экспорт-ноутбуков)
9. [EXE-агенты и сборка](#exe-агенты-и-сборка)
10. [Сверка по критериям](#сверка-по-критериям)
11. [Пути к файлам и форматы](#пути-к-файлам-и-форматы)
12. [Зависимости](#зависимости)

---

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Заполнить .env (уже содержит ключи)

# 3. Запустить Jupyter
jupyter notebook notebooks/

# 4. Выполнять ноутбуки по порядку:
#    3.1 → 3.2 → 3.3 → 3.4 → 3.5
```

После выполнения ноутбуков рекомендуется сразу экспортировать их в HTML, чтобы результаты, графики и текстовые выводы можно было смотреть без открытия `.ipynb`. Актуальные HTML-копии последних запусков хранятся в корне проекта.

Запуск `3.5_report.ipynb` работает как единая точка обновления: он автоматически прогоняет `3.1 → 3.4`, обновляет `reports/` и перезаписывает HTML-файлы в корне проекта.

**Предварительные условия:**
- PostgreSQL запущен, таблица `materials` заполнена (Модуль А)
- Кластерная разметка из Модуля Б доступна либо в таблице `cluster_labels`, либо прямо в полях `materials.parallel_cluster`, `materials.sequential_cluster`, `materials.complexity_cluster`, `materials.difficulty_label`

---

## Структура проекта

```
model_V_full/
│
├── notebooks/
│   ├── 3.1_model_training.ipynb       ← Обучение классификаторов
│   ├── 3.2_continuous_learning.ipynb  ← Непрерывное обучение + дрейф
│   ├── 3.3_time_estimation.ipynb      ← Оценка времени освоения
│   ├── 3.4_recommendations.ipynb      ← Индивидуальные траектории
│   └── 3.5_report.ipynb               ← Итоговый отчёт (этот файл)
│
├── 3.1_model_training.html            ← HTML-экспорт последнего запуска 3.1
├── 3.2_continuous_learning.html       ← HTML-экспорт последнего запуска 3.2
├── 3.3_time_estimation.html           ← HTML-экспорт последнего запуска 3.3
├── 3.4_recommendations.html           ← HTML-экспорт последнего запуска 3.4
├── 3.5_report.html                    ← HTML-экспорт последнего запуска 3.5
│
├── models/
│   ├── current_version.txt            ← Указатель на актуальную версию
│   └── v_YYYYMMDD_HHMMSS/            ← Папка версии (создаётся автоматически)
│       ├── parallel_cluster_best.joblib
│       ├── sequential_cluster_best.joblib
│       ├── complexity_cluster_best.joblib
│       ├── tfidf_vectorizer.joblib
│       ├── scaler.joblib
│       └── meta.json
│
├── logs/
│   ├── training_log.jsonl             ← История всех обучений
│   └── known_material_ids.json        ← ID материалов последнего обучения
├── PostgreSQL: public.module_v_model_versions ← Лог версий моделей в БД
│
├── reports/                           ← Все выходные файлы
│   ├── module_v_report.html           ← Итоговый HTML-отчёт (3.5)
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   ├── model_metrics_summary.csv
│   ├── drift_report.html
│   ├── drift_distributions.png
│   ├── model_update_comparison.png
│   ├── time_estimation_summary.csv
│   ├── time_by_subject.png
│   ├── gantt_sequential.html
│   ├── gantt_parallel.html
│   ├── trajectory.html
│   ├── trajectory_graph.html
│   ├── trajectory_graph.png
│   ├── trajectory_plan.csv
│   └── trajectory_comparison.png
│
├── src/
│   ├── config.py                      ← Пути, константы, переменные окружения
│   ├── db.py                          ← Загрузка данных из PostgreSQL
│   ├── features.py                    ← TF-IDF + числовые признаки
│   ├── model_registry.py              ← Логирование версий моделей в БД
│   ├── notebook_runner.py             ← CLI-раннер notebook-агентов
│   └── trajectory_graph.py            ← Построение ГРАФА траектории
│
├── agent_3_1.py                       ← Python-entrypoint агента 3.1
├── agent_3_2.py                       ← Python-entrypoint агента 3.2
├── agent_3_3.py                       ← Python-entrypoint агента 3.3
├── agent_3_4.py                       ← Python-entrypoint агента 3.4
├── agent_3_5.py                       ← Python-entrypoint агента 3.5
├── module_v_all.py                    ← единый Python-запуск всех агентов
├── launcher_entry.py                  ← общий exe-лаунчер
├── build.py                           ← сборка libs/ и exe в корне
├── libs/                              ← общие библиотеки для всех exe
├── requirements-agent-runtime.txt     ← минимальный runtime-набор для libs/
│
├── .env                               ← API-ключи, строка БД
├── requirements.txt                   ← Зависимости Python 3.12
└── README.md                          ← Этот файл
```

---

## 1. Обучение моделей

**Ноутбук:** `notebooks/3.1_model_training.ipynb`

### Задачи классификации

| Задача | Целевая переменная | Классов | Источник меток |
|--------|--------------------|---------|----------------|
| Параллельное изучение | `parallel_cluster` | 4 | K-Means (Модуль Б) |
| Последовательное изучение | `sequential_cluster` | 4 | Agglomerative (Модуль Б) |
| Сложность освоения | `complexity_cluster` | 3 | K-Means (Модуль Б) |

### Признаки

- **Текстовые:** TF-IDF по полю `text_content` (300 токенов, sublinear_tf=True)
- **Числовые:** word_count, avg_sentence_length, media_count, has_images, has_videos, has_questions, compliance_score, is_generated

### Рассмотренные алгоритмы (≥ 3 на каждую задачу)

| # | Алгоритм | Тип | Обоснование выбора |
|---|---------|-----|-------------------|
| 1 | **Logistic Regression** | Линейный | Baseline; эффективен на разреженных TF-IDF матрицах; интерпретируем |
| 2 | **Random Forest** | Ансамбль деревьев | Устойчив к шуму; не требует масштабирования; даёт feature importance |
| 3 | **XGBoost** | Градиентный бустинг | L1/L2 регуляризация; высокое качество на табличных данных |
| 4 | **LightGBM** ★ | Leaf-wise бустинг | Быстрее XGBoost на TF-IDF; поддерживает warm start для дообучения |

### Выбор лучшей модели

Победитель определяется **автоматически** по наибольшему **F1-macro** на Stratified K-Fold Cross-Validation (до 5 фолдов). Финальная модель проверяется на **отложенной выборке 20%** по метрикам:

| Метрика | Описание |
|---------|---------|
| **Accuracy** | Доля правильно классифицированных примеров |
| **Precision** | Точность (macro-average) |
| **Recall** | Полнота (macro-average) |
| **F1-macro** | Гармоническое среднее Precision и Recall |
| **ROC-AUC** | Площадь под ROC-кривой (OvR, macro) |

### Формат и пути сохранённых моделей

```
models/v_YYYYMMDD_HHMMSS/
├── parallel_cluster_best.joblib    # sklearn/LightGBM/XGBoost модель
├── sequential_cluster_best.joblib
├── complexity_cluster_best.joblib
├── tfidf_vectorizer.joblib         # sklearn TfidfVectorizer
├── scaler.joblib                   # sklearn StandardScaler
└── meta.json                       # алгоритм, метрики, дата, n_materials

models/current_version.txt          # имя актуальной версии
```

**Загрузка модели:**
```python
import joblib
from pathlib import Path

version = Path('models/current_version.txt').read_text().strip()
clf  = joblib.load(f'models/v_{version}/parallel_cluster_best.joblib')
tfidf = joblib.load(f'models/v_{version}/tfidf_vectorizer.joblib')
```

---

## 2. Непрерывное обучение

**Ноутбук:** `notebooks/3.2_continuous_learning.ipynb`

### Алгоритм работы агента

```
Новые данные из БД
        │
        ▼
┌─────────────────┐
│ Evidently Drift │  → reports/drift_report.html
│    Detection    │
└────────┬────────┘
    drift > 20%?
    ┌────┴────┐
   ДА       НЕТ
    ▼         ▼
 Полное   Дообучение
 переобу- LightGBM
 чение    (+50 деревьев,
 с нуля   warm start)
    └────┬────┘
         ▼
  models/v_YYYYMMDD_HHMMSS/
         │
         ▼
  logs/training_log.jsonl
```

### Контроль дрейфа данных (Evidently)

Используется `DataDriftPreset` из библиотеки **Evidently** для анализа распределений числовых признаков между эталонной (reference) и новой (current) выборками.

| Условие | Действие |
|---------|---------|
| Дрейф < 20% признаков | Инкрементальное дообучение LightGBM (+50 деревьев, `init_model`) |
| Дрейф ≥ 20% признаков | Полное переобучение с нуля |
| Деградация F1 ≥ 5% | Полное переобучение с нуля |

### Версионирование и логирование

- **Версии:** каждое обучение → новая папка `models/v_YYYYMMDD_HHMMSS/` с `meta.json`
- **Лог:** `logs/training_log.jsonl` — одна строка JSON на каждое обучение:
  ```json
  {
    "version": "20240501_143022",
    "trained_at": "2024-05-01T14:30:22",
    "update_type": "incremental",
    "drift_pct": 0.12,
    "n_materials": 87,
    "models": {
      "parallel_cluster": {"algorithm": "LGBMClassifier", "metrics": {"f1": 0.82, ...}},
      ...
    }
  }
  ```
- **Лог в БД:** каждая версия модели дополнительно записывается в PostgreSQL-таблицу `public.module_v_model_versions`
  с полями версии, времени обучения, типа обновления, `drift_pct`, числа материалов и JSON-метаданными.

---

## 3. Оценка времени освоения

**Ноутбук:** `notebooks/3.3_time_estimation.ipynb`

### Формула расчёта времени

```
t_material = (word_count / 200) × k_сложности
           + has_images    × 2 мин
           + has_videos    × 5 мин
           + has_questions × 3 мин
```

**Коэффициенты сложности:**

| Кластер | Уровень | Коэффициент |
|---------|---------|-------------|
| 0 | Базовый | × 1.0 |
| 1 | Средний | × 1.4 |
| 2 | Продвинутый | × 2.0 |

### Агрегация по предметам

| Режим | Формула | Смысл |
|-------|---------|-------|
| Последовательный | Σ время всех материалов | Изучаем всё подряд |
| Параллельный | max по кластерам | Параллельные потоки → итог по самому долгому |

### Визуализации

| Файл | Тип | Содержимое |
|------|-----|-----------|
| `reports/gantt_sequential.html` | HTML (Plotly) | Диаграмма Ганта: последовательный план |
| `reports/gantt_parallel.html` | HTML (Plotly) | Диаграмма Ганта: параллельный план |
| `reports/time_by_subject.png` | PNG | Последовательное vs параллельное время по предметам |
| `reports/time_boxplot.html` | HTML (Plotly) | Box-plot: время по предметам и уровням сложности |
| `reports/time_estimation_summary.csv` | CSV | Сводная таблица временных характеристик |

---

## 4. Индивидуальные траектории

**Ноутбук:** `notebooks/3.4_recommendations.ipynb`

### Входные данные пользователя

Агент запрашивает данные через **интерактивные виджеты (ipywidgets)**:

| Параметр | Виджет | Описание |
|---------|--------|---------|
| Уже изученные предметы | `SelectMultiple` | Исключаются из плана |
| Часов в день | `FloatSlider` | 0.5–12 ч |
| Недель на курс | `IntSlider` | Дедлайн |
| Дата начала | `DatePicker` | Начало плана |
| Приоритет | `RadioButtons` | Параллельный / последовательный |

### Алгоритм построения траектории

1. Исключаем уже изученные предметы
2. **Параллельный режим:** группируем по `parallel_cluster` → предметы внутри кластера идут одновременно, кластеры — последовательно
3. **Последовательный режим:** упорядочиваем по `sequential_cluster` → все предметы идут друг за другом
4. Проверяем, укладывается ли план в дедлайн

### Визуализации

| Файл | Содержимое |
|------|-----------|
| `reports/trajectory.html` | Интерактивная диаграмма Ганта с дедлайном |
| `reports/trajectory_graph.html` | ГРАФ индивидуальной траектории обучения |
| `reports/trajectory_graph.png` | Статичный рендер ГРАФА траектории |
| `reports/trajectory_plan.csv` | Таблица плана (предмет, даты, часов, материалов) |
| `reports/trajectory_comparison.png` | Сравнение: параллельный vs. последовательный план |
| `reports/trajectory_report.html` | HTML-отчёт со сводкой и таблицей плана |

Важно:
- для критерия визуализации траектории теперь используется не только диаграмма Ганта, но и отдельный **ГРАФ траектории обучения**;
- граф сохраняется в `reports/trajectory_graph.html` и включается в итоговый отчёт.

---

## 5. Итоговый отчёт

**Ноутбук:** `notebooks/3.5_report.ipynb`  
**Выходной файл:** `reports/module_v_report.html`

Сводный профессиональный HTML-отчёт, который автоматически собирает данные из предыдущих ноутбуков:

- Постановка задачи и признаки
- Сравнительная таблица алгоритмов с обоснованием
- Результаты тестирования (метрики + матрицы ошибок)
- Схема непрерывного обучения + история версий
- Оценки времени освоения с визуализациями
- Пример индивидуальной траектории
- Указание, что версии моделей логируются и в файлы, и в БД `public.module_v_model_versions`
- Полный реестр файлов со статусом существования

При запуске `3.5` автоматически пересобираются:
- артефакты в `reports/` от `3.1`, `3.2`, `3.3`, `3.4`
- HTML-файлы в корне: `3.1_model_training.html`, `3.2_continuous_learning.html`, `3.3_time_estimation.html`, `3.4_recommendations.html`, `3.5_report.html`

---

## HTML-экспорт ноутбуков

Для удобного просмотра результатов без открытия Jupyter в корень проекта сохраняются HTML-версии выполненных ноутбуков:

| Файл | Назначение |
|------|-----------|
| `3.1_model_training.html` | Полный вывод обучения моделей, сравнение алгоритмов, метрики и матрицы ошибок |
| `3.2_continuous_learning.html` | Полный вывод непрерывного обучения, drift-анализ и переобучение/дообучение |
| `3.3_time_estimation.html` | Полный вывод расчёта времени освоения и визуализаций |
| `3.4_recommendations.html` | Полный вывод построения траектории обучения |
| `3.5_report.html` | HTML-копия итогового notebook-отчёта |

Примечания:
- HTML-файлы в корне отражают **последний выполненный запуск** notebook.
- Основные интерактивные отчёты Plotly и Evidently также сохраняются в папку `reports/`.
- В статическом HTML интерактивные виджеты `ipywidgets` не заменяют живой ввод пользователя, но позволяют просматривать результаты последнего запуска без открытия `.ipynb`.

Если нужно быстро посмотреть результаты без Jupyter:
- `3.1_model_training.html` — обучение и метрики моделей
- `3.2_continuous_learning.html` — drift и обновление моделей
- `3.3_time_estimation.html` — оценка времени
- `3.4_recommendations.html` — траектория обучения
- `3.5_report.html` — сводный итоговый просмотр

---

## EXE-агенты и сборка

Для каждого агента подготовлен отдельный Python-entrypoint и отдельный exe-лаунчер с общей папкой `libs/`.

| Агент | Скрипт | Назначение |
|------|--------|-----------|
| 3.1 | `agent_3_1.py` | Обучение моделей |
| 3.2 | `agent_3_2.py` | Непрерывное обучение |
| 3.3 | `agent_3_3.py` | Оценка времени |
| 3.4 | `agent_3_4.py` | Построение рекомендаций |
| 3.5 | `agent_3_5.py` | Итоговый отчёт |
| Все агенты | `module_v_all.py` | Полный запуск проекта |

Все агенты выводят действия в консоль: старт, запуск notebook, экспорт HTML, завершение.

Сборка выполняется одной командой:

```bash
python build.py
```

После сборки в корне проекта появляются общая папка библиотек и exe:

```text
model_V_full/
├── libs/
├── agent_3_1_training.exe
├── agent_3_2_continuous_learning.exe
├── agent_3_3_time_estimation.exe
├── agent_3_4_recommendations.exe
├── agent_3_5_report.exe
└── module_v_all.exe
```

Важно:
- запускать нужно exe из корня проекта
- все exe используют одну общую папку `libs/`, поэтому библиотеки не дублируются в каждом агенте
- `build.py` наполняет `libs/` только нужными runtime-зависимостями из `requirements-agent-runtime.txt`, а не копией всего `site-packages`
- сами exe запускают существующий Python в системе или локальный `.venv`, а зависимости берут из `./libs`
- `.exe` читают настройки подключения из `.env` во время запуска, поэтому при смене БД пересборка не требуется; достаточно обновить `.env`
- временные каталоги сборки `build/`, `.pyinstaller/`, `.cachedir` можно удалять после сборки

Краткая инструкция запуска:
- `agent_3_1_training.exe` — обучает модели и обновляет артефакты 3.1
- `agent_3_2_continuous_learning.exe` — выполняет drift-анализ и обновление модели
- `agent_3_3_time_estimation.exe` — считает временные характеристики и визуализации
- `agent_3_4_recommendations.exe` — строит траекторию обучения
- `agent_3_5_report.exe` — обновляет полный пайплайн и итоговый отчёт
- `module_v_all.exe` — запускает весь процесс целиком

---

## Сверка по критериям

Ниже приведена сверка по критериям модуля В на текущем состоянии проекта после повторной проверки выполнения notebook.

### 1. Обучение модели — модуль В

| Критерий | Статус | Подтверждение |
|------|--------|---------------|
| Обучена модель для классификации материалов по возможности совместного (параллельного) изучения | Да | `parallel_cluster`, notebook `3.1`, артефакт `models/v_*/parallel_cluster_best.joblib` |
| Обучена модель для определения возможности последовательного изучения материалов | Да | `sequential_cluster`, notebook `3.1`, артефакт `models/v_*/sequential_cluster_best.joblib` |
| Обучена модель для определения уровня сложности освоения материала | Да | `complexity_cluster`, notebook `3.1`, артефакт `models/v_*/complexity_cluster_best.joblib` |
| Для каждой модели рассмотрено не менее трёх методов классификации и обоснован выбор алгоритма | Да | В `3.1` сравниваются `LogisticRegression`, `RandomForest`, `XGBoost`, `LightGBM`; обоснование приведено в README и итоговом отчёте |

### 2. Непрерывное обучение — модуль В

| Критерий | Статус | Подтверждение |
|------|--------|---------------|
| Реализован механизм дообучения модели при поступлении новых данных | Да | Notebook `3.2`, сценарий `incremental update`, обновление версии модели |
| Контролируется дрейф данных (например, через Evidently) | Да | Используется `Evidently` / `DataDriftPreset`, артефакт `reports/drift_report.html` |
| Предусмотрено полное переобучение модели при необходимости | Да | В `3.2` есть ветка полного переобучения при высоком drift или деградации метрик |
| Версии моделей сохраняются и логируются изменения | Да | `models/v_YYYYMMDD_HHMMSS/`, `models/current_version.txt`, `logs/training_log.jsonl`, `public.module_v_model_versions` |

### 3. Прогнозирование — модуль В

| Критерий | Статус | Подтверждение |
|------|--------|---------------|
| Агент способен оценить время освоения каждого учебного материала, предмета/дисциплины, набора предметов/дисциплин с учётом параллельного/последовательного изучения и сложности | Да | Notebook `3.3`, расчёт `time_min`, агрегация по предметам и по планам изучения |
| Присутствует визуализация временных характеристик | Да | `reports/time_by_subject.png`, `reports/gantt_sequential.html`, `reports/gantt_parallel.html`, `reports/time_boxplot.html` |

### 4. Построение рекомендаций — модуль В

| Критерий | Статус | Подтверждение |
|------|--------|---------------|
| Агент запрашивает сведения от пользователя в качестве входных данных | Да, с оговоркой | В notebook `3.4` используются `ipywidgets` (`SelectMultiple`, `FloatSlider`, `IntSlider`, `DatePicker`, `RadioButtons`). В статическом HTML это просмотр последнего запуска, а не живой ввод |
| Агент визуализирует траектории обучения для пользователя | Да | `reports/trajectory.html`, `reports/trajectory_graph.html`, `reports/trajectory_report.html`, `reports/trajectory_comparison.png` |

### 5. Подготовка отчётов — модуль В

| Критерий | Статус | Подтверждение |
|------|--------|---------------|
| Отчёт присутствует | Да | `reports/module_v_report.html`, а также `3.5_report.html` в корне проекта |
| Отчёт выполнен профессионально: описание модели, обоснование преимуществ, результаты тестирования, схема непрерывного обучения, визуализация оценок и рекомендаций, пути к файлам | Да | Всё перечисленное собирается в notebook `3.5` и сохраняется в итоговый HTML-отчёт |

### Итоговая самооценка по шкале отчёта

| Уровень | Оценка |
|------|--------|
| 0 | Не соответствует: отчёт отсутствует или несодержательный |
| 1 | Частично соответствует: отчёт содержательный, но оформлен непрофессионально |
| 2 | Соответствует: отчёт содержательный и выполнен профессионально |
| 3 | Высший уровень: отчёт выполнен на высшем профессиональном уровне |

Текущее состояние проекта оценивается как **2–3**, ближе к **3**, поскольку:
- есть отдельный итоговый HTML-отчёт;
- присутствуют метрики, визуализации, схема непрерывного обучения и реестр артефактов;
- дополнительно доступны HTML-версии всех notebook в корне проекта.

---

## Пути к файлам и форматы

### Модели (формат: `joblib`)

| Файл | Описание |
|------|---------|
| `models/v_*/parallel_cluster_best.joblib` | Классификатор параллельного изучения |
| `models/v_*/sequential_cluster_best.joblib` | Классификатор последовательного изучения |
| `models/v_*/complexity_cluster_best.joblib` | Классификатор уровня сложности |
| `models/v_*/tfidf_vectorizer.joblib` | TF-IDF векторизатор (общий для всех моделей) |
| `models/v_*/scaler.joblib` | StandardScaler числовых признаков |
| `models/v_*/meta.json` | Метаданные: алгоритм, метрики, дата обучения, тип обновления |
| `models/current_version.txt` | Имя актуальной версии |

### Логи (формат: `JSONL`)

| Файл | Описание |
|------|---------|
| `logs/training_log.jsonl` | История всех обучений: версия, метрики, дрейф, тип обновления |
| `logs/known_material_ids.json` | ID материалов, использованных в последнем обучении |
| `public.module_v_model_versions` | Таблица PostgreSQL с версиями моделей и метаданными обучения |

### Отчёты

| Файл | Формат | Источник |
|------|--------|---------|
| `reports/module_v_report.html` | HTML | 3.5 — итоговый отчёт |
| `reports/model_comparison.png` | PNG | 3.1 — сравнение алгоритмов |
| `reports/confusion_matrices.png` | PNG | 3.1 — матрицы ошибок |
| `reports/model_metrics_summary.csv` | CSV | 3.1 — итоговые метрики |
| `reports/drift_report.html` | HTML | 3.2 — Evidently drift report |
| `reports/drift_distributions.png` | PNG | 3.2 — гистограммы дрейфа |
| `reports/model_update_comparison.png` | PNG | 3.2 — F1 до/после обновления |
| `reports/time_estimation_summary.csv` | CSV | 3.3 — время по предметам |
| `reports/gantt_sequential.html` | HTML | 3.3 — последовательный Ганта |
| `reports/gantt_parallel.html` | HTML | 3.3 — параллельный Ганта |
| `reports/trajectory.html` | HTML | 3.4 — траектория пользователя |
| `reports/trajectory_graph.html` | HTML | 3.4 — ГРАФ траектории пользователя |
| `reports/trajectory_graph.png` | PNG | 3.4 — статичный рендер ГРАФА траектории |
| `reports/trajectory_plan.csv` | CSV | 3.4 — план траектории |
| `reports/trajectory_comparison.png` | PNG | 3.4 — сравнение планов |

### HTML-копии notebook в корне проекта

| Файл | Формат | Источник |
|------|--------|---------|
| `3.1_model_training.html` | HTML | Экспорт выполненного notebook 3.1 |
| `3.2_continuous_learning.html` | HTML | Экспорт выполненного notebook 3.2 |
| `3.3_time_estimation.html` | HTML | Экспорт выполненного notebook 3.3 |
| `3.4_recommendations.html` | HTML | Экспорт выполненного notebook 3.4 |
| `3.5_report.html` | HTML | Экспорт выполненного notebook 3.5 |

---

## Зависимости

```
Python 3.12

scikit-learn    — LR, Random Forest, TF-IDF, метрики
xgboost         — XGBoost классификатор
lightgbm        — LightGBM классификатор (warm start)
evidently       — контроль дрейфа данных
joblib          — сохранение/загрузка моделей
pandas, numpy   — работа с данными
plotly          — интерактивные диаграммы Ганта
matplotlib      — статичные графики
ipywidgets      — виджеты ввода в ноутбуках
psycopg         — подключение к PostgreSQL
python-dotenv   — загрузка .env
```

Полный список: `requirements.txt`

Для сборки общей папки `libs/` используется сокращённый runtime-список: `requirements-agent-runtime.txt`.
