"""
Statistical analysis and visualisation of the materials dataset.

Produces:
  - analysis_output/materials_dataset.csv
  - analysis_output/attribute_descriptions.md
  - analysis_output/full_report.md
  - analysis_output/*.png  (charts)
"""

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys


def _df_to_md(df: pd.DataFrame) -> str:
    """Markdown table without tabulate dependency."""
    if df is None or df.empty:
        return "Нет данных."
    headers = " | ".join(str(c) for c in df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(v) if v is not None else "" for v in row) for row in df.itertuples(index=False, name=None)]
    return "| " + " |\n| ".join([headers, sep] + rows) + " |"

sys.path.insert(0, str(Path(__file__).parent.parent))
import importlib.util as _ilu
_a_cfg_path = Path(__file__).parent.parent / "config.py"
_a_cfg_spec = _ilu.spec_from_file_location("_module_a_config", _a_cfg_path)
_a_cfg = _ilu.module_from_spec(_a_cfg_spec)
_a_cfg_spec.loader.exec_module(_a_cfg)
ANALYSIS_OUTPUT_DIR = _a_cfg.ANALYSIS_OUTPUT_DIR
REPORTS_DIR = _a_cfg.REPORTS_DIR
from database.manager import get_all_materials, get_feature_importance, get_material_features, get_media_items
from processors.text_extractor import word_count


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------

FEATURE_META: Dict[str, Dict] = {
    "id":                   {"name": "Идентификатор",            "dtype": "integer",         "purpose": "Уникальный ключ записи"},
    "url":                  {"name": "URL источника",             "dtype": "text",            "purpose": "Адрес загруженного материала"},
    "subject":              {"name": "Предмет",                  "dtype": "categorical",     "purpose": "Учебная дисциплина"},
    "topic":                {"name": "Тема",                     "dtype": "categorical",     "purpose": "Тема учебного материала"},
    "text_content":         {"name": "Текст материала",          "dtype": "text (long)",     "purpose": "Полный текст + описания медиа"},
    "annotation":           {"name": "Аннотация",                "dtype": "text",            "purpose": "Краткое резюме материала"},
    "file_type":            {"name": "Тип файла",                "dtype": "categorical",     "purpose": "Формат представления материала"},
    "language":             {"name": "Язык",                     "dtype": "categorical",     "purpose": "Язык материала"},
    "lesson_type":          {"name": "Тип занятия",             "dtype": "categorical",     "purpose": "Класс учебного материала: lecture / seminar / practice / lab / self_study / test / other"},
    "moderation_status":    {"name": "Статус модерации",         "dtype": "categorical",     "purpose": "approved / rejected / needs_revision"},
    "moderation_notes":     {"name": "Заметки модерации",        "dtype": "text",            "purpose": "Выводы и рекомендации модератора"},
    "compliance_score":     {"name": "Оценка соответствия",      "dtype": "float [0-10]",    "purpose": "Интегральная оценка качества"},
    "is_compliant":         {"name": "Допустимость использования","dtype": "boolean",        "purpose": "Итоговый бинарный вывод модератора"},
    "is_generated":         {"name": "Сгенерировано AI",         "dtype": "boolean",         "purpose": "True если материал создан агентом"},
    "has_previous":         {"name": "Есть предыдущий материал", "dtype": "boolean",         "purpose": "Наличие предшествующей темы в программе"},
    "previous_material_id": {"name": "ID предыдущего материала", "dtype": "integer",         "purpose": "Ссылка на предыдущий материал в тематической последовательности"},
    "has_next":             {"name": "Есть следующий материал",  "dtype": "boolean",         "purpose": "Наличие последующей темы в программе"},
    "next_material_id":     {"name": "ID следующего материала",  "dtype": "integer",         "purpose": "Ссылка на следующий материал в тематической последовательности"},
    "text_length":          {"name": "Длина текста (символы)",   "dtype": "integer",         "purpose": "Объём текстового контента"},
    "word_count":           {"name": "Количество слов",          "dtype": "integer",         "purpose": "Информационная насыщенность"},
    "sentence_count":       {"name": "Количество предложений",   "dtype": "integer",         "purpose": "Структурная сложность текста"},
    "avg_sentence_length":  {"name": "Средняя длина предложения","dtype": "float",           "purpose": "Языковая сложность изложения"},
    "has_introduction":     {"name": "Наличие введения",         "dtype": "binary (0/1)",    "purpose": "Структурность материала"},
    "has_conclusion":       {"name": "Наличие заключения",       "dtype": "binary (0/1)",    "purpose": "Полнота структуры"},
    "has_questions":        {"name": "Наличие вопросов",         "dtype": "binary (0/1)",    "purpose": "Педагогическая ценность"},
    "has_images":           {"name": "Наличие изображений",      "dtype": "binary (0/1)",    "purpose": "Визуальный компонент"},
    "has_videos":           {"name": "Наличие видео",            "dtype": "binary (0/1)",    "purpose": "Мультимедийный компонент"},
    "has_audio":            {"name": "Наличие аудио",            "dtype": "binary (0/1)",    "purpose": "Аудиальный компонент"},
    "media_count":          {"name": "Кол-во медиаэлементов",    "dtype": "integer",         "purpose": "Мультимедийная насыщенность"},
    "media_descriptions":   {"name": "Описания медиа",           "dtype": "text",            "purpose": "Текстовые описания изображений, видео и аудио, включенных в материал"},
    "combined_content":     {"name": "Объединенный контент",     "dtype": "text (long)",     "purpose": "Текст материала, дополненный описаниями нетекстовых элементов"},
    "usage_decision":       {"name": "Решение об использовании", "dtype": "categorical",     "purpose": "Итоговое представление вывода о допустимости: allowed / not_allowed"},
    "parallel_cluster":     {"name": "Кластер параллельного изучения", "dtype": "integer",   "purpose": "Метка кластера для материалов, подходящих для совместного изучения"},
    "sequential_cluster":   {"name": "Кластер последовательного изучения", "dtype": "integer", "purpose": "Метка кластера для логической цепочки материалов"},
    "complexity_cluster":   {"name": "Кластер сложности",        "dtype": "integer",         "purpose": "Числовой кластер сложности освоения"},
    "difficulty_label":     {"name": "Метка сложности",          "dtype": "categorical",     "purpose": "Интерпретируемая метка сложности: Базовый / Средний / Продвинутый"},
}


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def build_dataframe() -> pd.DataFrame:
    mats = get_all_materials()
    rows = []
    for m in mats:
        feats = get_material_features(m["id"])
        row = dict(m)
        media_items = get_media_items(m["id"])
        row["media_descriptions"] = "\n".join(
            f"{item.get('media_type', 'media')}: {item.get('description', '')}".strip()
            for item in media_items
            if item.get("description")
        )
        row["usage_decision"] = "allowed" if row.get("is_compliant") else "not_allowed"
        row["combined_content"] = "\n\n".join(
            part for part in [row.get("text_content", ""), row.get("media_descriptions", "")]
            if part
        )
        for k, v in feats.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Attribute report
# ---------------------------------------------------------------------------

def generate_attribute_report(df: pd.DataFrame) -> str:
    lines = ["# Описание атрибутов набора данных\n"]

    for col in df.columns:
        meta = FEATURE_META.get(col, {"name": col, "dtype": str(df[col].dtype), "purpose": "Дополнительный атрибут"})
        lines += [
            f"## `{col}`",
            f"**Название:** {meta['name']}",
            f"**Тип данных:** {meta['dtype']}",
            f"**Назначение:** {meta['purpose']}",
        ]

        n_unique = df[col].nunique(dropna=True)
        lines.append(f"**Уникальных значений:** {n_unique}")

        # Frequency table for low-cardinality
        if n_unique <= 15 or df[col].dtype == object:
            vc = df[col].value_counts(dropna=False).head(8)
            lines.append("**Топ значений:**")
            for val, cnt in vc.items():
                lines.append(f"  - `{val}`: {cnt} ({cnt/len(df)*100:.1f}%)")

        # Numeric stats
        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 2:
            lines += [
                f"**Среднее:** {df[col].mean():.2f}",
                f"**Мин:** {df[col].min():.2f}  |  **Макс:** {df[col].max():.2f}",
                f"**Станд. откл.:** {df[col].std():.2f}",
            ]

        # Text stats
        if meta["dtype"].startswith("text") or col in ("topic", "media_descriptions", "combined_content"):
            series = df[col].dropna().astype(str)
            series = series[series.str.len() > 0]
            if len(series):
                lengths = series.str.len()
                wc = series.apply(word_count)
                lines += [
                    "**Статистика текста:**",
                    f"  - Ср. длина: {lengths.mean():.0f} симв.",
                    f"  - Мин/Макс: {lengths.min()} / {lengths.max()} симв.",
                    f"  - Ср. слов: {wc.mean():.0f}",
                ]

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def generate_full_report(df: pd.DataFrame) -> str:
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    n_gen = int(df["is_generated"].sum()) if "is_generated" in df.columns else 0
    lines = [
        "# Аналитический отчёт по набору учебных материалов",
        f"**Дата:** {ts}  |  **Материалов всего:** {len(df)}  |  **Сгенерировано:** {n_gen}",
        "",
    ]

    # Moderation summary
    if "moderation_status" in df.columns:
        lines.append("## Результаты модерации\n")
        lines.append("| Статус | Кол-во | % |")
        lines.append("|--------|-------|----|")
        for status, cnt in df["moderation_status"].value_counts(dropna=False).items():
            lines.append(f"| {status} | {cnt} | {cnt/len(df)*100:.1f}% |")
        lines.append("")

    if "compliance_score" in df.columns:
        s = df["compliance_score"].dropna()
        if len(s):
            lines += [
                "## Оценки соответствия методическим рекомендациям\n",
                f"| Показатель | Значение |",
                f"|------------|---------|",
                f"| Средняя оценка | **{s.mean():.2f}/10** |",
                f"| Минимум | {s.min():.2f} |",
                f"| Максимум | {s.max():.2f} |",
                f"| Стандартное отклонение | {s.std():.2f} |",
                "",
            ]

    if "subject" in df.columns:
        lines.append("## Распределение по предметам\n")
        lines.append("| Предмет | Материалов |")
        lines.append("|---------|-----------|")
        for s, c in df["subject"].value_counts(dropna=False).items():
            lines.append(f"| {s or '—'} | {c} |")
        lines.append("")

    # Text statistics table
    for col in ["text_content", "annotation"]:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        series = series[series.str.len() > 0]
        if len(series) == 0:
            continue
        lens = series.str.len()
        wcs = series.apply(word_count)
        lines += [
            f"## Статистика поля «{col}»\n",
            f"| Показатель | Значение |",
            f"|------------|---------|",
            f"| Заполнено | {len(series)}/{len(df)} |",
            f"| Ср. длина | {lens.mean():.0f} симв. |",
            f"| Мин/Макс длина | {lens.min()} / {lens.max()} симв. |",
            f"| Ср. кол-во слов | {wcs.mean():.0f} |",
            "",
        ]

    return "\n".join(lines)


def generate_module_a_report(df: pd.DataFrame) -> str:
    sample_cols = [
        c for c in [
            "id", "subject", "topic", "lesson_type", "moderation_status",
            "compliance_score", "is_generated", "has_previous", "has_next",
        ]
        if c in df.columns
    ]
    sample_text = _df_to_md(df[sample_cols].head(10)) if sample_cols else "Нет данных."

    lines = [
        "# Отчет по Модулю А",
        "",
        "## Реализованный функционал",
        "- Агент загрузки в текущей конфигурации обрабатывает файлы из папки `test_files`; URL-ветка из `module_A/urls.txt` сохранена в коде и может быть включена при необходимости без перестройки архитектуры.",
        "- Агент анализирует медиаконтент: изображения, видео и аудио описываются отдельно и добавляются в итоговый набор данных.",
        "- Агент модерации формирует итоговое заключение о допустимости использования материала и сохраняет результаты проверки по категориям методических требований.",
        "- Агент предобработки рассчитывает признаки, значимость признаков и связи `previous/next` между материалами по тематике.",
        "- Агент расширения находит пробелы в тематике, предлагает пользователю новые темы и добавляет сгенерированные материалы в БД и датасет.",
        "",
        "## Обоснование выбранных подходов",
        "- Для загрузки в рабочем сценарии выбран file-driven pipeline с upsert-логикой по идентификатору источника, что исключает дубли и позволяет повторно прогонять проект на одних и тех же материалах без размножения записей.",
        "- Для текстовой обработки используются специализированные экстракторы по форматам файлов, чтобы поддержать разнородные учебные материалы.",
        "- Для медиаконтента используется мультимодальный LLM-анализ, поскольку он даёт унифицированные описания изображений, видео и аудио в текстовом виде.",
        "- Для значимых атрибутов используются статистические признаки, корреляционный анализ и permutation/SHAP importance, что соответствует критериям задания.",
        "",
        "## Структура базы данных",
        "- `materials`: основная аналитическая запись материала.",
        "- `media_items`: нетекстовые элементы и их описания.",
        "- `material_features`: признаки для анализа и кластеризации.",
        "- `feature_importance`: результаты оценки важности признаков.",
        "- `methodology_requirements`: перечень методических требований.",
        "- `methodology_compliance`: результаты проверки каждого материала по каждому требованию.",
        "",
        "## Форматы итоговых файлов",
        "- `PostgreSQL`: основная база данных проекта, используемая всеми агентами и дашбордом.",
        "- `module_A/analysis_output/materials_dataset.csv`: итоговый датасет.",
        "- `module_A/analysis_output/attribute_descriptions.md`: описание атрибутов и статистика.",
        "- `module_A/analysis_output/full_report.md`: аналитическая сводка по датасету.",
        "- `module_A/analysis_output/*.png`: графические визуализации.",
        "- `module_A/reports/report_module_a.md`: настоящий отчет.",
        "",
        "## Пример итоговой записи датасета",
        _df_to_md(df.head(1)) if not df.empty else "Нет данных.",
        "",
        "## Примеры записей датасета",
        sample_text,
        "",
        "## Пример полной записи",
        _df_to_md(df.head(1)) if not df.empty else "Нет данных.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def create_visualizations(df: pd.DataFrame) -> List[str]:
    paths: List[str] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    def save(name: str) -> str:
        p = str(ANALYSIS_OUTPUT_DIR / name)
        plt.tight_layout()
        plt.savefig(p, dpi=100)
        plt.close()
        paths.append(p)
        return p

    # 1. Moderation status distribution
    if "moderation_status" in df.columns:
        vc = df["moderation_status"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {"approved": "#4CAF50", "rejected": "#F44336", "needs_revision": "#FF9800", "pending": "#9E9E9E"}
        bar_colors = [colors.get(s, "#9E9E9E") for s in vc.index]
        ax.bar(vc.index.astype(str), vc.values, color=bar_colors)
        ax.set_title("Распределение статусов модерации")
        ax.set_ylabel("Количество материалов")
        for i, v in enumerate(vc.values):
            ax.text(i, v + 0.1, str(v), ha="center")
        save("01_moderation_status.png")

    # 2. Compliance score histogram
    if "compliance_score" in df.columns:
        scores = df["compliance_score"].dropna()
        if len(scores) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(scores, bins=min(10, len(scores)), color="#2196F3", edgecolor="white")
            ax.axvline(scores.mean(), color="red", linestyle="--", label=f"Среднее: {scores.mean():.1f}")
            ax.set_title("Распределение оценок соответствия (0–10)")
            ax.set_xlabel("Оценка")
            ax.set_ylabel("Количество материалов")
            ax.legend()
            save("02_compliance_scores.png")

    # 3. Subject distribution
    if "subject" in df.columns:
        vc = df["subject"].value_counts().head(10)
        if len(vc):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(vc.index.astype(str), vc.values, color="#009688")
            ax.set_title("Топ-10 предметов по количеству материалов")
            ax.set_xlabel("Количество материалов")
            save("03_subject_distribution.png")

    # 4. Correlation heatmap
    num_cols = [c for c in ["text_length", "word_count", "sentence_count", "media_count", "compliance_score"] if c in df.columns]
    if len(num_cols) >= 3:
        num_df = df[num_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(num_df) >= 3:
            fig, ax = plt.subplots(figsize=(7, 6))
            corr = num_df.corr()
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                        cbar_kws={"label": "Корреляция"})
            ax.set_title("Матрица корреляций числовых признаков")
            save("04_correlation_heatmap.png")

    # 5. Media presence
    media_cols = [c for c in ["has_images", "has_videos", "has_audio"] if c in df.columns]
    if media_cols:
        counts = {c.replace("has_", ""): int(pd.to_numeric(df[c], errors="coerce").sum()) for c in media_cols}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(list(counts.keys()), list(counts.values()), color=["#2196F3", "#F44336", "#4CAF50"])
        ax.set_title("Наличие медиаконтента в материалах")
        ax.set_ylabel("Количество материалов")
        save("05_media_presence.png")

    # 6. Word count by subject
    if "subject" in df.columns and "word_count" in df.columns:
        grouped = df.groupby("subject")["word_count"].mean().sort_values(ascending=False).head(10)
        if len(grouped) > 1:
            fig, ax = plt.subplots(figsize=(9, 5))
            grouped.apply(pd.to_numeric, errors="coerce").plot(kind="bar", ax=ax, color="teal")
            ax.set_title("Среднее кол-во слов по предметам")
            ax.set_xlabel("Предмет")
            ax.set_ylabel("Ср. кол-во слов")
            plt.xticks(rotation=35, ha="right")
            save("06_word_count_by_subject.png")

    # 7. Feature importance bar chart (if available)
    try:
        rows = get_feature_importance(limit=12)
        if rows:
            names = [r["feature_name"] for r in rows]
            scores = [r["importance_score"] for r in rows]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(names[::-1], scores[::-1], color="#673AB7")
            ax.set_title("Важность признаков (Feature Importance)")
            ax.set_xlabel("Нормализованная важность")
            save("07_feature_importance.png")
    except Exception:
        pass

    print(f"  Created {len(paths)} visualisations in {ANALYSIS_OUTPUT_DIR}")
    return paths


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run_analysis():
    df = build_dataframe()
    if df.empty:
        print("  No data to analyse.")
        return df

    csv_path = ANALYSIS_OUTPUT_DIR / "materials_dataset.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  Dataset saved: {csv_path}")

    attr_report = generate_attribute_report(df)
    (ANALYSIS_OUTPUT_DIR / "attribute_descriptions.md").write_text(attr_report, encoding="utf-8")

    full_report = generate_full_report(df)
    (ANALYSIS_OUTPUT_DIR / "full_report.md").write_text(full_report, encoding="utf-8")

    module_a_report = generate_module_a_report(df)
    (REPORTS_DIR / "report_module_a.md").write_text(module_a_report, encoding="utf-8")

    create_visualizations(df)
    return df
