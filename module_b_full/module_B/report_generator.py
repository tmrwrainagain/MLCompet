"""
Report generator for Module B.

Produces:
  - HTML report
  - Markdown report
"""

import importlib.util as _ilu
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

_b_cfg_spec = _ilu.spec_from_file_location("_mod_b_cfg_rg", Path(__file__).parent / "config.py")
_b_cfg = _ilu.module_from_spec(_b_cfg_spec)
_b_cfg_spec.loader.exec_module(_b_cfg)
REPORTS_DIR = _b_cfg.REPORTS_DIR


def _df_to_md(df: "pd.DataFrame") -> str:
    """Markdown table without tabulate dependency."""
    if df is None or df.empty:
        return "Нет данных."
    headers = " | ".join(str(c) for c in df.columns)
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(str(v) if v is not None else "" for v in row) for row in df.itertuples(index=False, name=None)]
    return "| " + " |\n| ".join([headers, sep] + rows) + " |"


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

_CSS = """
<style>
  body { font-family: 'Segoe UI', sans-serif; margin: 0; color: #2c2c2c; }
  .header { background: #1a237e; color: white; padding: 30px 40px; }
  .header h1 { margin: 0; font-size: 2em; }
  .header p { margin: 6px 0 0; opacity: .8; }
  .container { max-width: 1100px; margin: auto; padding: 30px 40px; }
  h2 { color: #1a237e; border-bottom: 2px solid #3949ab; padding-bottom: 6px; }
  h3 { color: #283593; }
  table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: .9em; }
  th { background: #3949ab; color: white; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) td { background: #f5f6ff; }
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                 gap: 16px; margin: 18px 0; }
  .metric-card { background: #e8eaf6; border-radius: 8px; padding: 18px;
                 text-align: center; }
  .metric-card .value { font-size: 2em; font-weight: bold; color: #1a237e; }
  .metric-card .label { font-size: .85em; color: #555; margin-top: 4px; }
  img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px;
        margin: 10px 0; }
  .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
              gap: 14px; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
           font-size: .8em; margin: 2px; }
  .badge-green  { background: #c8e6c9; color: #1b5e20; }
  .badge-orange { background: #ffe0b2; color: #e65100; }
  .badge-red    { background: #ffcdd2; color: #b71c1c; }
  pre { background: #f5f5f5; padding: 14px; border-radius: 6px;
        font-size: .85em; overflow-x: auto; }
  .conclusion { background: #e3f2fd; border-left: 4px solid #1565c0;
                padding: 16px 20px; border-radius: 4px; margin: 16px 0; }
  footer { background: #1a237e; color: rgba(255,255,255,.7); text-align: center;
           padding: 16px; font-size: .85em; }
</style>
"""


def _img_tag(path: str, alt: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return ""
    import base64
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{alt}" />'


def _df_to_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p><em>Данные отсутствуют.</em></p>"
    return df.to_html(index=False, border=0, classes="", na_rep="—")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def _section_overview(df: pd.DataFrame) -> str:
    total      = len(df)
    n_gen      = int(df["is_generated"].sum()) if "is_generated" in df.columns else 0
    n_orig     = total - n_gen
    n_subjects = df["subject"].nunique() if "subject" in df.columns else 0
    avg_score  = round(df["compliance_score"].dropna().mean(), 2) if "compliance_score" in df.columns else "—"

    return f"""
<h2>1. Обзор системы</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="value">{total}</div><div class="label">Материалов всего</div></div>
  <div class="metric-card"><div class="value">{n_orig}</div><div class="label">Исходных</div></div>
  <div class="metric-card"><div class="value">{n_gen}</div><div class="label">Сгенерированных</div></div>
  <div class="metric-card"><div class="value">{n_subjects}</div><div class="label">Предметов</div></div>
  <div class="metric-card"><div class="value">{avg_score}</div><div class="label">Средняя оценка<br>(0–10)</div></div>
</div>
<h3>Описание реализованного агента</h3>
<p>
  Аналитическая система (Модуль Б) представляет собой интерактивный Streamlit-дашборд,
  автоматически подключающийся к базе данных PostgreSQL, настроенной Модулем А.
  Дашборд поддерживает три роли доступа (<em>admin, teacher, student</em>),
  обеспечивает автоматическое обновление данных и отображает ключевые метрики
  в режиме реального времени.
</p>
<h3>Структура выходных файлов</h3>
<ul>
  <li><code>module_B/reports/report_module_b.html</code> — настоящий отчёт</li>
  <li><code>module_B/reports/quality_*.png</code>          — графики качества кластеризации</li>
</ul>
"""


def _section_clustering(df: pd.DataFrame, quality_results: Dict) -> str:
    html_parts = ["<h2>2. Разметка набора данных</h2>"]

    cluster_meta = [
        (
            "parallel_cluster", "parallel_cluster_label",
            "Параллельное изучение",
            "K-Means на TF-IDF представлении текста материалов + числовые признаки. "
            "Материалы одного кластера принадлежат независимым последовательностям "
            "дисциплин → допускают параллельное изучение.",
        ),
        (
            "sequential_cluster", "sequential_cluster_label",
            "Последовательное изучение",
            "Агломеративная иерархическая кластеризация (связь Ward) на TF-IDF. "
            "Метод сохраняет дендрограмму зависимостей тем, отражая "
            "логическую цепочку изучения материала (в рамках одной или нескольких дисциплин).",
        ),
        (
            "complexity_cluster", "difficulty_label",
            "Сложность освоения",
            "K-Means на числовых признаках: объём текста, средняя длина предложения, "
            "наличие практических заданий, оценка соответствия. "
            "Кластеры размечены как Базовый / Средний / Продвинутый.",
        ),
    ]

    for col, lbl_col, name, rationale in cluster_meta:
        if col not in df.columns:
            continue
        dist = (
            df.groupby(lbl_col if lbl_col in df.columns else col)
              .size()
              .reset_index(name="count")
        )
        html_parts += [
            f"<h3>2.{cluster_meta.index((col,lbl_col,name,rationale))+1}. {name}</h3>",
            f"<p><strong>Метод и обоснование:</strong> {rationale}</p>",
            _df_to_html(dist),
        ]

        # Plots from quality results
        if col in quality_results:
            for plot_path in quality_results[col].get("plots", []):
                html_parts.append(_img_tag(plot_path, f"{name} — визуализация"))

    return "\n".join(html_parts)


def _section_quality(quality_results: Dict) -> str:
    html_parts = ["<h2>3. Оценка качества разметки</h2>"]

    metric_names = {
        "silhouette":        ("Силуэт",            "−1 … 1",   "> 0.5 — хорошо"),
        "davies_bouldin":    ("Дэвис-Болдин",       "≥ 0",      "< 1.0 — хорошо"),
        "calinski_harabasz": ("Калинский-Харабаш",  "> 0",      "чем выше, тем лучше"),
    }

    for key, name in [("parallel_cluster",   "Параллельное изучение"),
                      ("sequential_cluster", "Последовательное изучение"),
                      ("complexity_cluster", "Сложность")]:
        if key not in quality_results:
            continue
        m = quality_results[key].get("metrics", {})
        html_parts.append(f"<h3>{name}</h3>")
        html_parts.append("<table><tr><th>Метрика</th><th>Значение</th>"
                          "<th>Диапазон</th><th>Интерпретация</th></tr>")
        for mkey, (mlabel, mrange, interp) in metric_names.items():
            val = m.get(mkey, float("nan"))
            val_str = f"{val:.4f}" if not pd.isna(val) else "—"
            html_parts.append(
                f"<tr><td>{mlabel}</td><td><strong>{val_str}</strong></td>"
                f"<td>{mrange}</td><td>{interp}</td></tr>"
            )
        html_parts.append("</table>")

    cmp = quality_results.get("method_comparison")
    if cmp is not None and not cmp.empty:
        html_parts.append("<h3>Сравнение методов кластеризации</h3>")
        html_parts.append(_df_to_html(cmp))

    conclusion = quality_results.get("conclusion", "")
    if conclusion:
        html_parts.append(f'<div class="conclusion">{conclusion.replace(chr(10), "<br>")}</div>')

    return "\n".join(html_parts)


def _section_dashboard() -> str:
    return """
<h2>4. Описание дашборда</h2>
<p>
  Интерактивный дашборд реализован на фреймворке <strong>Streamlit</strong>
  с использованием библиотеки <strong>Plotly</strong> для интерактивных графиков.
</p>
<h3>Уровни доступа</h3>
<table>
  <tr><th>Роль</th><th>Доступные разделы</th></tr>
  <tr><td>admin</td><td>Все разделы: обзор, аналитика, кластеризация, требования, данные</td></tr>
  <tr><td>teacher</td><td>Обзор, аналитика, кластеризация, требования</td></tr>
  <tr><td>student</td><td>Обзор</td></tr>
</table>
<h3>Разделы дашборда</h3>
<ol>
  <li><strong>Обзор</strong> — ключевые KPI, распределение по предметам и типам занятий</li>
  <li><strong>Аналитика материалов</strong> — покрытие тем, доля генерации, типы занятий,
      TOP выполняемых/невыполняемых требований</li>
  <li><strong>Методические требования</strong> — обеспеченность по категориям,
      сравнение исходных и генерированных материалов</li>
  <li><strong>Кластеризация</strong> — интерактивные scatter-графики, метрики качества</li>
  <li><strong>Данные</strong> — таблица всех материалов с фильтрацией (только admin)</li>
</ol>
<h3>Обновление данных</h3>
<p>
  Дашборд использует кэш Streamlit с TTL = 60 секунд — при каждом истечении TTL
  данные автоматически перезапрашиваются из PostgreSQL без перезапуска приложения.
</p>
"""


def _markdown_overview(df: pd.DataFrame) -> str:
    total = len(df)
    n_gen = int(df["is_generated"].sum()) if "is_generated" in df.columns else 0
    n_orig = total - n_gen
    n_subjects = df["subject"].nunique() if "subject" in df.columns else 0
    avg_score = round(df["compliance_score"].dropna().mean(), 2) if "compliance_score" in df.columns else "—"
    return "\n".join(
        [
            "# Отчёт — Модуль Б",
            "",
            "## 1. Описание реализованного агента",
            "Аналитическая система реализована как Streamlit-дашборд, работающий с PostgreSQL-базой данных, подготовленной в Модуле А.",
            "Система поддерживает уровни доступа `admin`, `teacher`, `student`, автоматически обновляет данные и показывает ключевые метрики по учебным материалам.",
            "",
            "## 2. Ключевые показатели",
            f"- Материалов всего: {total}",
            f"- Исходных: {n_orig}",
            f"- Сгенерированных: {n_gen}",
            f"- Предметов: {n_subjects}",
            f"- Средняя оценка соответствия: {avg_score}",
            "",
        ]
    )


def _markdown_clustering(df: pd.DataFrame) -> str:
    lines = [
        "## 3. Разметка набора данных",
        "- Параллельное изучение: K-Means на TF-IDF представлении текста и числовых признаках.",
        "- Последовательное изучение: агломеративная иерархическая кластеризация.",
        "- Сложность освоения: K-Means по числовым признакам сложности.",
        "",
        "### Примеры распределений",
    ]

    for col in ["parallel_cluster", "sequential_cluster", "complexity_cluster"]:
        if col in df.columns:
            dist = df[col].value_counts().sort_index().reset_index()
            dist.columns = ["Кластер", "Количество"]
            lines.append(f"#### {col}")
            lines.append(_df_to_md(dist))
            lines.append("")
    return "\n".join(lines)


def _markdown_quality(quality_results: Dict) -> str:
    lines = [
        "## 4. Оценка качества разметки",
        "Используются метрики silhouette, Davies-Bouldin и Calinski-Harabasz, а также визуальный анализ кластеров.",
        "",
    ]
    for key, title in [
        ("parallel_cluster", "Параллельное изучение"),
        ("sequential_cluster", "Последовательное изучение"),
        ("complexity_cluster", "Сложность"),
    ]:
        if key not in quality_results:
            continue
        metrics = quality_results[key].get("metrics", {})
        lines.extend(
            [
                f"### {title}",
                f"- Silhouette: {metrics.get('silhouette', float('nan'))}",
                f"- Davies-Bouldin: {metrics.get('davies_bouldin', float('nan'))}",
                f"- Calinski-Harabasz: {metrics.get('calinski_harabasz', float('nan'))}",
                "",
            ]
        )

    cmp = quality_results.get("method_comparison")
    if cmp is not None and not cmp.empty:
        lines.extend(
            [
                "### Сравнение методов",
                _df_to_md(cmp),
                "",
            ]
        )

    conclusion = quality_results.get("conclusion")
    if conclusion:
        lines.extend(["### Выводы", conclusion, ""])
    return "\n".join(lines)


def _markdown_dashboard() -> str:
    return "\n".join(
        [
            "## 5. Структура выходных файлов",
            "- `module_B/reports/report_module_b.html` — HTML-отчёт.",
            "- `module_B/reports/report_module_b.md` — Markdown-отчёт.",
            "- `module_B/reports/quality_*.png` — графики качества кластеризации.",
            "",
            "## 6. Примеры отображаемых метрик",
            "- Покрытие тем по предметам относительно среднего.",
            "- Доля сгенерированных материалов.",
            "- Распределение по типам занятий.",
            "- Обеспеченность методических требований по категориям.",
            "- TOP наиболее/наименее выполняемых требований.",
            "- Сравнение требований для исходных и сгенерированных материалов.",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, quality_results: Optional[Dict] = None) -> str:
    """Build and save the HTML report. Returns the output path."""
    if quality_results is None:
        quality_results = {}

    ts = datetime.now().strftime("%d.%m.%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт — Модуль Б</title>
{_CSS}
</head>
<body>
<div class="header">
  <h1>Отчёт — Модуль Б: Анализ и визуализация данных</h1>
  <p>Сформирован: {ts}</p>
</div>
<div class="container">
  {_section_overview(df)}
  <hr>
  {_section_clustering(df, quality_results)}
  <hr>
  {_section_quality(quality_results)}
  <hr>
  {_section_dashboard()}
</div>
<footer>Модуль Б — Аналитическая система учебных материалов | Gemini 2.5 Flash API</footer>
</body>
</html>"""

    out_path = REPORTS_DIR / "report_module_b.html"
    out_path.write_text(html, encoding="utf-8")
    md_report = "\n".join(
        [
            _markdown_overview(df),
            _markdown_clustering(df),
            _markdown_quality(quality_results),
            _markdown_dashboard(),
        ]
    )
    md_path = REPORTS_DIR / "report_module_b.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"  Report saved: {out_path}")
    print(f"  Markdown report saved: {md_path}")
    return str(out_path)
