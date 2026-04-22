"""
agent_clustering.py
Агент разметки данных — Модуль Б (п. 2.3, 2.4).
Выполняет кластеризацию и оценку качества разметки.
"""

import os
import sys
import traceback
from pathlib import Path

BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE_BASE = Path(getattr(sys, "_MEIPASS", BASE))
os.environ.setdefault("PROJECT_RUNTIME_DIR", str(BASE))

MOD_A = BUNDLE_BASE / "module_A"
MOD_B = BUNDLE_BASE / "module_B"

sys.path.insert(0, str(MOD_A))
sys.path.insert(0, str(MOD_B))


def load_env():
    for env_file in [
        BUNDLE_BASE / "module_A" / ".env",
        BASE / "module_A" / ".env",
        BASE / ".env",
    ]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            return


def pause_on_error():
    if getattr(sys, "frozen", False):
        try:
            input("\n  Произошла ошибка. Нажмите Enter для выхода...")
        except EOFError:
            pass


def main():
    print("=" * 60)
    print("  АГЕНТ РАЗМЕТКИ ДАННЫХ  (Модуль Б — п. 2.3, 2.4)")
    print("=" * 60)
    print("\n  Кластеризация учебных материалов по трём критериям:")
    print("    1. Параллельное изучение (K-Means + TF-IDF)")
    print("    2. Последовательное изучение (Агломеративная кластеризация)")
    print("    3. Сложность освоения (K-Means на числовых признаках)")
    print()

    load_env()

    if not os.environ.get("GEMINI_API_KEY"):
        key = input("  Введите GEMINI_API_KEY (или Enter, если используете OpenAI): ").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key

    if not os.environ.get("OPENAI_API_KEY"):
        key = input("  Введите OPENAI_API_KEY (или Enter для пропуска): ").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

    # ── Кластеризация ────────────────────────────────────────────────
    print("\n[1/3] Запуск кластеризации...")
    from clustering import run_clustering
    df = run_clustering()

    if df.empty:
        print("  Нет данных. Сначала запустите agent_loader.exe")
        input("\n  Нажмите Enter для выхода...")
        return

    print(f"  Размечено материалов: {len(df)}")

    # ── Оценка качества ──────────────────────────────────────────────
    print("\n[2/3] Оценка качества кластеризации...")
    from quality_metrics import evaluate_clustering
    quality = evaluate_clustering(df)

    print("\n  Метрики качества кластеризации:")
    for key in ("parallel_cluster", "sequential_cluster", "complexity_cluster"):
        if key not in quality:
            continue
        m    = quality[key].get("metrics", {})
        name = quality[key].get("name", key)
        sil  = m.get("silhouette", float("nan"))
        db   = m.get("davies_bouldin", float("nan"))
        ch   = m.get("calinski_harabasz", float("nan"))
        print(f"\n  [{name}]")
        print(f"    Силуэт:            {sil:.4f}" if sil == sil else "    Силуэт: —")
        print(f"    Дэвис-Болдин:      {db:.4f}"  if db  == db  else "    Дэвис-Болдин: —")
        print(f"    Калинский-Харабаш: {ch:.2f}"  if ch  == ch  else "    Калинский-Харабаш: —")

    plot_count = sum(len(item.get("plots", [])) for item in quality.values() if isinstance(item, dict))
    if plot_count:
        print(f"\n  Графики сохранены в: {MOD_B / 'reports'}")

    # ── HTML отчёт ───────────────────────────────────────────────────
    print("\n[3/3] Генерация HTML-отчёта...")
    from report_generator import generate_report
    path = generate_report(df, quality)
    print(f"  Отчёт: {path}")

    # ── Обоснование метода разметки ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ОБОСНОВАНИЕ МЕТОДА РАЗМЕТКИ И ГРУППИРОВКИ")
    print("=" * 60)
    print()
    print("  1. Параллельное изучение  →  K-Means + TF-IDF")
    print("     Метод: K-Means на объединённой матрице TF-IDF (текст)")
    print("     и нормированных числовых признаков.")
    print("     Обоснование: материалы из независимых тем образуют")
    print("     сферические кластеры в пространстве содержания —")
    print("     K-Means эффективно разделяет их без иерархии.")
    print()
    print("  2. Последовательное изучение  →  Агломеративная кластеризация")
    print("     Метод: иерархическая кластеризация (Ward) по TF-IDF.")
    print("     Обоснование: Ward-связь минимизирует внутрикластерную")
    print("     дисперсию и выявляет цепочки тем с нарастающей")
    print("     сложностью без задания числа кластеров заранее.")
    print()
    print("  3. Сложность освоения  →  K-Means на числовых признаках")
    print("     Признаки: объём текста, средняя длина предложения,")
    print("     наличие контрольных вопросов, оценка методики.")
    print("     Обоснование: числовые признаки прямо отражают сложность;")
    print("     3 кластера (Базовый / Средний / Продвинутый) дают")
    print("     интерпретируемое разбиение, удобное для учебного плана.")
    print("=" * 60)

    print("\n  Следующий шаг: запустите agent_dashboard.exe для")
    print("  просмотра результатов в интерактивном дашборде.")
    print("=" * 60)
    input("  Нажмите Enter для выхода...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Прервано пользователем.")
        pause_on_error()
        raise
    except Exception:
        print("\n  Критическая ошибка в agent_clustering.exe:")
        traceback.print_exc()
        pause_on_error()
        sys.exit(1)
