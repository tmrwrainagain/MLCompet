"""
Module B unified launcher.
Runs: clustering → quality evaluation → report → dashboard.
"""

import os
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path


RUNTIME_BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE_BASE  = Path(getattr(sys, "_MEIPASS", RUNTIME_BASE))
os.environ.setdefault("PROJECT_RUNTIME_DIR", str(RUNTIME_BASE))

MOD_A = BUNDLE_BASE / "module_A"
MOD_B = BUNDLE_BASE / "module_B"

sys.path.insert(0, str(MOD_A))
sys.path.insert(0, str(MOD_B))

SEP = "=" * 65


def banner(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def load_env():
    for env_file in [
        BUNDLE_BASE / "module_A" / ".env",
        RUNTIME_BASE / "module_A" / ".env",
        RUNTIME_BASE / ".env",
    ]:
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            print(f"  Загружен .env: {env_file}")
            return
    print("  .env не найден — используются системные переменные среды.")


def pause_on_error():
    if getattr(sys, "frozen", False):
        try:
            input("\n  Произошла ошибка. Нажмите Enter для выхода...")
        except EOFError:
            pass


def wait_and_open_browser(url: str, port: int = 8501, timeout: float = 30.0):
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.create_connection(("localhost", port), timeout=1)
            sock.close()
            break
        except OSError:
            time.sleep(0.5)
    print(f"\n  Открываем браузер: {url}")
    webbrowser.open(url)


def run_clustering_pipeline():
    from clustering import run_clustering
    from quality_metrics import evaluate_clustering
    from report_generator import generate_report

    print("\n[1/3] Кластеризация учебных материалов...")
    df = run_clustering()

    if df.empty:
        print("  Нет данных в базе. Сначала запустите Module A (agent_loader + agent_moderator).")
        return False

    print(f"  Размечено материалов: {len(df)}")

    print("\n[2/3] Оценка качества кластеризации...")
    quality = evaluate_clustering(df)

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

    print("\n[3/3] Генерация отчёта Модуля Б...")
    path = generate_report(df, quality)
    print(f"  Отчёт: {path}")

    print()
    print("=" * 65)
    print("  ОБОСНОВАНИЕ МЕТОДА РАЗМЕТКИ И ГРУППИРОВКИ")
    print("=" * 65)
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
    print("=" * 65)
    return True


def launch_dashboard():
    dashboard = MOD_B / "dashboard.py"
    url  = "http://localhost:8501"
    port = 8501

    print(f"\n  Запуск дашборда: {url}")
    print("  Логины: admin/admin123  teacher/teacher123  student/student123")
    print("  Нажмите Ctrl+C для остановки.\n")

    threading.Thread(
        target=wait_and_open_browser,
        args=(url, port, 30.0),
        daemon=True,
    ).start()

    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", str(dashboard),
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
    ]
    raise SystemExit(stcli.main())


def main():
    banner("МОДУЛЬ Б — АНАЛИТИЧЕСКАЯ СИСТЕМА УЧЕБНЫХ МАТЕРИАЛОВ")
    print("  Шаги: кластеризация → оценка качества → отчёт → дашборд")

    load_env()

    if not os.environ.get("GEMINI_API_KEY"):
        print("\n  [!] Переменная GEMINI_API_KEY не задана.")
        key = input("  Введите ключ Gemini API: ").strip()
        if key:
            os.environ["GEMINI_API_KEY"] = key
        else:
            print("  Ключ не введен. Выход.")
            input("  Нажмите Enter...")
            sys.exit(1)

    try:
        ok = run_clustering_pipeline()
    except KeyboardInterrupt:
        print("\n  Прервано пользователем.")
        ok = False
    except Exception as exc:
        print(f"\n  Ошибка кластеризации: {exc}")
        traceback.print_exc()
        ok = False

    if not ok:
        input("\n  Нажмите Enter для выхода...")
        sys.exit(1)

    banner("КЛАСТЕРИЗАЦИЯ ЗАВЕРШЕНА — ЗАПУСК ДАШБОРДА")
    launch_dashboard()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Процесс остановлен пользователем.")
        pause_on_error()
        raise
    except Exception:
        print("\n  Критическая ошибка:")
        traceback.print_exc()
        pause_on_error()
        sys.exit(1)
