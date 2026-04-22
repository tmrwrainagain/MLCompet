"""
Dashboard agent for Module B.
Starts the Streamlit dashboard and opens the browser automatically.
"""

import os
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path


BASE = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
BUNDLE_BASE = Path(getattr(sys, "_MEIPASS", BASE))
os.environ.setdefault("PROJECT_RUNTIME_DIR", str(BASE))

MOD_A = BUNDLE_BASE / "module_A"
MOD_B = BUNDLE_BASE / "module_B"


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


def pause_on_error():
    if getattr(sys, "frozen", False):
        try:
            input("\n  Произошла ошибка. Нажмите Enter для выхода...")
        except EOFError:
            pass


def main():
    print("=" * 60)
    print("  АГЕНТ АНАЛИТИЧЕСКОЙ СИСТЕМЫ  (Модуль Б — п. 2.1)")
    print("=" * 60)
    print("\n  Дашборд: анализ учебных материалов")
    print("  Уровни доступа: admin / teacher / student")
    print("  Логины: admin/admin123  teacher/teacher123  student/student123\n")

    load_env()

    dashboard = MOD_B / "dashboard.py"
    if not dashboard.exists():
        print(f"  [!] Файл дашборда не найден: {dashboard}")
        print("  Убедитесь, что папка module_B находится рядом с exe-файлом.")
        input("\n  Нажмите Enter для выхода...")
        sys.exit(1)

    url = "http://localhost:8501"
    port = 8501

    threading.Thread(
        target=wait_and_open_browser,
        args=(url, port, 30.0),
        daemon=True,
    ).start()

    print(f"  Запуск дашборда на {url} ...")
    print("  Нажмите Ctrl+C для остановки.\n")

    # Disable Streamlit's development mode — it blocks explicit --server.port usage
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(dashboard),
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light",
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Дашборд остановлен пользователем.")
        pause_on_error()
        raise
    except Exception:
        print("\n  Критическая ошибка в agent_dashboard.exe:")
        traceback.print_exc()
        pause_on_error()
        sys.exit(1)
