"""Launch both API server and Telegram bot simultaneously.

Usage:
    python run_module_g.py

Both processes run until Ctrl+C.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    print("=" * 60)
    print("Module G — Integration & Interaction")
    print("=" * 60)
    print("Starting API server and Telegram bot...")
    print()

    api_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "agent_api.py")],
        cwd=str(ROOT),
    )
    time.sleep(2)

    bot_proc = subprocess.Popen(
        [sys.executable, str(ROOT / "agent_bot.py")],
        cwd=str(ROOT),
    )

    print(f"  API PID:  {api_proc.pid}")
    print(f"  Bot PID:  {bot_proc.pid}")
    print()
    print("Press Ctrl+C to stop.")

    try:
        api_proc.wait()
        bot_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_proc.terminate()
        bot_proc.terminate()
        api_proc.wait()
        bot_proc.wait()
        print("Done.")


if __name__ == "__main__":
    main()
