"""Entry point: start the Telegram bot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    from module_G.bot import run
    run()
