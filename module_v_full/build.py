from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIBS_DIR = ROOT / "libs"
BUILD_DIR = ROOT / "build"
SPEC_DIR = ROOT / ".pyinstaller"
RUNTIME_REQUIREMENTS_FILE = ROOT / "requirements-agent-runtime.txt"
SHARE_DIR = ROOT / "share"

LAUNCHER_NAME = "agent_launcher"
LAUNCHER_SPEC = ROOT / f"{LAUNCHER_NAME}.spec"

EXE_TARGETS = [
    "agent_3_1_training.exe",
    "agent_3_2_continuous_learning.exe",
    "agent_3_3_time_estimation.exe",
    "agent_3_4_recommendations.exe",
    "agent_3_5_report.exe",
    "module_v_all.exe",
]

def log(message: str) -> None:
    print(f"[build] {message}", flush=True)


def run(cmd: list[str]) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        log(f"remove dir {path}")
        try:
            shutil.rmtree(path)
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot remove directory because it is in use: {path}"
            ) from exc


def safe_unlink(path: Path) -> None:
    if path.exists():
        log(f"remove file {path}")
        try:
            path.unlink()
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot remove file because it is in use: {path}"
            ) from exc


def clean() -> None:
    for path in (BUILD_DIR, SPEC_DIR, SHARE_DIR):
        safe_rmtree(path)
    safe_unlink(LAUNCHER_SPEC)
    for exe_name in EXE_TARGETS + [f"{LAUNCHER_NAME}.exe"]:
        safe_unlink(ROOT / exe_name)


def prepare_libs() -> None:
    if not RUNTIME_REQUIREMENTS_FILE.exists():
        raise RuntimeError(f"Missing runtime requirements file: {RUNTIME_REQUIREMENTS_FILE}")

    if LIBS_DIR.exists():
        log("libs already present, skipping")
        return

    LIBS_DIR.mkdir(parents=True, exist_ok=True)
    log(f"install runtime libs from {RUNTIME_REQUIREMENTS_FILE.name}")
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--target",
        str(LIBS_DIR),
        "-r",
        str(RUNTIME_REQUIREMENTS_FILE),
    ])


def prepare_jupyter_templates() -> None:
    source_dir = Path(sys.base_prefix) / "share" / "jupyter" / "nbconvert" / "templates"
    if not source_dir.exists():
        log(f"skip templates copy, source not found: {source_dir}")
        return

    target_dir = SHARE_DIR / "jupyter" / "nbconvert" / "templates"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    log(f"copy nbconvert templates from {source_dir}")
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)


def build_base_launcher() -> Path:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    SPEC_DIR.mkdir(parents=True, exist_ok=True)

    launcher_script = ROOT / "launcher_entry.py"
    run([
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        LAUNCHER_NAME,
        "--distpath",
        str(ROOT),
        "--workpath",
        str(BUILD_DIR),
        "--specpath",
        str(SPEC_DIR),
        str(launcher_script),
    ])
    return ROOT / f"{LAUNCHER_NAME}.exe"


def duplicate_launchers(base_launcher: Path) -> None:
    log("create per-agent exe launchers")
    for exe_name in EXE_TARGETS:
        shutil.copy2(base_launcher, ROOT / exe_name)
        log(f"created {exe_name}")
    safe_unlink(base_launcher)


def main() -> int:
    clean()
    prepare_libs()
    prepare_jupyter_templates()
    base_launcher = build_base_launcher()
    duplicate_launchers(base_launcher)

    log("done. executables in project root:")
    for exe in sorted(ROOT.glob("*.exe")):
        print(f"  - {exe.name}", flush=True)
    log("shared folders:")
    print(f"  - {LIBS_DIR.name}/", flush=True)
    if SHARE_DIR.exists():
        print(f"  - {SHARE_DIR.name}/", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"[build] error: {exc}", flush=True)
        print("[build] close the running EXE or terminal that uses it, then run build.py again.", flush=True)
        raise SystemExit(1)
