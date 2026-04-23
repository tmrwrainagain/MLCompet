from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ENTRYPOINTS = {
    "agent_3_1_training": "agent_3_1.py",
    "agent_3_2_continuous_learning": "agent_3_2.py",
    "agent_3_3_time_estimation": "agent_3_3.py",
    "agent_3_4_recommendations": "agent_3_4.py",
    "agent_3_5_report": "agent_3_5.py",
    "module_v_all": "module_v_all.py",
}


def main() -> int:
    exe_path = Path(sys.executable).resolve()
    project_root = exe_path.parent
    exe_stem = exe_path.stem

    script_name = ENTRYPOINTS.get(exe_stem)

    if script_name is None:
        print(f"[launcher] unknown launcher name: {exe_stem}", flush=True)
        return 1

    script_path = project_root / script_name
    libs_path = project_root / "libs"

    if not script_path.exists():
        print(f"[launcher] script not found: {script_path}", flush=True)
        return 1
    if not libs_path.exists():
        print(f"[launcher] libs not found: {libs_path}", flush=True)
        return 1

    python_candidates = [
        project_root / "run" / "pythonportable" / "python.exe",
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / "venv" / "Scripts" / "python.exe",
    ]

    python_cmd: list[str] | None = None
    for candidate in python_candidates:
        if candidate.exists():
            python_cmd = [str(candidate)]
            break

    if python_cmd is None:
        for candidate in (["py", "-3"], ["python"]):
            try:
                probe = subprocess.run(
                    [*candidate, "--version"],
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if probe.returncode == 0:
                    python_cmd = candidate
                    break
            except Exception:
                continue

    if python_cmd is None:
        print("[launcher] python runtime not found. Install Python or create .venv.", flush=True)
        return 1

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(libs_path), str(project_root)])

    print(f"[launcher] start {exe_stem}", flush=True)
    print(f"[launcher] python={' '.join(python_cmd)}", flush=True)
    print(f"[launcher] script={script_path.name}", flush=True)
    print(f"[launcher] libs={libs_path}", flush=True)

    bootstrap = (
        f"import sys, runpy; "
        f"sys.path[0:0] = [{str(project_root)!r}, {str(libs_path)!r}]; "
        f"runpy.run_path({str(script_path)!r}, run_name='__main__')"
    )
    completed = subprocess.run(
        [*python_cmd, "-c", bootstrap],
        cwd=project_root,
        env=env,
    )

    if completed.returncode == 0:
        print("\n[launcher] агент завершил работу успешно.", flush=True)
    else:
        print("\n[launcher] агент завершил работу.", flush=True)

    input("Нажмите Enter для закрытия...")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
