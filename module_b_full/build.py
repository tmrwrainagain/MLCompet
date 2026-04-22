"""
Build helper for Module B executables.
"""

import os
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT       = Path(__file__).parent
EXE_OUTPUT = ROOT

HIDDEN = [
    "psycopg", "psycopg2", "psycopg2.extras",
    "google.genai", "google.genai.types", "google.api_core",
    "google.auth", "google.auth.transport", "google.auth.transport.requests",
    "grpc", "pandas", "numpy",
    "sklearn.cluster", "sklearn.feature_extraction.text", "sklearn.manifold",
    "sklearn.preprocessing", "sklearn.ensemble", "sklearn.inspection", "sklearn.metrics",
    "scipy", "scipy.spatial.distance",
    "PIL", "PIL.Image",
    "pdfplumber", "docx", "openpyxl", "bs4", "lxml", "requests",
    "matplotlib", "matplotlib.pyplot", "seaborn", "tabulate",
    "openai",
    "streamlit", "streamlit.web", "streamlit.web.cli", "streamlit.runtime",
    "altair", "plotly", "plotly.express", "plotly.graph_objects",
    "tornado", "tornado.web", "tornado.ioloop", "click",
    "umap", "umap.umap_",
]

COLLECT = [
    "google.genai", "pdfplumber", "streamlit", "altair", "plotly", "umap",
]

EXCLUDE = [
    "sklearn.tests", "pandas.tests", "numpy.tests", "matplotlib.tests",
    "pytest", "_pytest", "IPython", "torch", "torchvision", "tensorflow", "keras",
]

AGENTS = [
    {"name": "run_module_b",    "script": "run_module_b.py",    "desc": "Module B unified launcher"},
    {"name": "agent_clustering","script": "agent_clustering.py","desc": "Clustering agent"},
    {"name": "agent_dashboard", "script": "agent_dashboard.py", "desc": "Dashboard agent"},
]


def ensure_pyinstaller():
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("  Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller", "--quiet"], check=True)


def build(agent: dict) -> bool:
    name   = agent["name"]
    script = ROOT / agent["script"]
    print(f"\n{'-'*60}\n  Building {name}.exe ({agent['desc']})\n{'-'*60}")

    if not script.exists():
        print(f"  [!] Missing: {script}")
        return False

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", name,
        "--distpath", str(EXE_OUTPUT),
        "--workpath", str(ROOT / "build_tmp" / name),
        "--specpath", str(ROOT / "build_tmp"),
        "--noconfirm",
    ]

    for pkg in COLLECT:
        cmd += ["--collect-all", pkg]
    for hi in HIDDEN:
        cmd += ["--hidden-import", hi]
    for mod in EXCLUDE:
        cmd += ["--exclude-module", mod]

    cmd += [
        "--add-data", f"{ROOT / 'module_A'}{os.pathsep}module_A",
        "--add-data", f"{ROOT / 'module_B'}{os.pathsep}module_B",
    ]

    for env_path in [ROOT / ".env", ROOT / "module_A" / ".env"]:
        if env_path.exists():
            cmd += ["--add-data", f"{env_path}{os.pathsep}module_A"]
            print(f"  Bundling: {env_path.name}")
            break

    cmd.append(str(script))

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  x Build failed: {name}")
        return False

    exe      = EXE_OUTPUT / f"{name}.exe"
    size_mb  = exe.stat().st_size / 1_048_576 if exe.exists() else 0
    print(f"  ok {name}.exe ({size_mb:.0f} MB)")
    return True


def main():
    print("=" * 60)
    print("  BUILD — Module B executables")
    print("=" * 60)

    ensure_pyinstaller()

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt"), "--quiet"],
        check=False,
    )

    results = {a["name"]: build(a) for a in AGENTS}

    print(f"\n{'='*60}\n  RESULTS\n{'='*60}")
    ok = sum(results.values())
    for name, success in results.items():
        print(f"  {'ok' if success else 'x '} {name}.exe")

    print(f"\n  Success: {ok}/{len(AGENTS)}")
    print(f"  Output:  {EXE_OUTPUT.resolve()}")
    if ok == len(AGENTS):
        print("\n  Done. Run with run_module_b.exe")


if __name__ == "__main__":
    main()
