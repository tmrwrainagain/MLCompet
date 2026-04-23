from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.exporters import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from src.config import ROOT as CONFIG_ROOT


def get_runtime_root() -> Path:
    cwd_root = Path.cwd()
    if (cwd_root / "notebooks").exists():
        return cwd_root
    return CONFIG_ROOT


ROOT = get_runtime_root()
NOTEBOOKS_DIR = ROOT / "notebooks"
EXEC_DIR = ROOT / ".nb-exec"
EXEC_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AgentSpec:
    key: str
    notebook: str
    html_name: str
    title: str


AGENTS: dict[str, AgentSpec] = {
    "3.1": AgentSpec("3.1", "3.1_model_training.ipynb", "3.1_model_training.html", "Обучение модели"),
    "3.2": AgentSpec("3.2", "3.2_continuous_learning.ipynb", "3.2_continuous_learning.html", "Непрерывное обучение"),
    "3.3": AgentSpec("3.3", "3.3_time_estimation.ipynb", "3.3_time_estimation.html", "Оценка времени"),
    "3.4": AgentSpec("3.4", "3.4_recommendations.ipynb", "3.4_recommendations.html", "Рекомендации"),
    "3.5": AgentSpec("3.5", "3.5_report.ipynb", "3.5_report.html", "Итоговый отчёт"),
}

PIPELINE_ORDER = ["3.1", "3.2", "3.3", "3.4", "3.5"]


def _log(message: str) -> None:
    print(message, flush=True)


def _fallback_notebook_html(nb) -> str:
    parts = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Notebook Export</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1f2937;}",
        ".cell{border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin:16px 0;background:#fff;}",
        ".markdown{line-height:1.6;white-space:pre-wrap;}",
        ".code{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;white-space:pre-wrap;}",
        ".output{margin-top:12px;padding:12px;background:#f8fafc;border-radius:8px;overflow:auto;}",
        "img{max-width:100%;height:auto;}",
        "</style></head><body>",
    ]

    for index, cell in enumerate(nb.cells, start=1):
        parts.append(f"<section class='cell'><div><b>Cell {index}</b></div>")
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            parts.append(f"<div class='markdown'>{html.escape(source)}</div>")
        elif cell.get("cell_type") == "code":
            parts.append(f"<pre class='code'>{html.escape(source)}</pre>")
            for output in cell.get("outputs", []):
                parts.append("<div class='output'>")
                data = output.get("data", {})
                if "text/html" in data:
                    html_data = data["text/html"]
                    if isinstance(html_data, list):
                        html_data = "".join(html_data)
                    parts.append(str(html_data))
                elif "image/png" in data:
                    parts.append(f"<img src='data:image/png;base64,{data['image/png']}' />")
                elif "text/plain" in data:
                    text_data = data["text/plain"]
                    if isinstance(text_data, list):
                        text_data = "".join(text_data)
                    parts.append(f"<pre>{html.escape(str(text_data))}</pre>")
                elif output.get("output_type") == "stream":
                    text_data = output.get("text", "")
                    if isinstance(text_data, list):
                        text_data = "".join(text_data)
                    parts.append(f"<pre>{html.escape(str(text_data))}</pre>")
                elif output.get("output_type") == "error":
                    traceback = "\n".join(output.get("traceback", []))
                    parts.append(f"<pre>{html.escape(traceback)}</pre>")
                parts.append("</div>")
        else:
            parts.append(f"<pre>{html.escape(source)}</pre>")
        parts.append("</section>")

    parts.append("</body></html>")
    return "".join(parts)


def _nbconvert_template_dirs() -> list[Path]:
    candidates = [
        ROOT / "share" / "jupyter" / "nbconvert" / "templates",
        ROOT / "libs" / "share" / "jupyter" / "nbconvert" / "templates",
        Path(sys.base_prefix) / "share" / "jupyter" / "nbconvert" / "templates",
        Path(sys.prefix) / "share" / "jupyter" / "nbconvert" / "templates",
    ]
    result: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path.exists() and path not in seen:
            result.append(path)
            seen.add(path)
    return result


@contextmanager
def _current_runtime_kernel() -> str:
    libs_path = ROOT / "libs"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(ROOT)]
    if libs_path.exists():
        pythonpath_parts.append(str(libs_path))
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    with tempfile.TemporaryDirectory(prefix="module_v_kernel_") as tmpdir:
        kernel_dir = Path(tmpdir) / "kernels" / "module_v_runtime"
        kernel_dir.mkdir(parents=True, exist_ok=True)

        bootstrap = (
            "import sys, runpy; "
            f"sys.path[0:0] = {pythonpath_parts!r}; "
            "runpy.run_module('ipykernel_launcher', run_name='__main__')"
        )

        kernel_spec = {
            "argv": [sys.executable, "-c", bootstrap, "-f", "{connection_file}"],
            "display_name": "Python (Module V runtime)",
            "language": "python",
            "env": {
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": os.pathsep.join(pythonpath_parts),
            },
        }
        (kernel_dir / "kernel.json").write_text(
            json.dumps(kernel_spec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        previous_jupyter_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = tmpdir
        try:
            yield "module_v_runtime"
        finally:
            if previous_jupyter_path is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous_jupyter_path


def execute_notebook(notebook_path: Path, executed_path: Path, timeout: int = 7200) -> Path:
    _log(f"[run] execute {notebook_path.name}")
    with notebook_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    with _current_runtime_kernel() as kernel_name:
        ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
        ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})

    with executed_path.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)

    _log(f"[done] executed -> {executed_path}")
    return executed_path


def export_executed_html(executed_path: Path, html_path: Path) -> Path:
    _log(f"[run] export html {html_path.name}")
    with executed_path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)

    try:
        exporter = HTMLExporter(
            template_name="classic",
            extra_template_basedirs=[str(path) for path in _nbconvert_template_dirs()],
        )
        body, _ = exporter.from_notebook_node(nb)
    except Exception as exc:
        _log(f"[warn] nbconvert html export failed, using fallback renderer: {exc}")
        body = _fallback_notebook_html(nb)

    html_path.write_text(body, encoding="utf-8")
    _log(f"[done] html -> {html_path}")
    return html_path


def run_agent(agent_key: str) -> tuple[Path, Path]:
    spec = AGENTS[agent_key]
    notebook_path = NOTEBOOKS_DIR / spec.notebook
    executed_path = EXEC_DIR / f"{Path(spec.notebook).stem}.exe_runner.executed.ipynb"
    html_path = ROOT / spec.html_name

    _log(f"[agent] {spec.key} | {spec.title}")
    executed = execute_notebook(notebook_path, executed_path)
    html = export_executed_html(executed, html_path)
    _log(f"[agent] completed {spec.key}")
    return executed, html


def run_all_agents() -> None:
    _log("[pipeline] start full Module V pipeline")
    for key in PIPELINE_ORDER:
        run_agent(key)
    _log("[pipeline] all agents completed")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Module V notebook agents.")
    parser.add_argument(
        "target",
        choices=[*PIPELINE_ORDER, "all"],
        help="Agent to run: 3.1, 3.2, 3.3, 3.4, 3.5 or all",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.target == "all":
        run_all_agents()
    else:
        run_agent(args.target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
