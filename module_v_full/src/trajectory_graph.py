from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go


def export_trajectory_graph(
    gantt_df,
    out_html: Path,
    out_png: Path | None = None,
    title: str = "Граф индивидуальной траектории обучения",
) -> tuple[Path, Path | None]:
    if gantt_df.empty:
        raise ValueError("gantt_df is empty")

    df = gantt_df.copy()
    df["phase_key"] = df["Начало"].astype(str)

    phase_order = sorted(df["phase_key"].unique())
    phase_index = {phase: idx for idx, phase in enumerate(phase_order)}

    unique_streams = list(dict.fromkeys(df["Поток"].tolist()))
    stream_index = {stream: idx for idx, stream in enumerate(unique_streams)}

    nodes: list[dict] = [
        {
            "id": "START",
            "label": "Старт",
            "x": -1,
            "y": 0,
            "color": "#1f2937",
            "size": 26,
            "hover": "Точка старта траектории",
        }
    ]
    node_lookup = {"START": nodes[0]}
    edges: list[tuple[str, str]] = []

    for _, row in df.iterrows():
        node_id = str(row["Предмет"])
        x = phase_index[row["phase_key"]]
        y = -stream_index[row["Поток"]]
        status = str(row.get("Статус", "к изучению"))
        color = "#16a34a" if "изуч" in status else "#2563eb"
        hover = (
            f"<b>{row['Предмет']}</b><br>"
            f"Поток: {row['Поток']}<br>"
            f"Начало: {row['Начало'].date()}<br>"
            f"Конец: {row['Конец'].date()}<br>"
            f"Часов: {row['Часов']}<br>"
            f"Материалов: {row['Материалов']}"
        )
        node_lookup[node_id] = {
            "id": node_id,
            "label": node_id,
            "x": x,
            "y": y,
            "color": color,
            "size": 22,
            "hover": hover,
        }
        nodes.append(node_lookup[node_id])

    grouped = {
        phase: df[df["phase_key"] == phase]["Предмет"].astype(str).tolist()
        for phase in phase_order
    }

    if phase_order:
        for subject in grouped[phase_order[0]]:
            edges.append(("START", subject))

    for idx in range(len(phase_order) - 1):
        current_subjects = grouped[phase_order[idx]]
        next_subjects = grouped[phase_order[idx + 1]]
        for source in current_subjects:
            for target in next_subjects:
                edges.append((source, target))

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source, target in edges:
        edge_x.extend([node_lookup[source]["x"], node_lookup[target]["x"], None])
        edge_y.extend([node_lookup[source]["y"], node_lookup[target]["y"], None])

    node_trace = go.Scatter(
        x=[node["x"] for node in nodes],
        y=[node["y"] for node in nodes],
        mode="markers+text",
        text=[node["label"] for node in nodes],
        textposition="bottom center",
        hoverinfo="text",
        hovertext=[node["hover"] for node in nodes],
        marker={
            "size": [node["size"] for node in nodes],
            "color": [node["color"] for node in nodes],
            "line": {"width": 1.5, "color": "white"},
        },
    )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 2, "color": "#94a3b8"},
        hoverinfo="none",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        template="plotly_white",
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
        height=max(420, 120 * len(unique_streams)),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.08,
        text="Синий — к изучению, зелёный — уже изучено",
        showarrow=False,
        font={"size": 12, "color": "#475569"},
    )

    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")

    if out_png is not None:
        try:
            fig.write_image(str(out_png), width=1400, height=max(700, 150 * len(unique_streams)))
        except Exception:
            try:
                plt.figure(figsize=(14, max(5, 1.8 * len(unique_streams))))
                for source, target in edges:
                    plt.plot(
                        [node_lookup[source]["x"], node_lookup[target]["x"]],
                        [node_lookup[source]["y"], node_lookup[target]["y"]],
                        color="#94a3b8",
                        linewidth=1.5,
                        zorder=1,
                    )
                for node in nodes:
                    plt.scatter(
                        node["x"],
                        node["y"],
                        s=node["size"] * 25,
                        c=node["color"],
                        edgecolors="white",
                        linewidths=1.2,
                        zorder=2,
                    )
                    plt.text(node["x"], node["y"] - 0.12, node["label"], ha="center", va="top", fontsize=10)
                plt.title(title)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_png, dpi=180, bbox_inches="tight")
                plt.close()
            except Exception:
                out_png = None

    return out_html, out_png
