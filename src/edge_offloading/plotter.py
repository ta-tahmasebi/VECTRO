
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data_reader import MobilityReader
from .edge_env import EdgeEnvironment

PlotProgress = Callable[[int, int | None, str], None]


@dataclass(slots=True)
class _PathAccumulator:
    x: list[float]
    y: list[float]
    observations: int = 0

    def add(self, x_m: float, y_m: float, maximum_points: int) -> None:
        self.observations += 1
        self.x.append(x_m)
        self.y.append(y_m)
        if len(self.x) > maximum_points * 2:
            self.x = self.x[::2]
            self.y = self.y[::2]


def plot_mobility(
    reader: MobilityReader,
    *,
    max_steps: int | None = 1_500,
    vehicles_to_show: int = 25,
    minimum_observations: int = 10,
    maximum_points_per_vehicle: int = 2_000,
    network_file: str | Path | None = None,
    environment: EdgeEnvironment | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
    seed: int = 42,
    progress: PlotProgress | None = None,
) -> Path | None:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive or None.")
    if vehicles_to_show <= 0 or maximum_points_per_vehicle < 2:
        raise ValueError("Plot sampling parameters must be positive.")

    paths: dict[str, _PathAccumulator] = {}
    for snapshot in reader.iter_snapshots(
        max_steps=max_steps,
        progress=progress,
    ):
        for identifier, state in snapshot.vehicles.items():
            accumulator = paths.setdefault(
                identifier,
                _PathAccumulator([], []),
            )
            accumulator.add(
                state.x_m,
                state.y_m,
                maximum_points_per_vehicle,
            )

    valid = {
        identifier: path
        for identifier, path in paths.items()
        if path.observations >= minimum_observations
    }
    if not valid:
        raise ValueError(
            "No vehicle has enough observations for visualization. "
            "Increase max_steps or lower minimum_observations."
        )

    rng = np.random.default_rng(seed)
    identifiers = np.asarray(sorted(valid), dtype=object)
    count = min(vehicles_to_show, len(identifiers))
    selected = rng.choice(identifiers, size=count, replace=False)

    if not show:
        import matplotlib

        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    with plt.style.context("seaborn-v0_8-whitegrid"):
        figure, (trajectory_axis, network_axis) = plt.subplots(
            1,
            2,
            figsize=(16.0, 7.2),
            sharex=True,
            sharey=True,
        )
        colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.95, count))
        for color, identifier in zip(colors, selected, strict=True):
            path = valid[str(identifier)]
            trajectory_axis.plot(
                path.x,
                path.y,
                color=color,
                linewidth=1.4,
                alpha=0.82,
            )
            trajectory_axis.scatter(
                path.x[0],
                path.y[0],
                color=color,
                marker="o",
                s=24,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
            trajectory_axis.scatter(
                path.x[-1],
                path.y[-1],
                color=color,
                marker="s",
                s=24,
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        trajectory_axis.scatter(
            [],
            [],
            color="gray",
            marker="o",
            label="Start",
        )
        trajectory_axis.scatter(
            [],
            [],
            color="gray",
            marker="s",
            label="End",
        )
        trajectory_axis.set_title(
            f"Representative trajectories (n={count})",
            fontweight="semibold",
        )
        trajectory_axis.legend(frameon=False, loc="best")

        network_drawn = False
        if network_file is not None:
            network_drawn = _draw_sumo_network(network_axis, Path(network_file))
        all_x = np.concatenate(
            [np.asarray(path.x, dtype=float) for path in valid.values()]
        )
        all_y = np.concatenate(
            [np.asarray(path.y, dtype=float) for path in valid.values()]
        )
        density = network_axis.hexbin(
            all_x,
            all_y,
            gridsize=150,
            mincnt=1,
            bins="log",
            cmap="magma",
            alpha=0.72 if network_drawn else 0.9,
            linewidths=0,
        )
        colorbar = figure.colorbar(
            density,
            ax=network_axis,
            fraction=0.046,
            pad=0.04,
        )
        colorbar.set_label("Log observation density")
        network_axis.set_title(
            (
                "Street network and mobility density"
                if network_drawn
                else "Inferred network from mobility density"
            ),
            fontweight="semibold",
        )

        if environment is not None:
            _draw_environment(trajectory_axis, environment, annotate=False)
            _draw_environment(network_axis, environment, annotate=True)

        for axis in (trajectory_axis, network_axis):
            axis.set_xlabel("Local east coordinate, x (m)")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.18, linestyle="--")
            axis.spines[["top", "right"]].set_visible(False)
        trajectory_axis.set_ylabel("Local north coordinate, y (m)")
        figure.suptitle(
            f"{reader.config.name}: mobility structure",
            fontsize=15,
            fontweight="bold",
        )
        figure.tight_layout()

        saved: Path | None = None
        if output_path is not None:
            saved = Path(output_path)
            saved.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(saved, dpi=260, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(figure)
    return saved


def _draw_environment(
    axis: Any,
    environment: EdgeEnvironment,
    *,
    annotate: bool,
) -> None:
    from matplotlib.patches import Circle

    for server in environment.servers:
        coverage = Circle(
            (server.x_m, server.y_m),
            server.coverage_radius_m,
            facecolor="#159e9c",
            edgecolor="#006d77",
            alpha=0.08,
            linewidth=0.8,
            zorder=2,
        )
        axis.add_patch(coverage)
        axis.scatter(
            server.x_m,
            server.y_m,
            marker="^",
            s=44,
            color="#006d77",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        if annotate:
            total_capacity = sum(
                resource.capacity_gcycles_s for resource in server.resources
            )
            axis.annotate(
                f"{server.server_id}\n{len(server.resources)}r/{total_capacity:g}G/s",
                (server.x_m, server.y_m),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.5,
                color="#073b4c",
                zorder=6,
            )


def _draw_sumo_network(axis: Any, network_file: Path) -> bool:
    if not network_file.is_file():
        raise FileNotFoundError(f"SUMO network file not found: {network_file}")
    drew_any = False
    try:
        iterator = ET.iterparse(network_file, events=("end",))
        for _, element in iterator:
            if element.tag.rsplit("}", 1)[-1] != "edge":
                continue
            if element.attrib.get("function") == "internal":
                element.clear()
                continue
            lane = next(
                (
                    child
                    for child in element
                    if child.tag.rsplit("}", 1)[-1] == "lane"
                    and child.attrib.get("shape")
                ),
                None,
            )
            if lane is not None:
                points = [
                    tuple(float(value) for value in pair.split(",")[:2])
                    for pair in lane.attrib["shape"].split()
                ]
                if len(points) >= 2:
                    x_values, y_values = zip(*points, strict=True)
                    axis.plot(
                        x_values,
                        y_values,
                        color="#73808c",
                        linewidth=0.35,
                        alpha=0.45,
                        zorder=0,
                    )
                    drew_any = True
            element.clear()
    except ET.ParseError as error:
        raise ValueError(f"Malformed SUMO network XML: {network_file}") from error
    return drew_any
