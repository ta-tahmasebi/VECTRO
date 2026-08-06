
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd

from .domain import VehicleState
from .models.base_model import OffloadingAgent
from .tasks import ComputeTask, TaskGenerator

EvaluationProgress = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class EvaluationConfig:

    arrival_rate_per_vehicle_minute: float = 0.5
    execution_durations_s: tuple[float, ...] = (10.0, 30.0)
    reference_capacity_gcycles_s: float = 10.0
    input_mb: float = 2.0
    output_mb: float = 0.2
    deadline_factor: float | None = 2.0
    seed: int = 42
    maximum_tasks: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_tasks is not None and self.maximum_tasks <= 0:
            raise ValueError("Maximum evaluation tasks must be positive.")


@dataclass(frozen=True, slots=True)
class EvaluationSamplingConfig:

    window_count: int = 12
    window_duration_s: float = 600.0

    def __post_init__(self) -> None:
        if self.window_count <= 0:
            raise ValueError("Evaluation window count must be positive.")
        if not np.isfinite(self.window_duration_s) or self.window_duration_s <= 0:
            raise ValueError("Evaluation window duration must be finite and positive.")


@dataclass(frozen=True, slots=True)
class EvaluationSampleSummary:

    mode: str
    source_rows: int
    sampled_rows: int
    source_vehicles: int
    sampled_vehicle_segments: int
    source_start_s: float
    source_end_s: float
    window_duration_s: float | None
    windows: tuple[tuple[float, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source_rows": self.source_rows,
            "sampled_rows": self.sampled_rows,
            "source_vehicles": self.source_vehicles,
            "sampled_vehicle_segments": self.sampled_vehicle_segments,
            "source_start_s": self.source_start_s,
            "source_end_s": self.source_end_s,
            "window_duration_s": self.window_duration_s,
            "windows": [
                {"start_s": start, "end_s": end} for start, end in self.windows
            ],
        }


def representative_temporal_subset(
    trajectories: pd.DataFrame,
    config: EvaluationSamplingConfig,
) -> tuple[pd.DataFrame, EvaluationSampleSummary]:
    if trajectories.empty:
        raise ValueError("Cannot sample an empty evaluation dataset.")
    required = {"vehicle_id", "time_s"}
    missing = sorted(required.difference(trajectories.columns))
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {missing}")

    times = trajectories["time_s"].to_numpy(dtype=float)
    if np.any(~np.isfinite(times)):
        raise ValueError("Evaluation timestamps must be finite.")
    source_start = float(times.min())
    source_end = float(times.max())
    source_span = source_end - source_start
    source_vehicles = int(trajectories["vehicle_id"].nunique())
    requested_span = config.window_count * config.window_duration_s
    if source_span <= requested_span:
        summary = EvaluationSampleSummary(
            mode="full",
            source_rows=len(trajectories),
            sampled_rows=len(trajectories),
            source_vehicles=source_vehicles,
            sampled_vehicle_segments=source_vehicles,
            source_start_s=source_start,
            source_end_s=source_end,
            window_duration_s=None,
            windows=((source_start, source_end),),
        )
        return trajectories.copy(), summary

    unique_times = np.unique(times)
    stratum_edges = np.linspace(
        source_start,
        source_end,
        num=config.window_count + 1,
    )
    half_window = config.window_duration_s / 2.0
    sampled_parts: list[pd.DataFrame] = []
    selected_windows: list[tuple[float, float]] = []
    for index in range(config.window_count):
        left = int(np.searchsorted(unique_times, stratum_edges[index], side="left"))
        right = int(
            np.searchsorted(unique_times, stratum_edges[index + 1], side="right")
        )
        if right <= left:
            continue
        observed = unique_times[left:right]
        center = float(observed[len(observed) // 2])
        start = max(source_start, center - half_window)
        end = min(source_end, start + config.window_duration_s)
        start = max(source_start, end - config.window_duration_s)
        part = trajectories.loc[
            trajectories["time_s"].between(start, end, inclusive="both")
        ].copy()
        if part.empty:
            continue
        part["vehicle_id"] = (
            part["vehicle_id"].astype("string") + f"::evaluation-window-{index:03d}"
        )
        sampled_parts.append(part)
        selected_windows.append((start, end))

    if not sampled_parts:
        raise ValueError("Temporal evaluation sampling selected no observations.")
    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sort_values(["vehicle_id", "time_s"], kind="stable")
    sampled = sampled.drop_duplicates(["vehicle_id", "time_s"], keep="last")
    sampled = sampled.reset_index(drop=True)
    summary = EvaluationSampleSummary(
        mode="temporal_stratified_windows",
        source_rows=len(trajectories),
        sampled_rows=len(sampled),
        source_vehicles=source_vehicles,
        sampled_vehicle_segments=int(sampled["vehicle_id"].nunique()),
        source_start_s=source_start,
        source_end_s=source_end,
        window_duration_s=config.window_duration_s,
        windows=tuple(selected_windows),
    )
    return sampled, summary


@dataclass(frozen=True, slots=True)
class _EvaluationCase:
    task: ComputeTask
    state: VehicleState


class OffloadingEvaluator:

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self.generated_case_count = 0
        self.selected_case_count = 0

    def evaluate(
        self,
        trajectories: pd.DataFrame,
        agents: Sequence[OffloadingAgent],
        *,
        progress: EvaluationProgress | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if trajectories.empty:
            raise ValueError("Evaluation trajectories must not be empty.")
        if not agents:
            raise ValueError("At least one offloading agent is required.")
        for agent in agents:
            if not agent.is_ready:
                raise RuntimeError(f"Agent {agent.algorithm_name} is not ready.")

        ordered = trajectories.sort_values(["vehicle_id", "time_s"], kind="stable")
        truth = _TrajectoryTruth(ordered)
        cases = self._generate_cases(ordered)
        if not cases:
            raise ValueError(
                "No evaluation tasks were generated. Increase the arrival rate "
                "or evaluate a longer trajectory."
            )

        total = len(cases) * len(agents)
        completed = 0
        event_records: list[dict[str, object]] = []
        for agent in agents:
            agent.environment = agent.environment.clone(reset_runtime=True)
            for case in cases:
                prediction_started = perf_counter_ns()
                selection_id = agent.predict(
                    case.state,
                    case.task.nominal_execution_s,
                )
                prediction_overhead_ms = (
                    perf_counter_ns() - prediction_started
                ) / 1_000_000.0
                server, resource = agent.environment.get_selection(selection_id)
                initial_coverage = server.covers(
                    case.state.x_m,
                    case.state.y_m,
                )
                latency = agent.environment.assign_task(
                    case.task,
                    selection_id,
                    case.state,
                )
                future = truth.interpolate(
                    case.task.vehicle_id,
                    latency.completion_time_s,
                )
                truth_available = future is not None
                completion_coverage = (
                    bool(server.covers(*future)) if future is not None else False
                )
                deadline_met = (
                    case.task.deadline_s is None
                    or latency.total_s <= case.task.deadline_s
                )
                success = initial_coverage and completion_coverage
                event_records.append(
                    {
                        "algorithm": agent.algorithm_name,
                        "task_id": case.task.task_id,
                        "vehicle_id": case.task.vehicle_id,
                        "selection_id": selection_id,
                        "server_id": server.server_id,
                        "resource_id": resource.resource_id,
                        "created_at_s": case.task.created_at_s,
                        "task_duration_s": case.task.nominal_execution_s,
                        "latency_s": latency.total_s,
                        "queue_s": latency.queue_s,
                        "queue_depth": latency.queue_depth,
                        "processing_s": latency.processing_s,
                        "energy_j": latency.energy_j,
                        "prediction_overhead_ms": prediction_overhead_ms,
                        "success": success,
                        "initial_coverage": initial_coverage,
                        "completion_coverage": completion_coverage,
                        "deadline_met": deadline_met,
                        "truth_available": truth_available,
                        "server_universe": json.dumps(
                            sorted(
                                server.server_id for server in agent.environment.servers
                            )
                        ),
                        "selection_universe": json.dumps(
                            sorted(agent.environment.selections())
                        ),
                    }
                )
                completed += 1
                if progress is not None:
                    progress(
                        completed,
                        total,
                        f"Evaluating {agent.algorithm_name}",
                    )

        events = pd.DataFrame.from_records(event_records)
        metrics = _aggregate_metrics(events)
        return metrics, events

    def _generate_cases(self, trajectories: pd.DataFrame) -> list[_EvaluationCase]:
        generator = TaskGenerator(
            arrival_rate_per_vehicle_minute=(
                self.config.arrival_rate_per_vehicle_minute
            ),
            execution_durations_s=self.config.execution_durations_s,
            reference_capacity_gcycles_s=(self.config.reference_capacity_gcycles_s),
            input_mb=self.config.input_mb,
            output_mb=self.config.output_mb,
            deadline_factor=self.config.deadline_factor,
            seed=self.config.seed,
        )
        cases: list[_EvaluationCase] = []
        for _, group in trajectories.groupby("vehicle_id", sort=False, observed=True):
            previous_state: VehicleState | None = None
            for row in group.itertuples(index=False):
                state = VehicleState(
                    vehicle_id=str(row.vehicle_id),
                    time_s=float(row.time_s),
                    x_m=float(row.x_m),
                    y_m=float(row.y_m),
                    speed_mps=float(row.speed_mps),
                    angle_deg=float(row.angle_deg),
                )
                if previous_state is None:
                    previous_state = state
                    continue
                interval = state.time_s - previous_state.time_s
                for offset in generator.sample_arrival_offsets(interval):
                    ratio = float(offset / interval)
                    arrival_state = VehicleState(
                        vehicle_id=state.vehicle_id,
                        time_s=previous_state.time_s + float(offset),
                        x_m=previous_state.x_m
                        + ratio * (state.x_m - previous_state.x_m),
                        y_m=previous_state.y_m
                        + ratio * (state.y_m - previous_state.y_m),
                        speed_mps=previous_state.speed_mps
                        + ratio * (state.speed_mps - previous_state.speed_mps),
                        angle_deg=state.angle_deg,
                    )
                    cases.append(
                        _EvaluationCase(generator.create(arrival_state), arrival_state)
                    )
                previous_state = state
        cases.sort(
            key=lambda case: (
                case.task.created_at_s,
                case.task.task_id,
            )
        )
        self.generated_case_count = len(cases)
        cases = self._bounded_case_sample(cases)
        self.selected_case_count = len(cases)
        return cases

    def _bounded_case_sample(
        self,
        cases: list[_EvaluationCase],
    ) -> list[_EvaluationCase]:
        maximum = self.config.maximum_tasks
        if maximum is None or len(cases) <= maximum:
            return cases

        stratum_count = min(12, maximum)
        boundaries = np.linspace(0, len(cases), stratum_count + 1, dtype=int)
        base_target, remainder = divmod(maximum, stratum_count)
        rng = np.random.default_rng(self.config.seed + 1_009)
        selected: list[_EvaluationCase] = []
        for index in range(stratum_count):
            start = int(boundaries[index])
            end = int(boundaries[index + 1])
            target = base_target + (1 if index < remainder else 0)
            if end - start <= target:
                selected.extend(cases[start:end])
                continue
            burst_start = int(rng.integers(start, end - target + 1))
            selected.extend(cases[burst_start : burst_start + target])
        selected.sort(key=lambda case: (case.task.created_at_s, case.task.task_id))
        return selected


class EvaluationReporter:

    def export(
        self,
        metrics: pd.DataFrame,
        events: pd.DataFrame,
        output_directory: str | Path,
        *,
        sampling_summary: Mapping[str, object] | None = None,
    ) -> dict[str, Path]:
        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)

        metrics_path = destination / "metrics.csv"
        events_path = destination / "task_events.csv"
        report_path = destination / "comparison.md"
        success_plot = destination / "success_rate.png"
        latency_plot = destination / "latency_by_duration.png"
        efficiency_plot = destination / "energy_queue_comparison.png"
        sampling_path = destination / "evaluation_sampling.json"
        metrics.to_csv(metrics_path, index=False)
        events.to_csv(events_path, index=False)
        self._plot_success(metrics, success_plot)
        self._plot_latency(metrics, latency_plot)
        self._plot_efficiency(metrics, efficiency_plot)
        report_path.write_text(
            self._build_markdown(metrics, events, sampling_summary=sampling_summary),
            encoding="utf-8",
        )
        outputs = {
            "metrics": metrics_path,
            "events": events_path,
            "report": report_path,
            "success_plot": success_plot,
            "latency_plot": latency_plot,
            "efficiency_plot": efficiency_plot,
        }
        if sampling_summary is not None:
            sampling_path.write_text(
                json.dumps(dict(sampling_summary), indent=2),
                encoding="utf-8",
            )
            outputs["sampling"] = sampling_path
        return outputs

    def _build_markdown(
        self,
        metrics: pd.DataFrame,
        events: pd.DataFrame,
        *,
        sampling_summary: Mapping[str, object] | None = None,
    ) -> str:
        overall = metrics[metrics["scope"] == "overall"].sort_values(
            ["success_rate", "average_latency_s"],
            ascending=[False, True],
        )
        headers = (
            "Algorithm",
            "Tasks (evaluable)",
            "Success rate",
            "Average latency (s)",
            "Energy/task (J)",
            "P95 queue (s)",
            "Queue depth",
            "Predict (ms)",
            "Resource CV",
        )
        rows = [
            (
                str(row.algorithm),
                f"{int(row.tasks)} ({int(row.evaluable_tasks)})",
                (
                    f"{float(row.success_rate):.2%}"
                    if pd.notna(row.success_rate)
                    else "N/A"
                ),
                f"{float(row.average_latency_s):.3f}",
                f"{float(row.average_energy_j):.2f}",
                f"{float(row.p95_queue_s):.3f}",
                f"{float(row.average_queue_depth):.2f}",
                f"{float(row.average_prediction_overhead_ms):.4f}",
                f"{float(row.resource_load_cv):.3f}",
            )
            for row in overall.itertuples(index=False)
        ]
        table = _markdown_table(headers, rows)
        analysis = _automated_analysis(metrics)
        excluded_truth = int((~events["truth_available"].astype(bool)).sum())
        sampling_note = ""
        if sampling_summary is not None:
            sampling_note = (
                "- Evaluation sampling: "
                f"{sampling_summary['mode']} "
                f"({sampling_summary['sampled_rows']} of "
                f"{sampling_summary['source_rows']} trajectory rows).\n"
            )
            if "selected_tasks" in sampling_summary:
                sampling_note += (
                    "- Representative task cases: "
                    f"{sampling_summary['selected_tasks']} of "
                    f"{sampling_summary['generated_tasks_before_cap']} generated.\n"
                )
        return (
            "# Offloading Algorithm Comparison\n\n"
            "Success requires radio coverage both when the task is submitted "
            "and when processing completes. Future positions are used only by "
            "the evaluator as ground truth, never by an agent. Tasks extending "
            "past the recorded trajectory are excluded from the success-rate "
            "denominator.\n\n"
            f"{table}\n\n"
            "## Automated analysis\n\n"
            f"{analysis}\n\n"
            "## Evaluation notes\n\n"
            f"{sampling_note}"
            f"- Total algorithm-task decisions: {len(events):,}\n"
            f"- Decisions without later ground truth: {excluded_truth:,}\n"
            "- Load CV is the coefficient of variation of per-server task counts; "
            "resource CV applies the same calculation to server-resource pairs.\n"
            "- Energy is active processing energy (power multiplied by service time).\n"
            "- Prediction overhead is wall-clock time spent inside predict(); lower "
            "is better, but should be interpreted beside latency and success.\n"
        )

    @staticmethod
    def _plot_success(metrics: pd.DataFrame, path: Path) -> None:
        import matplotlib.pyplot as plt

        overall = metrics[metrics["scope"] == "overall"].sort_values(
            "success_rate", ascending=False
        )
        success_values = overall["success_rate"].fillna(0.0)
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        colors = plt.get_cmap("viridis")(np.linspace(0.18, 0.82, len(overall)))
        bars = axis.bar(
            overall["algorithm"],
            success_values * 100.0,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        axis.set_ylabel("Successful tasks (%)")
        axis.set_xlabel("Offloading algorithm")
        axis.set_title("Mobility-aware task completion success")
        axis.set_ylim(0, max(100.0, float(success_values.max() * 110)))
        axis.grid(axis="y", alpha=0.25, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_latency(metrics: pd.DataFrame, path: Path) -> None:
        import matplotlib.pyplot as plt

        duration = metrics[metrics["scope"] == "duration"].copy()
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        for algorithm, group in duration.groupby("algorithm", sort=True):
            group = group.sort_values("task_duration_s")
            axis.plot(
                group["task_duration_s"],
                group["average_latency_s"],
                marker="o",
                linewidth=2.0,
                label=str(algorithm),
            )
        axis.set_xlabel("Nominal task execution time (s)")
        axis.set_ylabel("Average end-to-end latency (s)")
        axis.set_title("Latency sensitivity to task duration")
        axis.grid(alpha=0.25, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, ncols=2)
        figure.tight_layout()
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)

    @staticmethod
    def _plot_efficiency(metrics: pd.DataFrame, path: Path) -> None:
        import matplotlib.pyplot as plt

        overall = metrics[metrics["scope"] == "overall"].sort_values("average_energy_j")
        positions = np.arange(len(overall))
        figure, energy_axis = plt.subplots(figsize=(10.0, 5.4))
        bars = energy_axis.bar(
            positions,
            overall["average_energy_j"],
            color="#168aad",
            alpha=0.86,
            label="Energy/task",
        )
        energy_axis.set_ylabel("Average processing energy (J)")
        energy_axis.set_xticks(positions, overall["algorithm"], rotation=24)
        energy_axis.grid(axis="y", alpha=0.22, linestyle="--")
        queue_axis = energy_axis.twinx()
        queue_axis.plot(
            positions,
            overall["p95_queue_s"],
            color="#d97706",
            marker="o",
            linewidth=2.0,
            label="P95 queue",
        )
        positive_queue = overall.loc[overall["p95_queue_s"] > 0, "p95_queue_s"]
        queue_label = "P95 queueing delay (s)"
        if not positive_queue.empty and float(positive_queue.max()) > max(
            10.0, float(positive_queue.median()) * 10.0
        ):
            queue_axis.set_yscale("symlog", linthresh=1.0)
            queue_label = "P95 queueing delay (s, symlog)"
        queue_axis.set_ylabel(queue_label)
        energy_axis.set_title("Energy and queueing trade-off")
        energy_axis.spines["top"].set_visible(False)
        queue_axis.spines["top"].set_visible(False)
        handles = [bars, queue_axis.lines[0]]
        energy_axis.legend(handles, ["Energy/task", "P95 queue"], frameon=False)
        figure.tight_layout()
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)


class _TrajectoryTruth:

    def __init__(self, trajectories: pd.DataFrame) -> None:
        self._values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for vehicle_id, group in trajectories.groupby(
            "vehicle_id", sort=False, observed=True
        ):
            ordered = group.sort_values("time_s")
            self._values[str(vehicle_id)] = (
                ordered["time_s"].to_numpy(float),
                ordered["x_m"].to_numpy(float),
                ordered["y_m"].to_numpy(float),
            )

    def interpolate(
        self,
        vehicle_id: str,
        time_s: float,
    ) -> tuple[float, float] | None:
        values = self._values.get(vehicle_id)
        if values is None:
            return None
        times, x_values, y_values = values
        if time_s < times[0] or time_s > times[-1]:
            return None
        return (
            float(np.interp(time_s, times, x_values)),
            float(np.interp(time_s, times, y_values)),
        )


def _aggregate_metrics(events: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for algorithm, group in events.groupby("algorithm", sort=True):
        records.append(_metric_record(str(algorithm), "overall", np.nan, group))
        for duration, duration_group in group.groupby("task_duration_s", sort=True):
            records.append(
                _metric_record(
                    str(algorithm),
                    "duration",
                    float(duration),
                    duration_group,
                )
            )
    return pd.DataFrame.from_records(records)


def _metric_record(
    algorithm: str,
    scope: str,
    duration: float,
    group: pd.DataFrame,
) -> dict[str, object]:
    server_universe = json.loads(str(group["server_universe"].iloc[0]))
    selection_universe = json.loads(str(group["selection_universe"].iloc[0]))
    server_load_series = (
        group["server_id"]
        .value_counts()
        .reindex(
            server_universe,
            fill_value=0,
        )
    )
    resource_load_series = (
        group["selection_id"].value_counts().reindex(selection_universe, fill_value=0)
    )
    server_loads = server_load_series.to_numpy(dtype=float)
    resource_loads = resource_load_series.to_numpy(dtype=float)
    server_load_cv = (
        float(np.std(server_loads) / np.mean(server_loads))
        if server_loads.size and np.mean(server_loads) > 0
        else 0.0
    )
    resource_load_cv = (
        float(np.std(resource_loads) / np.mean(resource_loads))
        if resource_loads.size and np.mean(resource_loads) > 0
        else 0.0
    )
    server_load_distribution = {
        str(key): int(value) for key, value in server_load_series.items()
    }
    resource_load_distribution = {
        str(key): int(value) for key, value in resource_load_series.items()
    }
    evaluable = group[group["truth_available"].astype(bool)]
    success_rate = (
        float(evaluable["success"].mean()) if not evaluable.empty else float("nan")
    )
    return {
        "algorithm": algorithm,
        "scope": scope,
        "task_duration_s": duration,
        "tasks": int(len(group)),
        "evaluable_tasks": int(len(evaluable)),
        "success_rate": success_rate,
        "average_latency_s": float(group["latency_s"].mean()),
        "p95_latency_s": float(group["latency_s"].quantile(0.95)),
        "average_queue_s": float(group["queue_s"].mean()),
        "p95_queue_s": float(group["queue_s"].quantile(0.95)),
        "maximum_queue_s": float(group["queue_s"].max()),
        "average_queue_depth": float(group["queue_depth"].mean()),
        "p95_queue_depth": float(group["queue_depth"].quantile(0.95)),
        "average_energy_j": float(group["energy_j"].mean()),
        "total_energy_j": float(group["energy_j"].sum()),
        "average_prediction_overhead_ms": float(group["prediction_overhead_ms"].mean()),
        "p95_prediction_overhead_ms": float(
            group["prediction_overhead_ms"].quantile(0.95)
        ),
        "deadline_success_rate": float(group["deadline_met"].mean()),
        "server_load_cv": server_load_cv,
        "resource_load_cv": resource_load_cv,
        "server_load_distribution": json.dumps(
            server_load_distribution,
            sort_keys=True,
        ),
        "resource_load_distribution": json.dumps(
            resource_load_distribution,
            sort_keys=True,
        ),
    }


def _automated_analysis(metrics: pd.DataFrame) -> str:
    overall = metrics[metrics["scope"] == "overall"]
    best_latency = overall.loc[overall["average_latency_s"].idxmin()]
    best_balance = overall.loc[overall["resource_load_cv"].idxmin()]
    best_energy = overall.loc[overall["average_energy_j"].idxmin()]
    best_queue = overall.loc[overall["p95_queue_s"].idxmin()]
    statements: list[str] = []
    success_candidates = overall.dropna(subset=["success_rate"])
    if success_candidates.empty:
        statements.append(
            "Overall coverage success is unavailable because no task completed "
            "within the recorded ground-truth horizon."
        )
    else:
        best_success = success_candidates.loc[
            success_candidates["success_rate"].idxmax()
        ]
        statements.append(
            f"**{best_success['algorithm']}** achieved the highest overall "
            f"coverage success ({float(best_success['success_rate']):.2%})."
        )
    statements.extend(
        [
            (
                f"**{best_latency['algorithm']}** produced the lowest average latency "
                f"({float(best_latency['average_latency_s']):.3f} s)."
            ),
            (
                f"**{best_energy['algorithm']}** used the least processing energy "
                f"({float(best_energy['average_energy_j']):.2f} J/task) with a "
                f"coverage success of {float(best_energy['success_rate']):.2%}; "
                "energy must be interpreted beside completed-work quality."
            ),
            (
                f"**{best_queue['algorithm']}** produced the shortest P95 queue "
                f"({float(best_queue['p95_queue_s']):.3f} s)."
            ),
            (
                f"**{best_balance['algorithm']}** produced the most even resource "
                f"load (CV {float(best_balance['resource_load_cv']):.3f}). Uniform "
                "counts alone do not guarantee good coverage, energy, or queueing."
            ),
        ]
    )
    baseline_rows = overall[overall["algorithm"] == "random"]
    intelligent_names = {"kalman", "markov", "gru", "coverage_load"}
    intelligent = overall[overall["algorithm"].isin(intelligent_names)]
    if not baseline_rows.empty and not intelligent.empty:
        baseline = baseline_rows.iloc[0]
        statements.append(
            _baseline_comparison_statement(
                baseline,
                intelligent,
            )
        )
    duration_rows = metrics[metrics["scope"] == "duration"]
    for duration, group in duration_rows.groupby("task_duration_s", sort=True):
        candidates = group.dropna(subset=["success_rate"])
        if candidates.empty:
            continue
        winner = candidates.loc[candidates["success_rate"].idxmax()]
        statements.append(
            f"For {float(duration):g}-second tasks, "
            f"**{winner['algorithm']}** had the best success rate "
            f"({float(winner['success_rate']):.2%})."
        )
    return "\n\n".join(statements)


def _baseline_comparison_statement(
    baseline: pd.Series,
    intelligent: pd.DataFrame,
) -> str:
    comparisons: list[str] = []
    baseline_success = float(baseline["success_rate"])
    if np.isfinite(baseline_success):
        winner = intelligent.loc[intelligent["success_rate"].idxmax()]
        difference = float(winner["success_rate"]) - baseline_success
        comparisons.append(
            f"**{winner['algorithm']}** changed coverage success by "
            f"{difference:+.2%} ({float(winner['success_rate']):.2%} versus "
            f"{baseline_success:.2%})"
        )
    for column, label, unit in (
        ("average_latency_s", "average latency", "s"),
        ("average_energy_j", "energy per task", "J"),
        ("p95_queue_s", "P95 queueing delay", "s"),
    ):
        baseline_value = float(baseline[column])
        winner = intelligent.loc[intelligent[column].idxmin()]
        winner_value = float(winner[column])
        improvement = (
            (baseline_value - winner_value) / baseline_value
            if baseline_value > 0
            else 0.0
        )
        comparisons.append(
            f"**{winner['algorithm']}** changed {label} by "
            f"{-improvement:+.1%} ({winner_value:.3f} {unit} versus "
            f"{baseline_value:.3f} {unit})"
        )
    overhead_winner = intelligent.loc[
        intelligent["average_prediction_overhead_ms"].idxmin()
    ]
    comparisons.append(
        f"**{overhead_winner['algorithm']}** required "
        f"{float(overhead_winner['average_prediction_overhead_ms']):.4f} ms/decision "
        f"versus random's "
        f"{float(baseline['average_prediction_overhead_ms']):.4f} ms"
    )
    return "Against the random baseline, " + "; ".join(comparisons) + "."


def _markdown_table(
    headers: Iterable[str],
    rows: Iterable[Iterable[str]],
) -> str:
    header_values = list(headers)
    lines = [
        "| " + " | ".join(header_values) + " |",
        "| " + " | ".join("---" for _ in header_values) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
