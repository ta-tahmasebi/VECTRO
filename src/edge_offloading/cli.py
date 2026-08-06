
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .config import DatasetRegistry
from .data_reader import DataReaderFactory
from .domain import VehicleState
from .edge_env import (
    RESOURCE_PROFILE_LIBRARY,
    EdgeEnvironment,
    MapBounds,
    resolve_resource_profiles,
)
from .evaluator import (
    EvaluationConfig,
    EvaluationReporter,
    EvaluationSamplingConfig,
    OffloadingEvaluator,
    representative_temporal_subset,
)
from .models import ALGORITHM_NAMES, create_agent, load_agent
from .plotter import plot_mobility

console = Console()
LARGE_EVALUATION_DATASETS = frozenset({"roma", "tdrive"})
DEFAULT_EVALUATION_WINDOWS = 12
DEFAULT_EVALUATION_WINDOW_SECONDS = 600.0
DEFAULT_MAXIMUM_EVALUATION_TASKS = 50_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-project",
        description=(
            "Mobility-aware vehicular edge-computing training, inference, "
            "evaluation, and visualization."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/datasets.yaml"),
        help="Dataset registry YAML (default: configs/datasets.yaml).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    placement = subparsers.add_parser(
        "place-servers",
        help="Create an editable multi-resource edge-environment JSON file.",
    )
    placement.add_argument("--dataset", required=True)
    placement.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON file used by train, evaluate, and plot.",
    )
    placement.add_argument("--servers", type=int, default=9)
    placement.add_argument(
        "--placement",
        choices=("grid", "density"),
        default="density",
    )
    placement.add_argument(
        "--resource-profile",
        nargs="+",
        choices=tuple(RESOURCE_PROFILE_LIBRARY),
        default=list(RESOURCE_PROFILE_LIBRARY),
        help="Processing profiles installed on every server.",
    )
    placement.add_argument("--capacity-scale", type=float, default=1.0)
    placement.add_argument("--coverage-overlap", type=float, default=1.10)
    placement.add_argument("--coverage-percentile", type=float, default=95.0)
    placement.add_argument("--uplink-mbps", type=float, default=100.0)
    placement.add_argument("--downlink-mbps", type=float, default=200.0)
    placement.add_argument("--seed", type=int, default=42)
    placement.set_defaults(handler=_command_place_servers)

    train = subparsers.add_parser(
        "train",
        help="Train selected algorithms on the complete dataset.",
    )
    train.add_argument("--dataset", required=True)
    train.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=list(ALGORITHM_NAMES),
        help="One or more algorithms (default: all available policies).",
    )
    train.add_argument(
        "--saved-models",
        type=Path,
        default=Path("saved_models"),
    )
    train.add_argument(
        "--environment-file",
        type=Path,
        required=True,
        help="Versioned server/resource JSON created by place-servers.",
    )
    train.add_argument("--grid-size", type=int, default=24)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--sequence-length", type=int, default=16)
    train.add_argument(
        "--stride",
        type=int,
        help=(
            "GRU window stride; defaults to sequence length. Values cannot "
            "exceed sequence length, preventing temporal gaps."
        ),
    )
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(handler=_command_train)

    infer = subparsers.add_parser(
        "infer",
        help="Recommend one server-resource pair from a current vehicle state.",
    )
    infer.add_argument("--dataset", required=True)
    infer.add_argument("--algorithm", required=True, choices=ALGORITHM_NAMES)
    infer.add_argument(
        "--saved-models",
        type=Path,
        default=Path("saved_models"),
    )
    infer.add_argument("--vehicle-id", default="inference-vehicle")
    infer.add_argument("--time", type=float, default=0.0)
    infer.add_argument("--x", type=float, required=True)
    infer.add_argument("--y", type=float, required=True)
    infer.add_argument("--speed", type=float, required=True)
    infer.add_argument("--angle", type=float, required=True)
    infer.add_argument("--duration", type=float, required=True)
    infer.set_defaults(handler=_command_infer)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained algorithms and export academic reports.",
    )
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_NAMES,
        default=list(ALGORITHM_NAMES),
    )
    evaluate.add_argument(
        "--saved-models",
        type=Path,
        default=Path("saved_models"),
    )
    evaluate.add_argument(
        "--environment-file",
        type=Path,
        required=True,
        help="Must exactly match the environment used for training.",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluation"),
    )
    evaluate.add_argument("--arrival-rate", type=float, default=0.5)
    evaluate.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[10.0, 30.0],
    )
    sampling = evaluate.add_mutually_exclusive_group()
    sampling.add_argument(
        "--evaluation-windows",
        type=_positive_int,
        help=(
            "Number of time-stratified evaluation windows. Roma and T-Drive "
            f"default to {DEFAULT_EVALUATION_WINDOWS}; other datasets use "
            "their complete timelines."
        ),
    )
    sampling.add_argument(
        "--full-evaluation",
        action="store_true",
        help="Disable automatic representative sampling and evaluate all rows.",
    )
    evaluate.add_argument(
        "--evaluation-window-seconds",
        type=_positive_float,
        default=DEFAULT_EVALUATION_WINDOW_SECONDS,
        help="Duration of each representative evaluation window.",
    )
    evaluate.add_argument(
        "--maximum-evaluation-tasks",
        type=_positive_int,
        help=(
            "Maximum representative task cases. Roma and T-Drive default to "
            f"{DEFAULT_MAXIMUM_EVALUATION_TASKS:,}; other datasets are uncapped."
        ),
    )
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.set_defaults(handler=_command_evaluate)

    plot = subparsers.add_parser(
        "plot",
        help="Visualize network structure and mobility patterns.",
    )
    plot.add_argument("--dataset", required=True)
    plot.add_argument(
        "--environment-file",
        type=Path,
        required=True,
        help="Server/resource JSON to overlay on the mobility figure.",
    )
    plot.add_argument(
        "--max-steps",
        type=_positive_int_or_all,
        default=1_500,
        help="Maximum snapshots, or 'all' for the complete timeline.",
    )
    plot.add_argument("--vehicles", type=int, default=25)
    plot.add_argument(
        "--output",
        type=Path,
        default=Path("results/mobility.png"),
    )
    plot.add_argument("--show", action="store_true")
    plot.add_argument("--seed", type=int, default=42)
    plot.set_defaults(handler=_command_plot)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled.[/yellow]")
        return 130
    except Exception as error:
        console.print(
            Panel(
                f"[bold red]{type(error).__name__}[/bold red]\n{error}",
                title="Operation failed",
                border_style="red",
            )
        )
        return 1


def _command_place_servers(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.config)
    dataset = registry.get(args.dataset)
    reader = DataReaderFactory.create(dataset)
    trajectories = _read_with_progress(
        reader.read_all,
        "Loading complete placement dataset",
    )
    profiles = resolve_resource_profiles(
        args.resource_profile,
        capacity_scale=args.capacity_scale,
    )
    metadata = {
        "dataset": dataset.name,
        "seed": args.seed,
        "coverage_overlap": args.coverage_overlap,
        "resource_profiles": list(args.resource_profile),
        "capacity_scale": args.capacity_scale,
    }
    if args.placement == "density":
        environment = EdgeEnvironment.place_density_aware(
            trajectories,
            server_count=args.servers,
            resource_profiles=profiles,
            coverage_percentile=args.coverage_percentile,
            coverage_overlap=args.coverage_overlap,
            uplink_mbps=args.uplink_mbps,
            downlink_mbps=args.downlink_mbps,
            seed=args.seed,
            metadata=metadata,
        )
    else:
        environment = EdgeEnvironment.place_grid(
            MapBounds.from_trajectories(trajectories),
            server_count=args.servers,
            resource_profiles=profiles,
            coverage_overlap=args.coverage_overlap,
            uplink_mbps=args.uplink_mbps,
            downlink_mbps=args.downlink_mbps,
            metadata=metadata,
        )
    environment.save(args.output)
    table = _environment_table(environment)
    console.print(table)
    console.print(
        Panel(
            f"[bold green]Environment saved:[/bold green] {args.output}\n"
            f"Fingerprint: [cyan]{environment.fingerprint()}[/cyan]\n"
            "You may edit positions, radii, bandwidths, capacities, and power "
            "values before training.",
            title="Standalone server placement",
            border_style="green",
        )
    )
    return 0


def _command_train(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.config)
    dataset = registry.get(args.dataset)
    reader = DataReaderFactory.create(dataset)
    console.print(
        Panel(
            f"[bold]{dataset.name}[/bold]\n"
            "The complete configured timeline will be loaded. No temporal "
            "section is truncated.",
            title="Training dataset",
            border_style="cyan",
        )
    )
    trajectories = _read_with_progress(reader.read_all, "Loading full dataset")
    environment = EdgeEnvironment.load(args.environment_file)
    console.print(_dataset_table(trajectories, environment))

    summaries = []
    for algorithm in args.algorithm:
        artifact = args.saved_models / args.dataset / algorithm
        parameters: dict[str, Any] = {"seed": args.seed}
        if algorithm == "markov":
            parameters["grid_size"] = args.grid_size
        elif algorithm == "gru":
            parameters.update(
                {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "sequence_length": args.sequence_length,
                    "stride": args.stride,
                }
            )
        agent = create_agent(
            algorithm,
            environment.clone(),
            artifact,
            **parameters,
        )
        with _progress() as progress:
            task_id = progress.add_task(
                f"Training {algorithm}",
                total=None,
            )

            def callback(
                completed: int,
                total: int,
                description: str,
                current_task_id: TaskID = task_id,
            ) -> None:
                progress.update(
                    current_task_id,
                    completed=completed,
                    total=total,
                    description=description,
                )

            summary = agent.train(trajectories, progress=callback)
        summaries.append(summary)

    table = Table(title="Saved model artifacts", header_style="bold cyan")
    table.add_column("Algorithm")
    table.add_column("Observations", justify="right")
    table.add_column("Vehicles", justify="right")
    table.add_column("Artifact")
    for summary in summaries:
        table.add_row(
            summary.algorithm,
            f"{summary.observations:,}",
            f"{summary.vehicles:,}",
            str(summary.artifact_directory),
        )
    console.print(table)
    return 0


def _command_infer(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.config)
    registry.get(args.dataset)
    artifact = args.saved_models / args.dataset / args.algorithm
    agent = load_agent(artifact)
    state = VehicleState(
        vehicle_id=args.vehicle_id,
        time_s=args.time,
        x_m=args.x,
        y_m=args.y,
        speed_mps=args.speed,
        angle_deg=args.angle,
    )
    selection_id = agent.predict(state, args.duration)
    server, resource = agent.environment.get_selection(selection_id)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Algorithm", agent.algorithm_name)
    table.add_row("Vehicle", state.vehicle_id)
    table.add_row("Task horizon", f"{args.duration:.3f} s")
    table.add_row("Decision", f"[bold green]{selection_id}[/bold green]")
    table.add_row("Server", server.server_id)
    table.add_row("Resource", f"{resource.resource_id} ({resource.resource_type})")
    table.add_row(
        "Server position",
        f"({server.x_m:.2f}, {server.y_m:.2f}) m",
    )
    table.add_row("Coverage radius", f"{server.coverage_radius_m:.2f} m")
    table.add_row(
        "Resource capacity",
        f"{resource.capacity_gcycles_s:.2f} Gcycles/s",
    )
    table.add_row("Active power", f"{resource.active_power_w:.2f} W")
    console.print(Panel(table, title="Offloading decision", border_style="green"))
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.config)
    dataset = registry.get(args.dataset)
    reader = DataReaderFactory.create(dataset)
    trajectories = _read_with_progress(reader.read_all, "Loading evaluation data")
    sampling_summary = None
    window_count = args.evaluation_windows
    maximum_tasks = args.maximum_evaluation_tasks
    if args.full_evaluation:
        window_count = None
        maximum_tasks = None
    if (
        window_count is None
        and not args.full_evaluation
        and args.dataset.lower() in LARGE_EVALUATION_DATASETS
    ):
        window_count = DEFAULT_EVALUATION_WINDOWS
    if (
        maximum_tasks is None
        and not args.full_evaluation
        and args.dataset.lower() in LARGE_EVALUATION_DATASETS
    ):
        maximum_tasks = DEFAULT_MAXIMUM_EVALUATION_TASKS
    if window_count is not None and not args.full_evaluation:
        trajectories, sampling_summary = representative_temporal_subset(
            trajectories,
            EvaluationSamplingConfig(
                window_count=window_count,
                window_duration_s=args.evaluation_window_seconds,
            ),
        )
        console.print(
            Panel(
                f"[bold]{sampling_summary.sampled_rows:,}[/bold] of "
                f"{sampling_summary.source_rows:,} rows across "
                f"{len(sampling_summary.windows)} temporal strata.\n"
                "All active vehicles inside each window are retained to "
                "preserve traffic density and queue load.",
                title="Representative evaluation sample",
                border_style="cyan",
            )
        )
    environment = EdgeEnvironment.load(args.environment_file)
    agents = [
        load_agent(
            args.saved_models / args.dataset / algorithm,
            expected_environment=environment,
        )
        for algorithm in args.algorithm
    ]
    evaluator = OffloadingEvaluator(
        EvaluationConfig(
            arrival_rate_per_vehicle_minute=args.arrival_rate,
            execution_durations_s=tuple(args.durations),
            seed=args.seed,
            maximum_tasks=maximum_tasks,
        )
    )
    with _progress() as progress:
        task_id = progress.add_task("Evaluating algorithms", total=None)

        def callback(completed: int, total: int, description: str) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=total,
                description=description,
            )

        metrics, events = evaluator.evaluate(
            trajectories,
            agents,
            progress=callback,
        )
    sampling_payload = (
        sampling_summary.to_dict() if sampling_summary is not None else None
    )
    if sampling_payload is None and maximum_tasks is not None:
        sampling_payload = {
            "mode": "task_cap_only",
            "source_rows": len(trajectories),
            "sampled_rows": len(trajectories),
        }
    if sampling_payload is not None:
        sampling_payload["generated_tasks_before_cap"] = evaluator.generated_case_count
        sampling_payload["selected_tasks"] = evaluator.selected_case_count
        sampling_payload["maximum_tasks"] = maximum_tasks
    outputs = EvaluationReporter().export(
        metrics,
        events,
        args.output,
        sampling_summary=sampling_payload,
    )
    console.print(_metrics_table(metrics))
    console.print(
        Panel(
            "\n".join(f"[cyan]{name}[/cyan]: {path}" for name, path in outputs.items()),
            title="Evaluation exports",
            border_style="green",
        )
    )
    return 0


def _command_plot(args: argparse.Namespace) -> int:
    registry = DatasetRegistry.from_yaml(args.config)
    dataset = registry.get(args.dataset)
    reader = DataReaderFactory.create(dataset)
    environment = EdgeEnvironment.load(args.environment_file)
    network_value = dataset.options.get("network_file")
    network_path: Path | None = None
    if network_value:
        network_path = Path(str(network_value)).expanduser()
        if not network_path.is_absolute():
            network_path = (args.config.resolve().parent / network_path).resolve()
    with _progress() as progress:
        task_id = progress.add_task("Reading mobility snapshots", total=args.max_steps)

        def callback(
            completed: int,
            total: int | None,
            description: str,
        ) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=total,
                description=description,
            )

        saved = plot_mobility(
            reader,
            max_steps=args.max_steps,
            vehicles_to_show=args.vehicles,
            network_file=network_path,
            environment=environment,
            output_path=args.output,
            show=args.show,
            seed=args.seed,
            progress=callback,
        )
    console.print(f"[bold green]Mobility figure saved:[/bold green] {saved}")
    return 0


def _read_with_progress(
    operation: Callable[..., pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    with _progress() as progress:
        task_id = progress.add_task(label, total=None)

        def callback(
            completed: int,
            total: int | None,
            description: str,
        ) -> None:
            progress.update(
                task_id,
                completed=completed,
                total=total,
                description=description,
            )

        frame = operation(progress=callback)
        progress.update(
            task_id,
            completed=max(1, len(frame)),
            total=max(1, len(frame)),
            description=f"{label}: {len(frame):,} observations",
        )
    return frame


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def _dataset_table(
    trajectories: pd.DataFrame,
    environment: EdgeEnvironment,
) -> Table:
    table = Table(title="Training scope", header_style="bold cyan")
    table.add_column("Property")
    table.add_column("Value", justify="right")
    table.add_row("Observations", f"{len(trajectories):,}")
    table.add_row("Vehicles", f"{trajectories['vehicle_id'].nunique():,}")
    table.add_row(
        "Timeline",
        f"{trajectories['time_s'].min():.1f}–{trajectories['time_s'].max():.1f} s",
    )
    table.add_row("Edge servers", str(len(environment.servers)))
    table.add_row(
        "Processing resources",
        str(sum(len(server.resources) for server in environment.servers)),
    )
    table.add_row("Environment fingerprint", environment.fingerprint()[:16])
    return table


def _environment_table(environment: EdgeEnvironment) -> Table:
    table = Table(title="Edge environment", header_style="bold cyan")
    table.add_column("Server")
    table.add_column("Position")
    table.add_column("Radius", justify="right")
    table.add_column("Resources")
    for server in environment.servers:
        resources = ", ".join(
            f"{item.resource_id}={item.capacity_gcycles_s:g}G/s@"
            f"{item.active_power_w:g}W"
            for item in server.resources
        )
        table.add_row(
            server.server_id,
            f"({server.x_m:.1f}, {server.y_m:.1f})",
            f"{server.coverage_radius_m:.1f} m",
            resources,
        )
    return table


def _metrics_table(metrics: pd.DataFrame) -> Table:
    table = Table(title="Overall comparison", header_style="bold cyan")
    table.add_column("Algorithm")
    table.add_column("Tasks", justify="right")
    table.add_column("Evaluable", justify="right")
    table.add_column("Success", justify="right")
    table.add_column("Avg latency", justify="right")
    table.add_column("Energy/task", justify="right")
    table.add_column("P95 queue", justify="right")
    table.add_column("Predict", justify="right")
    overall = metrics[metrics["scope"] == "overall"].sort_values(
        "success_rate", ascending=False
    )
    for row in overall.itertuples(index=False):
        table.add_row(
            str(row.algorithm),
            f"{int(row.tasks):,}",
            f"{int(row.evaluable_tasks):,}",
            (f"{float(row.success_rate):.2%}" if pd.notna(row.success_rate) else "N/A"),
            f"{float(row.average_latency_s):.3f} s",
            f"{float(row.average_energy_j):.1f} J",
            f"{float(row.p95_queue_s):.2f} s",
            f"{float(row.average_prediction_overhead_ms):.4f} ms",
        )
    return table


def _positive_int_or_all(value: str) -> int | None:
    if value.lower() in {"all", "none", "full"}:
        return None
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer or 'all'.")
    return converted


def _positive_int(value: str) -> int:
    converted = int(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return converted


def _positive_float(value: str) -> float:
    converted = float(value)
    if converted <= 0:
        raise argparse.ArgumentTypeError("Expected a positive number.")
    return converted


if __name__ == "__main__":
    sys.exit(main())
