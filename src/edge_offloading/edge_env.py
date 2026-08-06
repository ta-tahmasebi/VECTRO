
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import ceil, hypot, isfinite, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .domain import VehicleState
from .tasks import ComputeTask

ENVIRONMENT_SCHEMA_VERSION = 2
SELECTION_SEPARATOR = ":"


def _validate_resource_values(
    resource_id: str,
    resource_type: str,
    capacity_gcycles_s: float,
    idle_power_w: float,
    active_power_w: float,
) -> None:
    if not resource_id or SELECTION_SEPARATOR in resource_id:
        raise ValueError("resource_id must be non-empty and cannot contain ':'.")
    if not resource_type:
        raise ValueError("resource_type must not be empty.")
    positive = (capacity_gcycles_s, active_power_w)
    if not all(isfinite(value) and value > 0 for value in positive):
        raise ValueError("Resource capacity and active power must be positive.")
    if not isfinite(idle_power_w) or idle_power_w < 0:
        raise ValueError("Resource idle power must be finite and non-negative.")
    if active_power_w < idle_power_w:
        raise ValueError("Resource active power cannot be below idle power.")


@dataclass(frozen=True, slots=True)
class MapBounds:

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        values = (self.min_x, self.min_y, self.max_x, self.max_y)
        if not all(isfinite(value) for value in values):
            raise ValueError("Map bounds must be finite.")
        if self.max_x <= self.min_x or self.max_y <= self.min_y:
            raise ValueError("Map bounds must have positive width and height.")

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @classmethod
    def from_trajectories(
        cls,
        frame: pd.DataFrame,
        *,
        padding_fraction: float = 0.02,
    ) -> MapBounds:
        if frame.empty:
            raise ValueError("Cannot infer map bounds from an empty dataset.")
        x_values = pd.to_numeric(frame["x_m"], errors="coerce").dropna()
        y_values = pd.to_numeric(frame["y_m"], errors="coerce").dropna()
        if x_values.empty or y_values.empty:
            raise ValueError("Cannot infer map bounds without Cartesian coordinates.")
        min_x, max_x = float(x_values.min()), float(x_values.max())
        min_y, max_y = float(y_values.min()), float(y_values.max())
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        return cls(
            min_x=min_x - width * padding_fraction,
            min_y=min_y - height * padding_fraction,
            max_x=max_x + width * padding_fraction,
            max_y=max_y + height * padding_fraction,
        )


@dataclass(frozen=True, slots=True)
class ResourceProfile:

    resource_id: str
    resource_type: str
    capacity_gcycles_s: float
    idle_power_w: float
    active_power_w: float

    def __post_init__(self) -> None:
        _validate_resource_values(
            self.resource_id,
            self.resource_type,
            self.capacity_gcycles_s,
            self.idle_power_w,
            self.active_power_w,
        )


RESOURCE_PROFILE_LIBRARY: Mapping[str, ResourceProfile] = {
    "eco": ResourceProfile("cpu-eco", "cpu", 8.0, 12.0, 52.0),
    "balanced": ResourceProfile("cpu-balanced", "cpu", 16.0, 24.0, 112.0),
    "accelerated": ResourceProfile("gpu-accelerated", "gpu", 32.0, 62.0, 288.0),
}


def resolve_resource_profiles(
    names: Sequence[str],
    *,
    capacity_scale: float = 1.0,
) -> tuple[ResourceProfile, ...]:
    if not names:
        raise ValueError("At least one resource profile is required.")
    if not isfinite(capacity_scale) or capacity_scale <= 0:
        raise ValueError("capacity_scale must be finite and positive.")
    profiles: list[ResourceProfile] = []
    for name in names:
        normalized = name.strip().lower().replace("-", "_")
        try:
            profile = RESOURCE_PROFILE_LIBRARY[normalized]
        except KeyError as error:
            choices = ", ".join(sorted(RESOURCE_PROFILE_LIBRARY))
            raise ValueError(
                f"Unknown resource profile {name!r}; choose from {choices}."
            ) from error
        profiles.append(
            ResourceProfile(
                resource_id=profile.resource_id,
                resource_type=profile.resource_type,
                capacity_gcycles_s=profile.capacity_gcycles_s * capacity_scale,
                idle_power_w=profile.idle_power_w,
                active_power_w=profile.active_power_w,
            )
        )
    identifiers = [profile.resource_id for profile in profiles]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Resource profiles must produce unique resource IDs.")
    return tuple(profiles)


@dataclass(slots=True)
class ProcessingResource:

    resource_id: str
    resource_type: str
    capacity_gcycles_s: float
    idle_power_w: float
    active_power_w: float
    available_at_s: float = 0.0
    completed_tasks: int = 0
    cumulative_compute_s: float = 0.0
    cumulative_energy_j: float = 0.0
    scheduled_completion_times_s: list[float] = field(
        default_factory=list,
        repr=False,
    )

    def __post_init__(self) -> None:
        _validate_resource_values(
            self.resource_id,
            self.resource_type,
            self.capacity_gcycles_s,
            self.idle_power_w,
            self.active_power_w,
        )
        runtime_values = (
            self.available_at_s,
            self.cumulative_compute_s,
            self.cumulative_energy_j,
        )
        if not all(isfinite(value) and value >= 0 for value in runtime_values):
            raise ValueError(
                "Resource runtime counters must be finite and non-negative."
            )
        if self.completed_tasks < 0:
            raise ValueError("completed_tasks must be non-negative.")

    @property
    def joules_per_gcycle(self) -> float:
        return self.active_power_w / self.capacity_gcycles_s

    def queue_depth(self, arrival_time_s: float) -> int:
        return sum(
            completion > arrival_time_s
            for completion in self.scheduled_completion_times_s
        )

    def reset_runtime(self) -> None:
        self.available_at_s = 0.0
        self.completed_tasks = 0
        self.cumulative_compute_s = 0.0
        self.cumulative_energy_j = 0.0
        self.scheduled_completion_times_s.clear()

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "capacity_gcycles_s": self.capacity_gcycles_s,
            "idle_power_w": self.idle_power_w,
            "active_power_w": self.active_power_w,
        }
        if include_runtime:
            value.update(
                {
                    "available_at_s": self.available_at_s,
                    "completed_tasks": self.completed_tasks,
                    "cumulative_compute_s": self.cumulative_compute_s,
                    "cumulative_energy_j": self.cumulative_energy_j,
                    "scheduled_completion_times_s": list(
                        self.scheduled_completion_times_s
                    ),
                }
            )
        return value


@dataclass(slots=True)
class EdgeServer:

    server_id: str
    x_m: float
    y_m: float
    coverage_radius_m: float
    resources: list[ProcessingResource]
    uplink_mbps: float = 100.0
    downlink_mbps: float = 200.0

    def __post_init__(self) -> None:
        if not self.server_id or SELECTION_SEPARATOR in self.server_id:
            raise ValueError("server_id must be non-empty and cannot contain ':'.")
        positive = (
            self.coverage_radius_m,
            self.uplink_mbps,
            self.downlink_mbps,
        )
        if not all(isfinite(value) and value > 0 for value in positive):
            raise ValueError("Server radius and bandwidth must be positive.")
        if not all(isfinite(value) for value in (self.x_m, self.y_m)):
            raise ValueError("Server coordinates must be finite.")
        if not self.resources:
            raise ValueError("Every server requires at least one processing resource.")
        identifiers = [resource.resource_id for resource in self.resources]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Resource identifiers must be unique within a server.")

    def distance_to(self, x_m: float, y_m: float) -> float:
        return hypot(x_m - self.x_m, y_m - self.y_m)

    def covers(self, x_m: float, y_m: float) -> bool:
        return self.distance_to(x_m, y_m) <= self.coverage_radius_m

    def get_resource(self, resource_id: str) -> ProcessingResource:
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        raise KeyError(f"Unknown resource {self.server_id}:{resource_id}")

    def selection_for(self, resource: ProcessingResource) -> str:
        return f"{self.server_id}{SELECTION_SEPARATOR}{resource.resource_id}"

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "coverage_radius_m": self.coverage_radius_m,
            "uplink_mbps": self.uplink_mbps,
            "downlink_mbps": self.downlink_mbps,
            "resources": [
                resource.to_dict(include_runtime=include_runtime)
                for resource in self.resources
            ],
        }


@dataclass(frozen=True, slots=True)
class LatencyBreakdown:

    upload_s: float
    queue_s: float
    queue_depth: int
    processing_s: float
    download_s: float
    propagation_s: float
    total_s: float
    completion_time_s: float
    energy_j: float


class EdgeEnvironment:

    def __init__(
        self,
        bounds: MapBounds,
        servers: Sequence[EdgeServer],
        *,
        propagation_speed_mps: float = 200_000_000.0,
        reference_capacity_gcycles_s: float = 10.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not servers:
            raise ValueError("At least one edge server is required.")
        identifiers = [server.server_id for server in servers]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Edge-server identifiers must be unique.")
        positive = (propagation_speed_mps, reference_capacity_gcycles_s)
        if not all(isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "Propagation speed and reference capacity must be positive."
            )
        self.bounds = bounds
        self.servers = list(servers)
        self.propagation_speed_mps = float(propagation_speed_mps)
        self.reference_capacity_gcycles_s = float(reference_capacity_gcycles_s)
        self.metadata = dict(metadata or {})
        self._by_id = {server.server_id: server for server in self.servers}

    @classmethod
    def place_grid(
        cls,
        bounds: MapBounds,
        *,
        server_count: int,
        resource_profiles: Sequence[ResourceProfile] | None = None,
        coverage_overlap: float = 1.10,
        uplink_mbps: float = 100.0,
        downlink_mbps: float = 200.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> EdgeEnvironment:
        if server_count <= 0:
            raise ValueError("server_count must be positive.")
        if coverage_overlap < 1.0:
            raise ValueError("coverage_overlap must be at least one.")
        profiles = tuple(
            resource_profiles
            or resolve_resource_profiles(tuple(RESOURCE_PROFILE_LIBRARY))
        )

        columns = max(1, ceil(sqrt(server_count * bounds.width / bounds.height)))
        rows = ceil(server_count / columns)
        row_populations = [
            server_count // rows + (1 if row < server_count % rows else 0)
            for row in range(rows)
        ]
        cell_height = bounds.height / rows
        widest_cell = max(bounds.width / population for population in row_populations)
        radius = 0.5 * hypot(widest_cell, cell_height) * coverage_overlap

        servers: list[EdgeServer] = []
        index = 0
        for row, population in enumerate(row_populations):
            cell_width = bounds.width / population
            for column in range(population):
                servers.append(
                    EdgeServer(
                        server_id=f"edge-{index:03d}",
                        x_m=bounds.min_x + (column + 0.5) * cell_width,
                        y_m=bounds.min_y + (row + 0.5) * cell_height,
                        coverage_radius_m=radius,
                        resources=_resources_from_profiles(profiles),
                        uplink_mbps=uplink_mbps,
                        downlink_mbps=downlink_mbps,
                    )
                )
                index += 1
        return cls(
            bounds,
            servers,
            metadata={"placement": "grid", **dict(metadata or {})},
        )

    @classmethod
    def place_density_aware(
        cls,
        frame: pd.DataFrame,
        *,
        server_count: int,
        resource_profiles: Sequence[ResourceProfile] | None = None,
        coverage_percentile: float = 95.0,
        coverage_overlap: float = 1.10,
        uplink_mbps: float = 100.0,
        downlink_mbps: float = 200.0,
        seed: int = 42,
        metadata: Mapping[str, Any] | None = None,
    ) -> EdgeEnvironment:
        if server_count <= 0:
            raise ValueError("server_count must be positive.")
        if not 50.0 <= coverage_percentile <= 100.0:
            raise ValueError("coverage_percentile must be between 50 and 100.")
        if coverage_overlap <= 0:
            raise ValueError("coverage_overlap must be positive.")
        profiles = tuple(
            resource_profiles
            or resolve_resource_profiles(tuple(RESOURCE_PROFILE_LIBRARY))
        )
        points = frame.loc[:, ["x_m", "y_m"]].dropna().to_numpy(dtype=float)
        if len(points) < server_count:
            raise ValueError("Fewer observations than requested edge servers.")
        from sklearn.cluster import MiniBatchKMeans

        estimator = MiniBatchKMeans(
            n_clusters=server_count,
            batch_size=min(4096, max(256, len(points))),
            n_init="auto",
            random_state=seed,
        )
        labels = estimator.fit_predict(points)
        centers = estimator.cluster_centers_
        servers: list[EdgeServer] = []
        for index, center in enumerate(centers):
            cluster = points[labels == index]
            distances = np.hypot(
                cluster[:, 0] - center[0],
                cluster[:, 1] - center[1],
            )
            radius = max(
                1.0,
                float(np.percentile(distances, coverage_percentile)) * coverage_overlap,
            )
            servers.append(
                EdgeServer(
                    server_id=f"edge-{index:03d}",
                    x_m=float(center[0]),
                    y_m=float(center[1]),
                    coverage_radius_m=radius,
                    resources=_resources_from_profiles(profiles),
                    uplink_mbps=uplink_mbps,
                    downlink_mbps=downlink_mbps,
                )
            )
        return cls(
            MapBounds.from_trajectories(frame),
            servers,
            metadata={"placement": "density", **dict(metadata or {})},
        )

    def clone(self, *, reset_runtime: bool = True) -> EdgeEnvironment:
        cloned = EdgeEnvironment.from_dict(self.to_dict(include_runtime=True))
        if reset_runtime:
            for _, resource in cloned.iter_resources():
                resource.reset_runtime()
        return cloned

    def get_server(self, server_id: str) -> EdgeServer:
        try:
            return self._by_id[server_id]
        except KeyError as error:
            raise KeyError(f"Unknown edge server: {server_id}") from error

    def get_selection(
        self,
        selection_id: str,
    ) -> tuple[EdgeServer, ProcessingResource]:
        if SELECTION_SEPARATOR not in selection_id:
            raise ValueError("Offloading decisions must use 'server_id:resource_id'.")
        server_id, resource_id = selection_id.rsplit(SELECTION_SEPARATOR, 1)
        server = self.get_server(server_id)
        return server, server.get_resource(resource_id)

    def iter_resources(self) -> Iterable[tuple[EdgeServer, ProcessingResource]]:
        for server in self.servers:
            for resource in server.resources:
                yield server, resource

    def selections(
        self,
        *,
        servers: Iterable[EdgeServer] | None = None,
    ) -> list[str]:
        pool = list(servers) if servers is not None else self.servers
        return [
            server.selection_for(resource)
            for server in pool
            for resource in server.resources
        ]

    def servers_covering(self, x_m: float, y_m: float) -> list[EdgeServer]:
        return [server for server in self.servers if server.covers(x_m, y_m)]

    def nearest_server(
        self,
        x_m: float,
        y_m: float,
        *,
        candidates: Iterable[EdgeServer] | None = None,
    ) -> EdgeServer:
        pool = list(candidates) if candidates is not None else self.servers
        if not pool:
            raise ValueError("No candidate edge servers were supplied.")
        return min(
            pool,
            key=lambda server: (server.distance_to(x_m, y_m), server.server_id),
        )

    def estimate_proxy_cost(
        self,
        server: EdgeServer,
        resource: ProcessingResource,
        state: VehicleState,
        task_duration_s: float,
        *,
        energy_weight: float = 2.50,
    ) -> float:
        queue_s = max(0.0, resource.available_at_s - state.time_s)
        processing_s = (
            task_duration_s
            * self.reference_capacity_gcycles_s
            / resource.capacity_gcycles_s
        )
        energy_j = processing_s * resource.active_power_w
        reference_energy = (
            task_duration_s
            * self.reference_capacity_gcycles_s
            * min(item.joules_per_gcycle for _, item in self.iter_resources())
        )
        return (
            (queue_s + processing_s) / task_duration_s
            + energy_weight * energy_j / max(reference_energy, 1e-9)
            + 0.02
            * server.distance_to(state.x_m, state.y_m)
            / max(server.coverage_radius_m, 1.0)
        )

    def best_resource_selection(
        self,
        server: EdgeServer,
        state: VehicleState,
        task_duration_s: float,
        *,
        energy_weight: float = 2.50,
    ) -> str:
        resource = min(
            server.resources,
            key=lambda item: (
                self.estimate_proxy_cost(
                    server,
                    item,
                    state,
                    task_duration_s,
                    energy_weight=energy_weight,
                ),
                item.resource_id,
            ),
        )
        return server.selection_for(resource)

    def estimate_latency(
        self,
        task: ComputeTask,
        selection_id: str,
        state: VehicleState,
    ) -> LatencyBreakdown:
        server, resource = self.get_selection(selection_id)
        distance = server.distance_to(state.x_m, state.y_m)
        propagation = 2.0 * distance / self.propagation_speed_mps
        upload = task.input_mb * 8.0 / server.uplink_mbps
        download = task.output_mb * 8.0 / server.downlink_mbps
        upload_finished = task.created_at_s + upload + propagation / 2.0
        processing_start = max(upload_finished, resource.available_at_s)
        queue = processing_start - upload_finished
        processing = task.compute_gcycles / resource.capacity_gcycles_s
        processing_finished = processing_start + processing
        completion = processing_finished + download + propagation / 2.0
        return LatencyBreakdown(
            upload_s=upload,
            queue_s=queue,
            queue_depth=resource.queue_depth(upload_finished),
            processing_s=processing,
            download_s=download,
            propagation_s=propagation,
            total_s=completion - task.created_at_s,
            completion_time_s=completion,
            energy_j=processing * resource.active_power_w,
        )

    def assign_task(
        self,
        task: ComputeTask,
        selection_id: str,
        state: VehicleState,
    ) -> LatencyBreakdown:
        latency = self.estimate_latency(task, selection_id, state)
        _, resource = self.get_selection(selection_id)
        processing_finished = (
            latency.completion_time_s - latency.download_s - latency.propagation_s / 2.0
        )
        resource.available_at_s = processing_finished
        resource.completed_tasks += 1
        resource.cumulative_compute_s += latency.processing_s
        resource.cumulative_energy_j += latency.energy_j
        resource.scheduled_completion_times_s = [
            completion
            for completion in resource.scheduled_completion_times_s
            if completion > task.created_at_s
        ]
        resource.scheduled_completion_times_s.append(processing_finished)
        return latency

    def load_distribution(self) -> dict[str, int]:
        return {
            server.selection_for(resource): resource.completed_tasks
            for server, resource in self.iter_resources()
        }

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        return {
            "schema_version": ENVIRONMENT_SCHEMA_VERSION,
            "bounds": asdict(self.bounds),
            "servers": [
                server.to_dict(include_runtime=include_runtime)
                for server in self.servers
            ],
            "propagation_speed_mps": self.propagation_speed_mps,
            "reference_capacity_gcycles_s": self.reference_capacity_gcycles_s,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EdgeEnvironment:
        bounds = MapBounds(**dict(value["bounds"]))
        servers: list[EdgeServer] = []
        for raw_server in value["servers"]:
            server_value = dict(raw_server)
            raw_resources = server_value.pop("resources", None)
            if raw_resources is None:
                legacy_capacity = float(
                    server_value.pop("processing_capacity_gcycles_s")
                )
                for key in (
                    "available_at_s",
                    "completed_tasks",
                    "cumulative_compute_s",
                ):
                    server_value.pop(key, None)
                raw_resources = [
                    {
                        "resource_id": "cpu-legacy",
                        "resource_type": "cpu",
                        "capacity_gcycles_s": legacy_capacity,
                        "idle_power_w": 20.0,
                        "active_power_w": 120.0,
                    }
                ]
            resources = [ProcessingResource(**dict(item)) for item in raw_resources]
            servers.append(EdgeServer(resources=resources, **server_value))
        return cls(
            bounds,
            servers,
            propagation_speed_mps=float(
                value.get("propagation_speed_mps", 200_000_000.0)
            ),
            reference_capacity_gcycles_s=float(
                value.get("reference_capacity_gcycles_s", 10.0)
            ),
            metadata=dict(value.get("metadata", {})),
        )

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(include_runtime=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> EdgeEnvironment:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Environment file not found: {source}")
        try:
            with source.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid environment JSON: {source}") from error
        if not isinstance(value, Mapping):
            raise ValueError("Environment JSON root must be an object.")
        return cls.from_dict(value)


def _resources_from_profiles(
    profiles: Sequence[ResourceProfile],
) -> list[ProcessingResource]:
    return [
        ProcessingResource(
            resource_id=profile.resource_id,
            resource_type=profile.resource_type,
            capacity_gcycles_s=profile.capacity_gcycles_s,
            idle_power_w=profile.idle_power_w,
            active_power_w=profile.active_power_w,
        )
        for profile in profiles
    ]
