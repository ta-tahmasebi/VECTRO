from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..domain import VehicleState
from ..edge_env import EdgeEnvironment

TrainingProgress = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class TrainingSummary:

    algorithm: str
    observations: int
    vehicles: int
    artifact_directory: Path
    details: Mapping[str, Any]


class OffloadingAgent(ABC):
    algorithm_name: str
    trainable: bool = False

    def __init__(
        self,
        environment: EdgeEnvironment,
        artifact_directory: str | Path,
        *,
        seed: int = 42,
    ) -> None:
        self.environment = environment
        self.artifact_directory = Path(artifact_directory)
        self.seed = int(seed)
        self._is_ready = False

    @abstractmethod
    def train(
        self,
        trajectories: pd.DataFrame,
        *,
        progress: TrainingProgress | None = None,
    ) -> TrainingSummary:

    @abstractmethod
    def predict(
        self,
        state: VehicleState,
        task_duration_s: float,
    ) -> str:

    @abstractmethod
    def load(self) -> None:

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _validate_training_frame(self, trajectories: pd.DataFrame) -> None:
        required = {
            "vehicle_id",
            "time_s",
            "x_m",
            "y_m",
            "speed_mps",
            "angle_deg",
        }
        missing = sorted(required.difference(trajectories.columns))
        if missing:
            raise ValueError(f"Training data is missing columns: {missing}")
        if trajectories.empty:
            raise ValueError("Training data must not be empty.")

    def _validate_inference(
        self,
        state: VehicleState,
        task_duration_s: float,
    ) -> None:
        if not self._is_ready:
            raise RuntimeError(
                f"{self.algorithm_name} must be trained or loaded before inference."
            )
        if not isfinite(task_duration_s) or task_duration_s <= 0:
            raise ValueError("task_duration_s must be finite and positive.")
        if not isinstance(state, VehicleState):
            raise TypeError("state must be a VehicleState instance.")

    def _select_for_future_position(
        self,
        x_m: float,
        y_m: float,
        state: VehicleState,
        task_duration_s: float,
    ) -> str:
        future_covered = self.environment.servers_covering(x_m, y_m)
        current_covered = self.environment.servers_covering(state.x_m, state.y_m)
        current_ids = {server.server_id for server in current_covered}
        continuous_candidates = [
            server for server in future_covered if server.server_id in current_ids
        ]
        candidates = (
            continuous_candidates
            or future_covered
            or current_covered
            or self.environment.servers
        )
        server, resource = min(
            (
                (server, resource)
                for server in candidates
                for resource in server.resources
            ),
            key=lambda choice: (
                self.environment.estimate_proxy_cost(
                    choice[0],
                    choice[1],
                    state,
                    task_duration_s,
                )
                + 0.20 * choice[0].distance_to(x_m, y_m) / choice[0].coverage_radius_m,
                choice[0].selection_for(choice[1]),
            ),
        )
        return server.selection_for(resource)

    def _save_metadata(
        self,
        trajectories: pd.DataFrame,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> TrainingSummary:
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": 2,
            "algorithm": self.algorithm_name,
            "trainable": self.trainable,
            "seed": self.seed,
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "training_observations": int(len(trajectories)),
            "training_vehicles": int(trajectories["vehicle_id"].nunique()),
            "environment": self.environment.to_dict(),
            "environment_fingerprint": self.environment.fingerprint(),
            "details": dict(details or {}),
        }
        destination = self.artifact_directory / "metadata.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(destination)
        self._is_ready = True
        return TrainingSummary(
            algorithm=self.algorithm_name,
            observations=int(len(trajectories)),
            vehicles=int(trajectories["vehicle_id"].nunique()),
            artifact_directory=self.artifact_directory,
            details=dict(details or {}),
        )

    def _read_metadata(self) -> dict[str, Any]:
        source = self.artifact_directory / "metadata.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"Saved model metadata not found: {source}. Run training first."
            )
        try:
            metadata = cast(
                dict[str, Any],
                json.loads(source.read_text(encoding="utf-8")),
            )
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid model metadata: {source}") from error
        if metadata.get("algorithm") != self.algorithm_name:
            raise ValueError(
                f"Artifact contains {metadata.get('algorithm')!r}, "
                f"not {self.algorithm_name!r}."
            )
        return metadata
