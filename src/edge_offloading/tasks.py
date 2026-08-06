
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite

import numpy as np
from numpy.typing import NDArray

from .domain import VehicleState


@dataclass(frozen=True, slots=True)
class ComputeTask:

    task_id: str
    vehicle_id: str
    created_at_s: float
    nominal_execution_s: float
    compute_gcycles: float
    input_mb: float
    output_mb: float
    deadline_s: float | None = None

    def __post_init__(self) -> None:
        positive = (
            self.nominal_execution_s,
            self.compute_gcycles,
            self.input_mb,
        )
        if not self.task_id or not self.vehicle_id:
            raise ValueError("Task and vehicle identifiers must not be empty.")
        if not all(isfinite(value) and value > 0 for value in positive):
            raise ValueError(
                "Execution time, compute demand, and input must be positive."
            )
        if not isfinite(self.output_mb) or self.output_mb < 0:
            raise ValueError("output_mb must be finite and non-negative.")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("deadline_s must be positive when supplied.")


class TaskGenerator:

    def __init__(
        self,
        *,
        arrival_rate_per_vehicle_minute: float = 0.5,
        execution_durations_s: Sequence[float] = (10.0, 30.0),
        duration_probabilities: Sequence[float] | None = None,
        reference_capacity_gcycles_s: float = 10.0,
        input_mb: float = 2.0,
        output_mb: float = 0.2,
        deadline_factor: float | None = 2.0,
        seed: int = 42,
    ) -> None:
        if arrival_rate_per_vehicle_minute <= 0:
            raise ValueError("Task arrival rate must be positive.")
        durations = np.asarray(execution_durations_s, dtype=float)
        if (
            durations.size == 0
            or np.any(~np.isfinite(durations))
            or np.any(durations <= 0)
        ):
            raise ValueError("Execution durations must be finite and positive.")
        if duration_probabilities is None:
            probabilities = np.full(durations.size, 1.0 / durations.size)
        else:
            probabilities = np.asarray(duration_probabilities, dtype=float)
            if probabilities.shape != durations.shape:
                raise ValueError("Duration probabilities must match durations.")
            if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
                raise ValueError("Duration probabilities must sum to one.")

        self.arrival_rate_per_second = arrival_rate_per_vehicle_minute / 60.0
        self.durations = durations
        self.probabilities = probabilities
        self.reference_capacity = float(reference_capacity_gcycles_s)
        self.input_mb = float(input_mb)
        self.output_mb = float(output_mb)
        self.deadline_factor = deadline_factor
        self._rng = np.random.default_rng(seed)
        self._counter = 0

    def should_generate(self, interval_s: float) -> bool:
        if interval_s <= 0:
            return False
        probability = 1.0 - exp(-self.arrival_rate_per_second * interval_s)
        return bool(self._rng.random() < probability)

    def sample_arrival_offsets(self, interval_s: float) -> NDArray[np.float64]:
        
        if not isfinite(interval_s) or interval_s <= 0:
            return np.empty(0, dtype=float)
        count = int(self._rng.poisson(self.arrival_rate_per_second * interval_s))
        if count == 0:
            return np.empty(0, dtype=float)
        return np.sort(self._rng.uniform(0.0, interval_s, size=count))

    def create(self, state: VehicleState) -> ComputeTask:
        duration = float(self._rng.choice(self.durations, p=self.probabilities))
        self._counter += 1
        deadline = (
            duration * float(self.deadline_factor)
            if self.deadline_factor is not None
            else None
        )
        return ComputeTask(
            task_id=f"task-{self._counter:09d}",
            vehicle_id=state.vehicle_id,
            created_at_s=state.time_s,
            nominal_execution_s=duration,
            compute_gcycles=duration * self.reference_capacity,
            input_mb=self.input_mb,
            output_mb=self.output_mb,
            deadline_s=deadline,
        )
