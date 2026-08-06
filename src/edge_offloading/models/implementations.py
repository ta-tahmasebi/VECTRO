from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from math import ceil, hypot
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..domain import VehicleState
from ..edge_env import EdgeEnvironment, EdgeServer
from .base_model import OffloadingAgent, TrainingProgress, TrainingSummary


class _ArtifactOnlyAgent(OffloadingAgent):

    def train(
        self,
        trajectories: pd.DataFrame,
        *,
        progress: TrainingProgress | None = None,
    ) -> TrainingSummary:
        self._validate_training_frame(trajectories)
        if progress is not None:
            progress(len(trajectories), len(trajectories), "Validating full dataset")
        return self._save_metadata(
            trajectories,
            details={"fitted_parameters": 0},
        )

    def load(self) -> None:
        self._read_metadata()
        self._is_ready = True


class GreedyNearestAgent(_ArtifactOnlyAgent):

    algorithm_name = "greedy"

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        covered = self.environment.servers_covering(state.x_m, state.y_m)
        server = self.environment.nearest_server(
            state.x_m,
            state.y_m,
            candidates=covered or self.environment.servers,
        )
        return self.environment.best_resource_selection(
            server,
            state,
            task_duration_s,
        )


class RandomServerAgent(_ArtifactOnlyAgent):

    algorithm_name = "random"

    def __init__(
        self,
        environment: EdgeEnvironment,
        artifact_directory: str | Path,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__(environment, artifact_directory, seed=seed)
        self._rng = np.random.default_rng(seed)

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        candidates = self.environment.servers_covering(state.x_m, state.y_m)
        if not candidates:
            candidates = self.environment.servers
        server = candidates[int(self._rng.integers(0, len(candidates)))]
        resource = server.resources[int(self._rng.integers(0, len(server.resources)))]
        return server.selection_for(resource)


class UniformRandomAgent(RandomServerAgent):

    algorithm_name = "uniform_random"

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        selections = self.environment.selections()
        return selections[int(self._rng.integers(0, len(selections)))]


class FurthestServerAgent(_ArtifactOnlyAgent):
    algorithm_name = "furthest"

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        candidates = self.environment.servers_covering(state.x_m, state.y_m)
        candidates = candidates or self.environment.servers
        server = max(
            candidates,
            key=lambda item: (
                item.distance_to(state.x_m, state.y_m),
                item.server_id,
            ),
        )
        resource = min(
            server.resources,
            key=lambda item: (
                abs(
                    item.capacity_gcycles_s
                    - self.environment.reference_capacity_gcycles_s
                ),
                item.resource_id,
            ),
        )
        return server.selection_for(resource)


class KalmanMobilityAgent(OffloadingAgent):
    algorithm_name = "kalman"
    trainable = True

    def __init__(
        self,
        environment: EdgeEnvironment,
        artifact_directory: str | Path,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__(environment, artifact_directory, seed=seed)
        self.transition_dt_s = 1.0
        self.velocity_retention = np.ones(2, dtype=float)
        self.process_variance = np.ones(2, dtype=float)
        self.measurement_variance = np.ones(2, dtype=float)

    def train(
        self,
        trajectories: pd.DataFrame,
        *,
        progress: TrainingProgress | None = None,
    ) -> TrainingSummary:
        self._validate_training_frame(trajectories)
        frame = trajectories.sort_values(["vehicle_id", "time_s"], kind="stable")
        group = frame.groupby("vehicle_id", sort=False, observed=True)
        delta_t = group["time_s"].diff()
        velocity_x = group["x_m"].diff().div(delta_t)
        velocity_y = group["y_m"].diff().div(delta_t)
        velocity = pd.DataFrame({"x": velocity_x, "y": velocity_y})
        next_velocity = velocity.groupby(frame["vehicle_id"], observed=True).shift(-1)
        valid_dt = delta_t[np.isfinite(delta_t) & delta_t.gt(0)]
        if valid_dt.empty:
            raise ValueError("Kalman training requires increasing timestamps.")
        self.transition_dt_s = float(valid_dt.median())

        retention: list[float] = []
        process_variance: list[float] = []
        measurement_variance: list[float] = []
        for axis, position_column in (("x", "x_m"), ("y", "y_m")):
            current = velocity[axis].to_numpy(float)
            following = next_velocity[axis].to_numpy(float)
            valid = np.isfinite(current) & np.isfinite(following)
            denominator = float(np.dot(current[valid], current[valid]))
            coefficient = (
                float(np.dot(current[valid], following[valid])) / denominator
                if denominator > 1e-12
                else 1.0
            )
            coefficient = float(np.clip(coefficient, 0.0, 1.2))
            residual = following[valid] - coefficient * current[valid]
            retention.append(coefficient)
            process_variance.append(_safe_variance(residual))

            previous_position = group[position_column].shift()
            predicted = previous_position + velocity[axis] * delta_t
            measurement_residual = (frame[position_column] - predicted).to_numpy(float)
            measurement_variance.append(_safe_variance(measurement_residual))

        self.velocity_retention = np.asarray(retention, dtype=float)
        self.process_variance = np.asarray(process_variance, dtype=float)
        self.measurement_variance = np.asarray(measurement_variance, dtype=float)
        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.artifact_directory / "kalman_parameters.npz",
            transition_dt_s=self.transition_dt_s,
            velocity_retention=self.velocity_retention,
            process_variance=self.process_variance,
            measurement_variance=self.measurement_variance,
        )
        if progress is not None:
            progress(len(frame), len(frame), "Estimating Kalman dynamics")
        return self._save_metadata(
            trajectories,
            details={
                "transition_dt_s": self.transition_dt_s,
                "velocity_retention": self.velocity_retention.tolist(),
                "process_variance": self.process_variance.tolist(),
                "measurement_variance": self.measurement_variance.tolist(),
            },
        )

    def load(self) -> None:
        self._read_metadata()
        source = self.artifact_directory / "kalman_parameters.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Kalman weights not found: {source}")
        with np.load(source) as values:
            self.transition_dt_s = float(values["transition_dt_s"])
            self.velocity_retention = values["velocity_retention"].astype(float)
            self.process_variance = values["process_variance"].astype(float)
            self.measurement_variance = values["measurement_variance"].astype(float)
        self._is_ready = True

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        position = np.asarray([state.x_m, state.y_m], dtype=float)
        velocity = np.asarray(state.velocity_xy, dtype=float)
        remaining = float(task_duration_s)
        while remaining > 1e-12:
            step = min(self.transition_dt_s, remaining)
            fraction = step / self.transition_dt_s
            retention = np.power(self.velocity_retention, fraction)
            next_velocity = velocity * retention
            position += 0.5 * (velocity + next_velocity) * step
            velocity = next_velocity
            remaining -= step
        return self._select_for_future_position(
            float(position[0]),
            float(position[1]),
            state,
            task_duration_s,
        )


class MarkovTrajectoryAgent(OffloadingAgent):

    algorithm_name = "markov"
    trainable = True

    def __init__(
        self,
        environment: EdgeEnvironment,
        artifact_directory: str | Path,
        *,
        grid_size: int = 24,
        seed: int = 42,
    ) -> None:
        super().__init__(environment, artifact_directory, seed=seed)
        if grid_size < 2:
            raise ValueError("grid_size must be at least two.")
        self.grid_size = int(grid_size)
        self.transition_dt_s = 1.0
        self._transitions: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def train(
        self,
        trajectories: pd.DataFrame,
        *,
        progress: TrainingProgress | None = None,
    ) -> TrainingSummary:
        self._validate_training_frame(trajectories)
        counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        time_deltas: list[np.ndarray] = []
        groups = trajectories.sort_values(
            ["vehicle_id", "time_s"], kind="stable"
        ).groupby("vehicle_id", sort=False, observed=True)
        total_groups = int(trajectories["vehicle_id"].nunique())
        for group_index, (_, group) in enumerate(groups, start=1):
            x = group["x_m"].to_numpy(float)
            y = group["y_m"].to_numpy(float)
            times = group["time_s"].to_numpy(float)
            if len(group) < 2:
                if progress is not None:
                    progress(group_index, total_groups, "Learning Markov transitions")
                continue
            states = self._encode_many(x, y)
            deltas = np.diff(times)
            valid = np.isfinite(deltas) & (deltas > 0)
            time_deltas.append(deltas[valid])
            encoded_pairs = states[:-1][valid] * (self.grid_size**2) + states[1:][valid]
            pair_keys, pair_counts = np.unique(encoded_pairs, return_counts=True)
            for key, count in zip(pair_keys, pair_counts, strict=True):
                source, target = divmod(int(key), self.grid_size**2)
                counts[source][target] += int(count)
            if progress is not None:
                progress(group_index, total_groups, "Learning Markov transitions")

        nonempty_deltas = [values for values in time_deltas if values.size]
        if not nonempty_deltas or not counts:
            raise ValueError("Markov training found no valid state transitions.")
        self.transition_dt_s = float(np.median(np.concatenate(nonempty_deltas)))
        rows: list[int] = []
        columns: list[int] = []
        probabilities: list[float] = []
        self._transitions.clear()
        for source, targets in counts.items():
            target_ids = np.fromiter(targets.keys(), dtype=np.int64)
            target_counts = np.fromiter(targets.values(), dtype=float)
            target_probabilities = target_counts / target_counts.sum()
            self._transitions[source] = (target_ids, target_probabilities)
            rows.extend([source] * len(target_ids))
            columns.extend(target_ids.tolist())
            probabilities.extend(target_probabilities.tolist())

        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.artifact_directory / "markov_transitions.npz",
            rows=np.asarray(rows, dtype=np.int64),
            columns=np.asarray(columns, dtype=np.int64),
            probabilities=np.asarray(probabilities, dtype=float),
            grid_size=self.grid_size,
            transition_dt_s=self.transition_dt_s,
        )
        return self._save_metadata(
            trajectories,
            details={
                "grid_size": self.grid_size,
                "transition_dt_s": self.transition_dt_s,
                "nonzero_transitions": len(probabilities),
            },
        )

    def load(self) -> None:
        self._read_metadata()
        source = self.artifact_directory / "markov_transitions.npz"
        if not source.is_file():
            raise FileNotFoundError(f"Markov model not found: {source}")
        with np.load(source) as values:
            self.grid_size = int(values["grid_size"])
            self.transition_dt_s = float(values["transition_dt_s"])
            rows = values["rows"].astype(np.int64)
            columns = values["columns"].astype(np.int64)
            probabilities = values["probabilities"].astype(float)
        self._transitions.clear()
        for row in np.unique(rows):
            mask = rows == row
            self._transitions[int(row)] = (columns[mask], probabilities[mask])
        self._is_ready = True

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        initial = self._encode(state.x_m, state.y_m)
        if initial not in self._transitions:
            velocity_x, velocity_y = state.velocity_xy
            return self._select_for_future_position(
                state.x_m + velocity_x * task_duration_s,
                state.y_m + velocity_y * task_duration_s,
                state,
                task_duration_s,
            )

        distribution: dict[int, float] = {initial: 1.0}
        steps = max(1, ceil(task_duration_s / self.transition_dt_s))
        for _ in range(steps):
            updated: dict[int, float] = defaultdict(float)
            for source, source_probability in distribution.items():
                transition = self._transitions.get(source)
                if transition is None:
                    updated[source] += source_probability
                    continue
                targets, probabilities = transition
                for target, probability in zip(targets, probabilities, strict=True):
                    updated[int(target)] += source_probability * float(probability)
            distribution = dict(updated)
        x_m = 0.0
        y_m = 0.0
        total = sum(distribution.values())
        for state_id, probability in distribution.items():
            center_x, center_y = self._decode_center(state_id)
            x_m += center_x * probability
            y_m += center_y * probability
        return self._select_for_future_position(
            x_m / total,
            y_m / total,
            state,
            task_duration_s,
        )

    def _encode_many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        bounds = self.environment.bounds
        x_index = np.floor((x - bounds.min_x) / bounds.width * self.grid_size).astype(
            int
        )
        y_index = np.floor((y - bounds.min_y) / bounds.height * self.grid_size).astype(
            int
        )
        x_index = np.clip(x_index, 0, self.grid_size - 1)
        y_index = np.clip(y_index, 0, self.grid_size - 1)
        return y_index * self.grid_size + x_index

    def _encode(self, x_m: float, y_m: float) -> int:
        return int(
            self._encode_many(
                np.asarray([x_m], dtype=float),
                np.asarray([y_m], dtype=float),
            )[0]
        )

    def _decode_center(self, state_id: int) -> tuple[float, float]:
        row, column = divmod(state_id, self.grid_size)
        bounds = self.environment.bounds
        return (
            bounds.min_x + (column + 0.5) * bounds.width / self.grid_size,
            bounds.min_y + (row + 0.5) * bounds.height / self.grid_size,
        )


class GRUMobilityAgent(OffloadingAgent):

    algorithm_name = "gru"
    trainable = True

    def __init__(
        self,
        environment: EdgeEnvironment,
        artifact_directory: str | Path,
        *,
        hidden_size: int = 64,
        sequence_length: int = 16,
        stride: int | None = None,
        epochs: int = 12,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(environment, artifact_directory, seed=seed)
        if hidden_size <= 0 or sequence_length < 2:
            raise ValueError("hidden_size must be positive and sequence_length >= 2.")
        self.hidden_size = int(hidden_size)
        self.sequence_length = int(sequence_length)
        self.stride = int(stride or sequence_length)
        if self.stride <= 0 or self.stride > self.sequence_length:
            raise ValueError(
                "stride must be between 1 and sequence_length so no time section "
                "is skipped."
            )
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError("GRU training hyperparameters must be positive.")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.normalization: dict[str, float] = {}
        self._model: Any | None = None

    def train(
        self,
        trajectories: pd.DataFrame,
        *,
        progress: TrainingProgress | None = None,
    ) -> TrainingSummary:
        self._validate_training_frame(trajectories)
        torch, nn = _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        frame = trajectories.sort_values(["vehicle_id", "time_s"], kind="stable")
        velocity_x = frame["speed_mps"].to_numpy(float) * np.sin(
            np.deg2rad(frame["angle_deg"].to_numpy(float))
        )
        velocity_y = frame["speed_mps"].to_numpy(float) * np.cos(
            np.deg2rad(frame["angle_deg"].to_numpy(float))
        )
        x_values = frame["x_m"].to_numpy(float)
        y_values = frame["y_m"].to_numpy(float)
        positive_dt = (
            frame.groupby("vehicle_id", observed=True)["time_s"].diff().dropna()
        )
        positive_dt = positive_dt[positive_dt > 0]
        if positive_dt.empty:
            raise ValueError("GRU training requires increasing timestamps.")

        self.normalization = {
            "x_mean": float(np.mean(x_values)),
            "x_scale": _safe_scale(x_values),
            "y_mean": float(np.mean(y_values)),
            "y_scale": _safe_scale(y_values),
            "vx_mean": float(np.mean(velocity_x)),
            "vx_scale": _safe_scale(velocity_x),
            "vy_mean": float(np.mean(velocity_y)),
            "vy_scale": _safe_scale(velocity_y),
            "dt_s": float(np.median(positive_dt.to_numpy(float))),
            "max_speed_mps": float(max(1.0, np.percentile(frame["speed_mps"], 99.5))),
        }

        sequences = self._build_sequences(frame)
        if not sequences:
            raise ValueError("GRU training found no trajectories with two samples.")
        dataset = _SequenceDataset(sequences, torch)
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: _collate_sequences(batch, torch),
        )
        model = _build_gru(nn, self.hidden_size)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        total_batches = self.epochs * len(loader)
        completed = 0
        final_loss = float("nan")
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            observations = 0.0
            for features, targets, mask in loader:
                optimizer.zero_grad(set_to_none=True)
                predictions, _ = model(features)
                squared_error = (predictions - targets).pow(2).sum(dim=-1)
                loss = (squared_error * mask).sum() / mask.sum().clamp_min(1.0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("GRU training produced a non-finite loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += float(loss.detach()) * float(mask.sum())
                observations += float(mask.sum())
                completed += 1
                if progress is not None:
                    progress(
                        completed,
                        total_batches,
                        f"Training GRU epoch {epoch + 1}/{self.epochs}",
                    )
            final_loss = epoch_loss / max(observations, 1.0)

        self.artifact_directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.artifact_directory / "gru_weights.pt")
        normalization_path = self.artifact_directory / "gru_normalization.json"
        normalization_path.write_text(
            json.dumps(
                {
                    **self.normalization,
                    "hidden_size": self.hidden_size,
                    "sequence_length": self.sequence_length,
                    "stride": self.stride,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._model = model.eval()
        return self._save_metadata(
            trajectories,
            details={
                "sequences": len(sequences),
                "epochs": self.epochs,
                "final_loss": final_loss,
                "sequence_length": self.sequence_length,
                "stride": self.stride,
                "temporal_coverage": "complete",
            },
        )

    def load(self) -> None:
        self._read_metadata()
        torch, nn = _require_torch()
        normalization_path = self.artifact_directory / "gru_normalization.json"
        weights_path = self.artifact_directory / "gru_weights.pt"
        if not normalization_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                f"Incomplete GRU artifact in {self.artifact_directory}"
            )
        values = json.loads(normalization_path.read_text(encoding="utf-8"))
        self.hidden_size = int(values.pop("hidden_size"))
        self.sequence_length = int(values.pop("sequence_length"))
        self.stride = int(values.pop("stride"))
        self.normalization = {key: float(value) for key, value in values.items()}
        model = _build_gru(nn, self.hidden_size)
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        self._model = model.eval()
        self._is_ready = True

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        if self._model is None:
            raise RuntimeError("GRU weights are not loaded.")
        torch, _ = _require_torch()
        x_m, y_m = state.x_m, state.y_m
        velocity_x, velocity_y = state.velocity_xy
        remaining = float(task_duration_s)
        hidden = None
        self._model.eval()
        with torch.no_grad():
            while remaining > 1e-9:
                step = min(self.normalization["dt_s"], remaining)
                features = np.asarray(
                    [
                        (x_m - self.normalization["x_mean"])
                        / self.normalization["x_scale"],
                        (y_m - self.normalization["y_mean"])
                        / self.normalization["y_scale"],
                        (velocity_x - self.normalization["vx_mean"])
                        / self.normalization["vx_scale"],
                        (velocity_y - self.normalization["vy_mean"])
                        / self.normalization["vy_scale"],
                        step / self.normalization["dt_s"],
                    ],
                    dtype=np.float32,
                )
                tensor = torch.from_numpy(features).reshape(1, 1, -1)
                output, hidden = self._model(tensor, hidden)
                delta_x = float(output[0, 0, 0]) * self.normalization["x_scale"]
                delta_y = float(output[0, 0, 1]) * self.normalization["y_scale"]
                max_displacement = self.normalization["max_speed_mps"] * step * 1.5
                magnitude = hypot(delta_x, delta_y)
                if magnitude > max_displacement > 0:
                    scale = max_displacement / magnitude
                    delta_x *= scale
                    delta_y *= scale
                x_m += delta_x
                y_m += delta_y
                velocity_x = delta_x / step
                velocity_y = delta_y / step
                remaining -= step
        return self._select_for_future_position(
            x_m,
            y_m,
            state,
            task_duration_s,
        )

    def _build_sequences(
        self,
        frame: pd.DataFrame,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        norm = self.normalization
        sequences: list[tuple[np.ndarray, np.ndarray]] = []
        for _, group in frame.groupby("vehicle_id", sort=False, observed=True):
            if len(group) < 2:
                continue
            x = group["x_m"].to_numpy(float)
            y = group["y_m"].to_numpy(float)
            times = group["time_s"].to_numpy(float)
            speed = group["speed_mps"].to_numpy(float)
            angle = np.deg2rad(group["angle_deg"].to_numpy(float))
            velocity_x = speed * np.sin(angle)
            velocity_y = speed * np.cos(angle)
            delta_t = np.diff(times)
            valid = np.isfinite(delta_t) & (delta_t > 0)
            if not valid.any():
                continue
            features = np.column_stack(
                [
                    (x[:-1] - norm["x_mean"]) / norm["x_scale"],
                    (y[:-1] - norm["y_mean"]) / norm["y_scale"],
                    (velocity_x[:-1] - norm["vx_mean"]) / norm["vx_scale"],
                    (velocity_y[:-1] - norm["vy_mean"]) / norm["vy_scale"],
                    delta_t / norm["dt_s"],
                ]
            )[valid].astype(np.float32)
            targets = np.column_stack(
                [
                    np.diff(x) / norm["x_scale"],
                    np.diff(y) / norm["y_scale"],
                ]
            )[valid].astype(np.float32)
            for start in range(0, len(features), self.stride):
                end = min(start + self.sequence_length, len(features))
                if end > start:
                    sequences.append((features[start:end], targets[start:end]))
        return sequences


class CoverageAwareLoadAgent(_ArtifactOnlyAgent):

    algorithm_name = "coverage_load"

    def predict(self, state: VehicleState, task_duration_s: float) -> str:
        self._validate_inference(state, task_duration_s)
        velocity_x, velocity_y = state.velocity_xy
        future_x = state.x_m + velocity_x * task_duration_s
        future_y = state.y_m + velocity_y * task_duration_s

        def score(
            choice: tuple[EdgeServer, Any],
        ) -> tuple[float, float, str]:
            server, resource = choice
            current_ratio = (
                server.distance_to(state.x_m, state.y_m) / server.coverage_radius_m
            )
            future_ratio = (
                server.distance_to(future_x, future_y) / server.coverage_radius_m
            )
            coverage_penalty = (
                max(0.0, current_ratio - 1.0) * 20.0
                + max(0.0, future_ratio - 1.0) * 100.0
            )
            service_cost = self.environment.estimate_proxy_cost(
                server,
                resource,
                state,
                task_duration_s,
                energy_weight=2.50,
            )
            queue_depth = resource.queue_depth(state.time_s)
            risk = max(current_ratio, future_ratio)
            selection = server.selection_for(resource)
            return (
                coverage_penalty
                + 0.45 * risk
                + 0.55 * service_cost
                + 0.04 * queue_depth,
                future_ratio,
                selection,
            )

        server, resource = min(self.environment.iter_resources(), key=score)
        return server.selection_for(resource)


ALGORITHM_NAMES = (
    "greedy",
    "random",
    "uniform_random",
    "furthest",
    "kalman",
    "markov",
    "gru",
    "coverage_load",
)

_AGENT_TYPES: dict[str, type[OffloadingAgent]] = {
    "greedy": GreedyNearestAgent,
    "random": RandomServerAgent,
    "uniform_random": UniformRandomAgent,
    "furthest": FurthestServerAgent,
    "kalman": KalmanMobilityAgent,
    "markov": MarkovTrajectoryAgent,
    "gru": GRUMobilityAgent,
    "coverage_load": CoverageAwareLoadAgent,
}


def create_agent(
    algorithm: str,
    environment: EdgeEnvironment,
    artifact_directory: str | Path,
    **parameters: Any,
) -> OffloadingAgent:
    normalized = algorithm.strip().lower().replace("-", "_")
    try:
        agent_type = _AGENT_TYPES[normalized]
    except KeyError as error:
        raise ValueError(
            f"Unknown algorithm {algorithm!r}; choose from {ALGORITHM_NAMES}."
        ) from error
    return agent_type(environment, artifact_directory, **parameters)


def load_agent(
    artifact_directory: str | Path,
    *,
    expected_environment: EdgeEnvironment | None = None,
) -> OffloadingAgent:
    directory = Path(artifact_directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    algorithm = str(metadata.get("algorithm", ""))
    artifact_environment = EdgeEnvironment.from_dict(metadata["environment"])
    if (
        expected_environment is not None
        and artifact_environment.fingerprint() != expected_environment.fingerprint()
    ):
        raise ValueError(
            "Saved model environment does not match --environment-file. "
            "Retrain the model with this server configuration."
        )
    environment = (
        expected_environment.clone()
        if expected_environment is not None
        else artifact_environment
    )
    agent = create_agent(
        algorithm,
        environment,
        directory,
        seed=int(metadata.get("seed", 42)),
    )
    agent.load()
    return agent


def _safe_variance(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 1e-6
    return max(float(np.var(finite, ddof=1)), 1e-6)


def _safe_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    scale = float(np.std(finite))
    return max(scale, 1e-6)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "The GRU model requires PyTorch. Install the 'deep-learning' extra."
        ) from error
    return torch, nn


def _build_gru(nn: Any, hidden_size: int) -> Any:
    class TrajectoryGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.GRU(
                input_size=5,
                hidden_size=hidden_size,
                batch_first=True,
            )
            self.output = nn.Sequential(
                nn.Linear(hidden_size, max(1, hidden_size // 2)),
                nn.SiLU(),
                nn.Linear(max(1, hidden_size // 2), 2),
            )

        def forward(
            self,
            features: Any,
            hidden: Any | None = None,
        ) -> tuple[Any, Any]:
            encoded, next_hidden = self.recurrent(features, hidden)
            return self.output(encoded), next_hidden

    return TrajectoryGRU()


class _SequenceDataset:
    def __init__(
        self,
        sequences: list[tuple[np.ndarray, np.ndarray]],
        torch: Any,
    ) -> None:
        self.sequences = sequences
        self.torch = torch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        features, targets = self.sequences[index]
        return (
            self.torch.from_numpy(features),
            self.torch.from_numpy(targets),
        )


def _collate_sequences(batch: Iterable[tuple[Any, Any]], torch: Any) -> Any:
    values = list(batch)
    max_length = max(features.shape[0] for features, _ in values)
    feature_batch = torch.zeros((len(values), max_length, 5), dtype=torch.float32)
    target_batch = torch.zeros((len(values), max_length, 2), dtype=torch.float32)
    mask = torch.zeros((len(values), max_length), dtype=torch.float32)
    for index, (features, targets) in enumerate(values):
        length = features.shape[0]
        feature_batch[index, :length] = features
        target_batch[index, :length] = targets
        mask[index, :length] = 1.0
    return feature_batch, target_batch, mask
