
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
import pandas as pd

TRAJECTORY_COLUMNS: Final[tuple[str, ...]] = (
    "dataset",
    "vehicle_id",
    "time_s",
    "x_m",
    "y_m",
    "speed_mps",
    "angle_deg",
    "latitude",
    "longitude",
)


@dataclass(frozen=True, slots=True)
class VehicleState:

    vehicle_id: str
    time_s: float
    x_m: float
    y_m: float
    speed_mps: float
    angle_deg: float

    def __post_init__(self) -> None:
        numeric = (
            self.time_s,
            self.x_m,
            self.y_m,
            self.speed_mps,
            self.angle_deg,
        )
        if not self.vehicle_id:
            raise ValueError("vehicle_id must not be empty.")
        if not all(isfinite(value) for value in numeric):
            raise ValueError("Vehicle-state values must be finite.")
        if self.speed_mps < 0:
            raise ValueError("speed_mps must be non-negative.")

    @property
    def velocity_xy(self) -> tuple[float, float]:
        radians = np.deg2rad(self.angle_deg)
        return (
            float(self.speed_mps * np.sin(radians)),
            float(self.speed_mps * np.cos(radians)),
        )


@dataclass(frozen=True, slots=True)
class MobilitySnapshot:

    time_s: float
    vehicles: Mapping[str, VehicleState]


def normalize_trajectory_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    required = {"vehicle_id", "time_s", "x_m", "y_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Trajectory data is missing columns: {missing}")

    result = frame.copy()
    result["dataset"] = dataset
    for column in TRAJECTORY_COLUMNS:
        if column not in result:
            result[column] = np.nan

    result["vehicle_id"] = result["vehicle_id"].astype("string")
    numeric_columns = [
        "time_s",
        "x_m",
        "y_m",
        "speed_mps",
        "angle_deg",
        "latitude",
        "longitude",
    ]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["vehicle_id", "time_s", "x_m", "y_m"])
    result = result[np.isfinite(result[["time_s", "x_m", "y_m"]]).all(axis=1)]
    result = result.sort_values(
        ["vehicle_id", "time_s"], kind="stable"
    ).drop_duplicates(["vehicle_id", "time_s"], keep="last")

    grouped = result.groupby("vehicle_id", sort=False, observed=True)
    delta_t = grouped["time_s"].diff()
    delta_x = grouped["x_m"].diff()
    delta_y = grouped["y_m"].diff()
    valid_delta = delta_t.gt(0)

    derived_speed = np.hypot(delta_x, delta_y).div(delta_t.where(valid_delta))
    derived_angle = np.degrees(np.arctan2(delta_x, delta_y)) % 360.0

    result["speed_mps"] = result["speed_mps"].where(
        result["speed_mps"].ge(0) & np.isfinite(result["speed_mps"]),
        derived_speed,
    )
    result["angle_deg"] = result["angle_deg"].where(
        np.isfinite(result["angle_deg"]),
        derived_angle,
    )
    result["speed_mps"] = (
        result.groupby("vehicle_id", observed=True)["speed_mps"]
        .transform(lambda values: values.bfill().fillna(0.0))
        .clip(lower=0.0)
    )
    result["angle_deg"] = (
        result.groupby("vehicle_id", observed=True)["angle_deg"].transform(
            lambda values: values.bfill().fillna(0.0)
        )
        % 360.0
    )

    if result.empty:
        raise ValueError("No valid trajectory observations were found.")
    return result.loc[:, TRAJECTORY_COLUMNS].reset_index(drop=True)


def states_from_frame(frame: pd.DataFrame) -> list[VehicleState]:
    return [
        VehicleState(
            vehicle_id=str(row.vehicle_id),
            time_s=float(row.time_s),
            x_m=float(row.x_m),
            y_m=float(row.y_m),
            speed_mps=float(row.speed_mps),
            angle_deg=float(row.angle_deg),
        )
        for row in frame.itertuples(index=False)
    ]
