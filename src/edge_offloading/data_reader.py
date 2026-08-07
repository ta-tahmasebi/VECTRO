
from __future__ import annotations

import json
import math
import tarfile
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, suppress
from hashlib import sha256
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .domain import (
    TRAJECTORY_COLUMNS,
    MobilitySnapshot,
    VehicleState,
    normalize_trajectory_frame,
)

ProgressCallback = Callable[[int, int | None, str], None]
_CACHE_SCHEMA_VERSION = 1
_EARTH_RADIUS_M = 6_371_008.8


class MobilityReader(ABC):

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    @abstractmethod
    def iter_snapshots(
        self,
        *,
        max_steps: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[MobilitySnapshot]:
        pass

    @abstractmethod
    def read_all(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        pass


class FCDReader(MobilityReader):

    def iter_snapshots(
        self,
        *,
        max_steps: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[MobilitySnapshot]:
        _validate_max_steps(max_steps)
        if not self.config.source.is_file():
            raise FileNotFoundError(f"SUMO FCD file not found: {self.config.source}")

        yielded = 0
        geographic = (
            str(self.config.options.get("coordinates", "cartesian")).lower()
            == "geographic"
        )
        origin_latitude = self.config.options.get("origin_latitude")
        origin_longitude = self.config.options.get("origin_longitude")
        origin_lat = float(origin_latitude) if origin_latitude is not None else None
        origin_lon = float(origin_longitude) if origin_longitude is not None else None
        try:
            iterator = ET.iterparse(self.config.source, events=("end",))
            for _, element in iterator:
                if _local_tag(element.tag) != "timestep":
                    continue
                if max_steps is not None and yielded >= max_steps:
                    element.clear()
                    break

                time_s = _finite_float(element.attrib.get("time"), "FCD time")
                vehicles: dict[str, VehicleState] = {}
                for vehicle in element:
                    if _local_tag(vehicle.tag) != "vehicle":
                        continue
                    identifier = vehicle.attrib.get("id", "")
                    source_x = _finite_float(vehicle.attrib.get("x"), "vehicle x")
                    source_y = _finite_float(vehicle.attrib.get("y"), "vehicle y")
                    if geographic:
                        if origin_lon is None:
                            origin_lon = source_x
                        if origin_lat is None:
                            origin_lat = source_y
                        x_m, y_m = _project_lon_lat(
                            source_x,
                            source_y,
                            origin_lon,
                            origin_lat,
                        )
                    else:
                        x_m, y_m = source_x, source_y
                    speed = _optional_float(vehicle.attrib.get("speed"), 0.0)
                    angle = _optional_float(vehicle.attrib.get("angle"), 0.0)
                    state = VehicleState(
                        vehicle_id=identifier,
                        time_s=time_s,
                        x_m=x_m,
                        y_m=y_m,
                        speed_mps=max(0.0, speed),
                        angle_deg=angle % 360.0,
                    )
                    vehicles[identifier] = state
                yielded += 1
                if progress is not None:
                    progress(yielded, max_steps, "Streaming SUMO FCD")
                yield MobilitySnapshot(time_s=time_s, vehicles=vehicles)
                element.clear()
        except ET.ParseError as error:
            raise ValueError(
                f"Malformed SUMO FCD XML in {self.config.source}: {error}"
            ) from error

    def read_all(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        cache = self.config.cache
        fingerprint = _fingerprint(
            [self.config.source],
            salt=self.config.options,
        )
        if cache is not None and _cache_is_valid(cache, fingerprint):
            result = pd.read_parquet(cache)
            return normalize_trajectory_frame(result, dataset=self.config.name)

        chunks: list[pd.DataFrame] = []
        buffered: list[dict[str, Any]] = []
        sink = _ParquetSink(cache) if cache is not None else None
        try:
            for snapshot in self.iter_snapshots(progress=progress):
                for state in snapshot.vehicles.values():
                    buffered.append(_state_record(self.config.name, state))
                if len(buffered) >= 100_000:
                    chunk = pd.DataFrame.from_records(
                        buffered, columns=TRAJECTORY_COLUMNS
                    )
                    if sink is not None:
                        sink.write(chunk)
                    else:
                        chunks.append(chunk)
                    buffered.clear()
            if buffered:
                chunk = pd.DataFrame.from_records(buffered, columns=TRAJECTORY_COLUMNS)
                if sink is not None:
                    sink.write(chunk)
                else:
                    chunks.append(chunk)
            if sink is not None:
                sink.commit()
                assert cache is not None
                _write_cache_metadata(cache, fingerprint)
                result = pd.read_parquet(cache)
            else:
                if not chunks:
                    raise ValueError(f"No vehicle rows found in {self.config.source}")
                result = pd.concat(chunks, ignore_index=True)
        except Exception:
            if sink is not None:
                sink.abort()
            raise
        return normalize_trajectory_frame(result, dataset=self.config.name)


class RawGPSReader(MobilityReader):

    def read_all(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        source_files = self._source_files()
        fingerprint = _fingerprint(source_files, salt=self.config.options)
        cache = self.config.cache
        if cache is not None and _cache_is_valid(cache, fingerprint):
            result = pd.read_parquet(cache)
            return normalize_trajectory_frame(result, dataset=self.config.name)

        chunks: list[pd.DataFrame] = []
        raw_sink: _ParquetSink | None = None
        raw_cache: Path | None = None
        if cache is not None:
            raw_cache = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.raw")
            raw_sink = _ParquetSink(raw_cache)

        processed_files = 0
        try:
            for source_path, chunk in self._iter_raw_chunks(source_files):
                canonical = self._canonicalize_chunk(chunk, source_path)
                if not canonical.empty:
                    if raw_sink is not None:
                        raw_sink.write(canonical)
                    else:
                        chunks.append(canonical)
                processed_files += 1
                if progress is not None:
                    progress(
                        processed_files,
                        len(source_files),
                        "Parsing raw GPS files",
                    )
            if raw_sink is not None:
                raw_sink.commit()
                assert raw_cache is not None
                raw = pd.read_parquet(raw_cache)
            else:
                if not chunks:
                    raise ValueError(
                        f"No usable GPS observations found under {self.config.source}"
                    )
                raw = pd.concat(chunks, ignore_index=True)

            finalized = self._finalize_raw(raw)
            if cache is not None:
                _atomic_write_parquet(finalized, cache)
                _write_cache_metadata(cache, fingerprint)
            return finalized
        except Exception:
            if raw_sink is not None:
                raw_sink.abort()
            raise
        finally:
            if raw_cache is not None and raw_cache.exists():
                raw_cache.unlink()

    def iter_snapshots(
        self,
        *,
        max_steps: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[MobilitySnapshot]:
        _validate_max_steps(max_steps)
        frame = self.read_all(progress=progress)
        for index, (time_s, group) in enumerate(
            frame.groupby("time_s", sort=True, observed=True)
        ):
            if max_steps is not None and index >= max_steps:
                break
            vehicles = {
                str(row.vehicle_id): VehicleState(
                    vehicle_id=str(row.vehicle_id),
                    time_s=float(row.time_s),
                    x_m=float(row.x_m),
                    y_m=float(row.y_m),
                    speed_mps=float(row.speed_mps),
                    angle_deg=float(row.angle_deg),
                )
                for row in group.itertuples(index=False)
            }
            yield MobilitySnapshot(time_s=float(time_s), vehicles=vehicles)

    def _source_files(self) -> list[Path]:
        source = self.config.source
        data_format = str(self.config.options.get("format", "csv")).lower()
        if source.is_file():
            files = [source]
        elif source.is_dir():
            default_pattern = {
                "geolife": "**/*.plt",
                "tdrive": "**/*.txt",
                "csv": "**/*.csv",
            }.get(data_format, "**/*")
            pattern = str(self.config.options.get("glob", default_pattern))
            files = sorted(path for path in source.glob(pattern) if path.is_file())
        else:
            raise FileNotFoundError(f"Raw GPS source does not exist: {source}")
        if not files:
            raise FileNotFoundError(f"No raw GPS files found under {source}")
        return files

    def _iter_raw_chunks(
        self,
        files: list[Path],
    ) -> Iterator[tuple[Path, pd.DataFrame]]:
        data_format = str(self.config.options.get("format", "csv")).lower()
        chunk_size = int(self.config.options.get("chunk_size", 250_000))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if data_format == "roma":
            yield from self._iter_roma(files[0], chunk_size)
            return

        for path in files:
            if data_format == "geolife":
                frame = pd.read_csv(
                    path,
                    skiprows=6,
                    header=None,
                    usecols=[0, 1, 5, 6],
                    names=["latitude", "longitude", "date", "clock"],
                )
                frame["timestamp"] = (
                    frame["date"].astype("string")
                    + " "
                    + frame["clock"].astype("string")
                )
                frame["vehicle_id"] = path.parents[1].name
                yield path, frame.drop(columns=["date", "clock"])
                continue

            if data_format == "tdrive":
                reader = pd.read_csv(
                    path,
                    header=None,
                    names=["vehicle_id", "timestamp", "longitude", "latitude"],
                    usecols=[0, 1, 2, 3],
                    chunksize=chunk_size,
                )
                for chunk in reader:
                    yield path, chunk
                continue

            if data_format != "csv":
                raise ValueError(
                    f"Unsupported raw GPS format {data_format!r}; "
                    "expected csv, geolife, tdrive, or roma."
                )
            options: dict[str, Any] = {
                "sep": self.config.options.get("delimiter", ","),
                "chunksize": chunk_size,
                "low_memory": False,
            }
            if not bool(self.config.options.get("header", True)):
                options["header"] = None
            for chunk in pd.read_csv(path, **options):
                yield path, chunk

    def _iter_roma(
        self,
        path: Path,
        chunk_size: int,
    ) -> Iterator[tuple[Path, pd.DataFrame]]:
        def chunks(stream: Any) -> Iterator[pd.DataFrame]:
            yield from pd.read_csv(
                stream,
                sep=";",
                header=None,
                names=["vehicle_id", "timestamp", "position"],
                usecols=[0, 1, 2],
                chunksize=chunk_size,
            )

        if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
            with tarfile.open(path, mode="r:gz") as archive:
                members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile()
                    and member.name.lower().endswith((".txt", ".csv"))
                ]
                if not members:
                    raise FileNotFoundError(f"No trajectory text file exists in {path}")
                extracted = archive.extractfile(members[0])
                if extracted is None:
                    raise OSError(f"Could not read {members[0].name} from {path}")
                with TextIOWrapper(
                    extracted, encoding="utf-8", errors="replace"
                ) as archive_stream:
                    for chunk in chunks(archive_stream):
                        yield path, chunk
        else:
            with path.open("r", encoding="utf-8", errors="replace") as file_stream:
                for chunk in chunks(file_stream):
                    yield path, chunk

    def _canonicalize_chunk(
        self,
        frame: pd.DataFrame,
        source_path: Path,
    ) -> pd.DataFrame:
        data_format = str(self.config.options.get("format", "csv")).lower()
        result = frame.copy()
        if data_format == "roma":
            coordinates = (
                result["position"]
                .astype("string")
                .str.extract(
                    r"POINT\s*\(\s*([-+]?\d+(?:\.\d+)?)\s+"
                    r"([-+]?\d+(?:\.\d+)?)\s*\)",
                    expand=True,
                )
            )
            result["latitude"] = coordinates[0]
            result["longitude"] = coordinates[1]
            result = result.drop(columns=["position"])
        elif data_format == "csv":
            columns = self.config.options.get("columns")
            if not isinstance(columns, Mapping):
                raise ValueError(
                    "Generic CSV datasets require a canonical 'columns' mapping."
                )
            renamed: dict[Any, str] = {}
            for canonical, source in columns.items():
                source_key: Any = int(source) if isinstance(source, int) else source
                if source_key not in result.columns:
                    raise ValueError(
                        f"Configured CSV column {source!r} is absent in {source_path}"
                    )
                renamed[source_key] = str(canonical)
            result = result.rename(columns=renamed)

        allowed = {
            "vehicle_id",
            "timestamp",
            "time_s",
            "latitude",
            "longitude",
            "x_m",
            "y_m",
            "speed_mps",
            "angle_deg",
        }
        selected = [column for column in result.columns if column in allowed]
        result = result.loc[:, selected]
        if "vehicle_id" not in result:
            result["vehicle_id"] = source_path.stem
        return result

    def _finalize_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "timestamp" in result:
            timestamps = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
            result["timestamp"] = timestamps
            if "time_s" not in result or result["time_s"].isna().all():
                start = timestamps.min()
                result["time_s"] = (timestamps - start).dt.total_seconds()
        if "time_s" not in result:
            raise ValueError("Raw GPS data needs either timestamp or time_s.")

        for column in ("latitude", "longitude", "x_m", "y_m"):
            if column in result:
                result[column] = pd.to_numeric(result[column], errors="coerce")
        has_xy = {"x_m", "y_m"}.issubset(result.columns) and result[
            ["x_m", "y_m"]
        ].notna().any(axis=None)
        has_geo = {"latitude", "longitude"}.issubset(result.columns)
        if not has_xy:
            if not has_geo:
                raise ValueError(
                    "Raw GPS data needs x_m/y_m or latitude/longitude coordinates."
                )
            valid = result["latitude"].between(-90.0, 90.0) & result[
                "longitude"
            ].between(-180.0, 180.0)
            result = result.loc[valid].copy()
            if result.empty:
                raise ValueError("No valid latitude/longitude observations found.")
            origin_lat = float(result["latitude"].median())
            origin_lon = float(result["longitude"].median())
            latitude_rad = np.deg2rad(result["latitude"].to_numpy(float))
            longitude_rad = np.deg2rad(result["longitude"].to_numpy(float))
            origin_lat_rad = math.radians(origin_lat)
            result["x_m"] = (
                _EARTH_RADIUS_M
                * (longitude_rad - math.radians(origin_lon))
                * math.cos(origin_lat_rad)
            )
            result["y_m"] = _EARTH_RADIUS_M * (latitude_rad - origin_lat_rad)
        return normalize_trajectory_frame(result, dataset=self.config.name)


class SumoTraCIReader(MobilityReader, AbstractContextManager["SumoTraCIReader"]):

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self._connection: Any | None = None
        self._traci: Any | None = None
        self._constants: Any | None = None
        self._anchor_id: str | None = None
        self._label = f"edge-offloading-{uuid.uuid4().hex}"

    def start(self) -> None:
        if self._connection is not None:
            raise RuntimeError("SUMO reader is already running.")
        if not self.config.source.is_file():
            raise FileNotFoundError(
                f"SUMO configuration not found: {self.config.source}"
            )
        try:
            import traci
            import traci.constants as constants
        except ImportError as error:
            raise RuntimeError(
                "SUMO TraCI is not installed. Install the 'simulation' extra "
                "or add SUMO_HOME/tools to PYTHONPATH."
            ) from error

        binary = "sumo-gui" if bool(self.config.options.get("gui", False)) else "sumo"
        command = [
            str(self.config.options.get("binary", binary)),
            "-c",
            str(self.config.source),
            "--no-warnings",
        ]
        step_length = self.config.options.get("step_length")
        if step_length is not None:
            command.extend(["--step-length", str(float(step_length))])
        extra = self.config.options.get("sumo_args", [])
        if not isinstance(extra, list):
            raise ValueError("sumo_args must be a list of command arguments.")
        command.extend(str(item) for item in extra)

        try:
            traci.start(command, label=self._label)
            connection = traci.getConnection(self._label)
            boundary = connection.simulation.getNetBoundary()
            junction_ids = list(connection.junction.getIDList())
            if not junction_ids:
                raise RuntimeError("SUMO network contains no junctions.")
            min_xy, max_xy = boundary
            center_x = (float(min_xy[0]) + float(max_xy[0])) / 2.0
            center_y = (float(min_xy[1]) + float(max_xy[1])) / 2.0
            anchor = min(
                junction_ids,
                key=lambda identifier: _squared_distance(
                    connection.junction.getPosition(identifier),
                    (center_x, center_y),
                ),
            )
            anchor_position = connection.junction.getPosition(anchor)
            corners = (
                (min_xy[0], min_xy[1]),
                (min_xy[0], max_xy[1]),
                (max_xy[0], min_xy[1]),
                (max_xy[0], max_xy[1]),
            )
            radius = max(math.dist(anchor_position, corner) for corner in corners) + 1.0
            connection.junction.subscribeContext(
                anchor,
                constants.CMD_GET_VEHICLE_VARIABLE,
                radius,
                [
                    constants.VAR_POSITION,
                    constants.VAR_SPEED,
                    constants.VAR_ANGLE,
                ],
            )
        except Exception:
            with suppress(Exception):
                traci.close(False)
            raise

        self._traci = traci
        self._constants = constants
        self._connection = connection
        self._anchor_id = anchor

    def close(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None
            self._anchor_id = None

    def __enter__(self) -> SumoTraCIReader:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def iter_snapshots(
        self,
        *,
        max_steps: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Iterator[MobilitySnapshot]:
        _validate_max_steps(max_steps)
        self.start()
        steps = 0
        assert self._connection is not None
        assert self._constants is not None
        assert self._anchor_id is not None
        connection = self._connection
        constants = self._constants
        try:
            while connection.simulation.getMinExpectedNumber() > 0:
                if max_steps is not None and steps >= max_steps:
                    break
                connection.simulationStep()
                time_s = float(connection.simulation.getTime())
                subscribed = (
                    connection.junction.getContextSubscriptionResults(self._anchor_id)
                    or {}
                )
                vehicles: dict[str, VehicleState] = {}
                for identifier, values in subscribed.items():
                    position = values.get(constants.VAR_POSITION)
                    if position is None:
                        continue
                    state = VehicleState(
                        vehicle_id=str(identifier),
                        time_s=time_s,
                        x_m=float(position[0]),
                        y_m=float(position[1]),
                        speed_mps=max(0.0, float(values.get(constants.VAR_SPEED, 0.0))),
                        angle_deg=float(values.get(constants.VAR_ANGLE, 0.0)) % 360.0,
                    )
                    vehicles[state.vehicle_id] = state
                steps += 1
                if progress is not None:
                    progress(steps, max_steps, "Running SUMO via TraCI")
                yield MobilitySnapshot(time_s=time_s, vehicles=vehicles)
        finally:
            self.close()

    def read_all(
        self,
        *,
        progress: ProgressCallback | None = None,
    ) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for snapshot in self.iter_snapshots(progress=progress):
            records.extend(
                _state_record(self.config.name, state)
                for state in snapshot.vehicles.values()
            )
        if not records:
            raise ValueError("SUMO simulation produced no vehicle observations.")
        return normalize_trajectory_frame(
            pd.DataFrame.from_records(records, columns=TRAJECTORY_COLUMNS),
            dataset=self.config.name,
        )


class DataReaderFactory:

    @classmethod
    def create(cls, config: DatasetConfig) -> MobilityReader:
        if config.reader == "fcd":
            return FCDReader(config)
        if config.reader == "raw_gps":
            return RawGPSReader(config)
        if config.reader == "sumo":
            return SumoTraCIReader(config)
        raise ValueError(f"No reader registered for {config.reader!r}.")


def _state_record(dataset: str, state: VehicleState) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "vehicle_id": state.vehicle_id,
        "time_s": state.time_s,
        "x_m": state.x_m,
        "y_m": state.y_m,
        "speed_mps": state.speed_mps,
        "angle_deg": state.angle_deg,
        "latitude": np.nan,
        "longitude": np.nan,
    }


def _validate_max_steps(max_steps: int | None) -> None:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when supplied.")


def _finite_float(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite.")
    return converted


def _optional_float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _squared_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _fingerprint(
    paths: list[Path],
    *,
    salt: Mapping[str, Any] | None = None,
) -> str:
    digest = sha256()
    for path in sorted(paths):
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    if salt is not None:
        digest.update(
            json.dumps(dict(salt), sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def _project_lon_lat(
    longitude: float,
    latitude: float,
    origin_longitude: float,
    origin_latitude: float,
) -> tuple[float, float]:
    origin_latitude_rad = math.radians(origin_latitude)
    return (
        _EARTH_RADIUS_M
        * math.radians(longitude - origin_longitude)
        * math.cos(origin_latitude_rad),
        _EARTH_RADIUS_M * math.radians(latitude - origin_latitude),
    )


def _metadata_path(cache: Path) -> Path:
    return cache.with_suffix(cache.suffix + ".metadata.json")


def _cache_is_valid(cache: Path, fingerprint: str) -> bool:
    metadata_path = _metadata_path(cache)
    if not cache.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == _CACHE_SCHEMA_VERSION
        and metadata.get("source_fingerprint") == fingerprint
    )


def _write_cache_metadata(cache: Path, fingerprint: str) -> None:
    metadata_path = _metadata_path(cache)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "source_fingerprint": fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(metadata_path)


class _ParquetSink:

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        self._writer: Any | None = None

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "Parquet caching requires pyarrow; install project dependencies."
            ) from error
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression="zstd",
            )
        self._writer.write_table(table)

    def commit(self) -> None:
        if self._writer is None:
            raise ValueError("Cannot commit an empty Parquet cache.")
        self._writer.close()
        self._writer = None
        self.temporary.replace(self.destination)

    def abort(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self.temporary.exists():
            self.temporary.unlink()


def _atomic_write_parquet(frame: pd.DataFrame, destination: Path) -> None:
    sink = _ParquetSink(destination)
    try:
        sink.write(frame)
        sink.commit()
    except Exception:
        sink.abort()
        raise
