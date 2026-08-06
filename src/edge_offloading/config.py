
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class DatasetConfig:

    name: str
    reader: str
    source: Path
    cache: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        supported = {"fcd", "sumo", "raw_gps"}
        if not self.name:
            raise ValueError("Dataset name must not be empty.")
        if self.reader not in supported:
            raise ValueError(
                f"Unsupported reader {self.reader!r}; expected one of {supported}."
            )


@dataclass(frozen=True, slots=True)
class DatasetRegistry:

    datasets: Mapping[str, DatasetConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetRegistry:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Dataset configuration not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}

        raw_datasets = document.get("datasets")
        if not isinstance(raw_datasets, dict) or not raw_datasets:
            raise ValueError("Configuration must define a non-empty 'datasets' map.")

        base_dir = config_path.parent
        parsed: dict[str, DatasetConfig] = {}
        for name, raw in raw_datasets.items():
            if not isinstance(raw, dict):
                raise ValueError(f"Dataset {name!r} must be a mapping.")
            if "reader" not in raw or "source" not in raw:
                raise ValueError(f"Dataset {name!r} must define 'reader' and 'source'.")
            source = _resolve_path(base_dir, str(raw["source"]))
            cache_value = raw.get("cache")
            cache = (
                _resolve_path(base_dir, str(cache_value))
                if cache_value is not None
                else None
            )
            options = {
                key: value
                for key, value in raw.items()
                if key not in {"reader", "source", "cache"}
            }
            parsed[str(name)] = DatasetConfig(
                name=str(name),
                reader=str(raw["reader"]).lower(),
                source=source,
                cache=cache,
                options=options,
            )
        return cls(parsed)

    def get(self, name: str) -> DatasetConfig:
        try:
            return self.datasets[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.datasets))
            raise KeyError(
                f"Unknown dataset {name!r}. Configured datasets: {choices}"
            ) from error


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
