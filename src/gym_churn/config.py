"""Typed configuration for the churn system.

The single YAML file at ``configs/config.yaml`` is parsed into the Pydantic
models below. Anything invalid (a negative cost, a test snapshot that is not
one of the snapshot dates, a malformed date) fails at load time — before any
compute is spent.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

_CONFIG_ENV_VAR = "GYM_CHURN_CONFIG"
_DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


class _StrictModel(BaseModel):
    """Reject unknown keys so config typos surface immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_StrictModel):
    name: str
    random_seed: int = Field(ge=0)


class PathsConfig(_StrictModel):
    """All paths are declared relative to the repository root."""

    raw_dir: Path
    processed_dir: Path
    sample_dir: Path
    models_dir: Path
    assets_dir: Path
    img_dir: Path
    experiments_dir: Path
    logs_dir: Path

    def resolved(self, root: Path) -> "PathsConfig":
        return PathsConfig(**{name: root / value for name, value in self})


class SimulationConfig(_StrictModel):
    n_members: int = Field(ge=100, le=1_000_000)
    window_start: date
    window_end: date

    @model_validator(mode="after")
    def _window_ordered(self) -> "SimulationConfig":
        if self.window_end <= self.window_start:
            raise ValueError("simulation.window_end must be after window_start")
        return self


class SnapshotConfig(_StrictModel):
    dates: list[date] = Field(min_length=1)
    test_snapshot: date
    label_horizon_days: int = Field(ge=7, le=180)
    min_tenure_days: int = Field(ge=0)

    @model_validator(mode="after")
    def _test_snapshot_known(self) -> "SnapshotConfig":
        if self.test_snapshot not in self.dates:
            raise ValueError("snapshots.test_snapshot must be one of snapshots.dates")
        if self.test_snapshot != max(self.dates):
            raise ValueError(
                "snapshots.test_snapshot must be the most recent snapshot "
                "(train on the past, test on the future)"
            )
        return self

    @property
    def train_dates(self) -> list[date]:
        return [d for d in self.dates if d != self.test_snapshot]


class TrainingConfig(_StrictModel):
    cv_folds: int = Field(ge=2, le=10)
    n_search_iter: int = Field(ge=1, le=500)
    scoring: str
    validation_fraction: float = Field(gt=0.0, lt=0.5)
    calibration: str

    @field_validator("calibration")
    @classmethod
    def _calibration_known(cls, value: str) -> str:
        if value not in {"auto", "none"}:
            raise ValueError("training.calibration must be 'auto' or 'none'")
        return value


class BusinessConfig(_StrictModel):
    """Financial levers that translate probabilities into dollars."""

    currency: str
    currency_symbol: str
    avg_monthly_fee: float = Field(gt=0)
    contact_cost: float = Field(ge=0)
    save_rate: float = Field(gt=0, le=1)
    retention_months: float = Field(gt=0)
    top_fraction: float = Field(gt=0, le=1)

    @property
    def value_per_save(self) -> float:
        """Revenue gained when one would-be churner is retained."""
        return self.avg_monthly_fee * self.retention_months


class AppConfig(_StrictModel):
    project: ProjectConfig
    paths: PathsConfig
    simulation: SimulationConfig
    snapshots: SnapshotConfig
    training: TrainingConfig
    business: BusinessConfig

    @property
    def root(self) -> Path:
        return REPO_ROOT


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load, validate and path-resolve the system configuration.

    Resolution order: explicit ``path`` argument, the ``GYM_CHURN_CONFIG``
    environment variable, then the repository default.
    """
    config_path = Path(path or os.environ.get(_CONFIG_ENV_VAR) or _DEFAULT_CONFIG)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    config = AppConfig(**payload)
    return config.model_copy(update={"paths": config.paths.resolved(REPO_ROOT)})


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Cached accessor for the default configuration."""
    return load_config()
