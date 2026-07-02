"""Lightweight, file-based experiment tracking (MLflow-style, zero servers).

Every training run gets a directory under ``experiments/runs/`` containing:

* ``meta.json``     — run id, name, tags, status, timestamps, environment
* ``params.json``   — hyperparameters and configuration used
* ``metrics.json``  — all logged metrics (latest value per key)
* ``artifacts/``    — copied model files, plots, reports

An append-only ``experiments/index.jsonl`` gives a queryable history of every
run — one JSON object per line, greppable and diff-friendly. The format is
deliberately plain files: reviewable in a PR, no tracking server to operate.
"""

from __future__ import annotations

import json
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gym_churn.logging_utils import get_logger

log = get_logger(module="tracking")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


class Run:
    """A single tracked experiment run. Use via ``ExperimentTracker.start_run``."""

    def __init__(self, root: Path, name: str, tags: dict[str, str]):
        started = datetime.now(timezone.utc)
        self.run_id = f"{started.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.name = name
        self.dir = root / "runs" / f"{self.run_id}_{name}"
        self.artifacts_dir = self.dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = root / "index.jsonl"
        self._params: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}
        self._meta: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "tags": tags,
            "status": "running",
            "started_at": started.isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        self._flush()
        log.info("Started run {run_id} ({name})", run_id=self.run_id, name=name)

    # ------------------------------------------------------------------ #

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)
        self._flush()

    def log_metrics(self, metrics: dict[str, float]) -> None:
        clean = {k: float(v) for k, v in metrics.items()}
        self._metrics.update(clean)
        log.info("Metrics: {metrics}", metrics=clean)
        self._flush()

    def log_dict(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.artifacts_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default))
        return path

    def log_artifact(self, source: str | Path) -> Path:
        source = Path(source)
        target = self.artifacts_dir / source.name
        shutil.copy2(source, target)
        return target

    def finish(self, status: str = "completed") -> None:
        self._meta["status"] = status
        self._meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._flush()
        with self._index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {**self._meta, "params": self._params, "metrics": self._metrics},
                    default=_json_default,
                )
                + "\n"
            )
        log.info("Run {run_id} finished ({status})", run_id=self.run_id, status=status)

    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        (self.dir / "meta.json").write_text(
            json.dumps(self._meta, indent=2, default=_json_default)
        )
        (self.dir / "params.json").write_text(
            json.dumps(self._params, indent=2, default=_json_default)
        )
        (self.dir / "metrics.json").write_text(
            json.dumps(self._metrics, indent=2, default=_json_default)
        )

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(status="failed" if exc_type else "completed")


class ExperimentTracker:
    def __init__(self, experiments_dir: str | Path):
        self.root = Path(experiments_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def start_run(self, name: str, tags: dict[str, str] | None = None) -> Run:
        return Run(self.root, name=name, tags=tags or {})

    def history(self) -> list[dict[str, Any]]:
        """All completed runs, most recent last."""
        index = self.root / "index.jsonl"
        if not index.exists():
            return []
        return [
            json.loads(line)
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
