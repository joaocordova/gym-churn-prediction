"""Structured logging for every pipeline stage.

Two sinks are configured:

* a human-readable console sink for interactive runs, and
* a machine-readable JSON-lines sink under ``logs/`` so pipeline executions
  are auditable after the fact (each record carries module, function, elapsed
  time and any structured ``extra`` fields).
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


def setup_logging(logs_dir: str | Path | None = None, level: str = "INFO") -> None:
    """Configure loguru sinks once per process (idempotent)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logger.remove()
    logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT, enqueue=False)

    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            logs_dir / "run_{time:YYYYMMDD_HHmmss}.jsonl",
            level="DEBUG",
            serialize=True,  # structured JSON lines — one event per line
            rotation="20 MB",
            retention=10,
            enqueue=False,
        )

    _CONFIGURED = True


def get_logger(**context: object):
    """Return a logger bound with structured context fields."""
    return logger.bind(**context) if context else logger
