"""Command-line entry point for every pipeline stage.

Usage::

    python -m gym_churn.cli simulate    # generate + validate raw data
    python -m gym_churn.cli features    # point-in-time feature engineering
    python -m gym_churn.cli train       # tracked model bake-off + calibration
    python -m gym_churn.cli evaluate    # test metrics, plots, business report
    python -m gym_churn.cli explain     # SHAP global explainability assets
    python -m gym_churn.cli all         # the whole thing, in order
"""

from __future__ import annotations

import argparse
import sys

from gym_churn.config import load_config
from gym_churn.logging_utils import get_logger, setup_logging

log = get_logger(module="cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gym-churn", description="Gym churn prediction pipeline"
    )
    parser.add_argument(
        "stage",
        choices=["simulate", "features", "train", "evaluate", "explain", "all"],
        help="pipeline stage to run",
    )
    parser.add_argument(
        "--config", default=None, help="path to an alternative config.yaml"
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    setup_logging(config.paths.logs_dir)

    stages = (
        ["simulate", "features", "train", "evaluate", "explain"]
        if args.stage == "all"
        else [args.stage]
    )
    for stage in stages:
        log.info("=== stage: {stage} ===", stage=stage)
        if stage == "simulate":
            from gym_churn.simulation import run_simulation

            run_simulation(config)
        elif stage == "features":
            from gym_churn.features import build_dataset

            build_dataset(config)
        elif stage == "train":
            from gym_churn.train import run_training

            run_training(config)
        elif stage == "evaluate":
            from gym_churn.evaluate import run_evaluation

            run_evaluation(config)
        elif stage == "explain":
            from gym_churn.explain import run_explain

            run_explain(config)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
