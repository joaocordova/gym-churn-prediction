"""Shared fixtures.

``trained_pipeline`` runs the real pipeline (simulate → features → train) on a
small, fast configuration in a temp directory once per test session, so the
integration tests exercise exactly the code paths production uses — no mocks
of our own modules.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from gym_churn.config import AppConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fast_config(root: Path) -> Path:
    """Write a small/fast variant of the default config with temp paths."""
    payload = yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())
    payload["simulation"]["n_members"] = 900
    payload["training"]["n_search_iter"] = 4
    payload["training"]["cv_folds"] = 2
    for key in payload["paths"]:
        payload["paths"][key] = str(root / Path(payload["paths"][key]).name)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    return config_path


@pytest.fixture(scope="session")
def fast_config(tmp_path_factory) -> AppConfig:
    root = tmp_path_factory.mktemp("gym_churn_pipeline")
    return load_config(_fast_config(root))


@pytest.fixture(scope="session")
def raw_tables(fast_config):
    from gym_churn.simulation import run_simulation

    return run_simulation(fast_config)


@pytest.fixture(scope="session")
def dataset(fast_config, raw_tables):
    from gym_churn.features import build_dataset

    return build_dataset(fast_config, raw_tables)


@pytest.fixture(scope="session")
def trained_pipeline(fast_config, dataset) -> AppConfig:
    from gym_churn.train import run_training

    run_training(fast_config)
    return fast_config


@pytest.fixture()
def micro_tables():
    """Hand-built raw tables with known, hand-checkable values."""
    from gym_churn.simulation import SimulatedTables

    members = pd.DataFrame(
        {
            "member_id": [1, 2, 3],
            "join_date": pd.to_datetime(["2025-06-01", "2025-11-15", "2026-03-20"]),
            "plan_type": ["monthly", "annual", "monthly"],
            "monthly_fee": [60.0, 45.0, 70.0],
            "age": [30, 45, 22],
            "gender": ["female", "male", "other"],
            "home_location": ["downtown", "riverside", "downtown"],
            "referral_source": ["friend", "website", "instagram"],
        }
    )
    checkins = pd.DataFrame(
        {
            "member_id": [1, 1, 1, 2, 2, 1],
            "checkin_date": pd.to_datetime(
                [
                    "2026-03-25",  # inside 7d window of snapshot 2026-03-31
                    "2026-03-10",  # inside 30d window
                    "2026-02-10",  # inside 60d window (prior-30d bucket)
                    "2026-01-15",  # member 2: inside 90d window
                    "2025-12-01",  # member 2: outside 90d window
                    "2026-04-05",  # AFTER the snapshot — must never leak
                ]
            ),
            "hour": [18, 7, 19, 12, 9, 18],
            "is_class": [True, False, False, False, True, True],
        }
    )
    payments = pd.DataFrame(
        {
            "member_id": [1, 1, 2, 3],
            "due_date": pd.to_datetime(
                ["2026-03-01", "2026-02-01", "2026-03-15", "2026-03-20"]
            ),
            "amount": [60.0, 60.0, 45.0, 70.0],
            "status": ["failed", "late", "paid", "paid"],
        }
    )
    cancellations = pd.DataFrame(
        {
            "member_id": [2],
            "cancel_date": pd.to_datetime(["2026-04-20"]),  # within 30d of snapshot
            "reason": ["low_usage"],
        }
    )
    return SimulatedTables(members, checkins, payments, cancellations)
