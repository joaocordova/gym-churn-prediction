"""Point-in-time correctness of the feature pipeline, on hand-checkable data.

The micro fixture (see conftest) is designed so every expected value below
can be verified by reading the fixture — including the check-in that occurs
AFTER the snapshot, which must never leak into any feature.
"""

import pandas as pd
import pytest

from gym_churn.features import (
    FEATURE_COLUMNS,
    TARGET,
    build_snapshot_frame,
    validate_feature_frame,
)

SNAPSHOT = pd.Timestamp("2026-03-31").date()


@pytest.fixture()
def frame(micro_tables, request):
    from gym_churn.config import load_config

    config = load_config()
    return build_snapshot_frame(micro_tables, SNAPSHOT, config).set_index("member_id")


def test_population_excludes_low_tenure_members(frame):
    # Member 3 joined 2026-03-20 — only 11 days of tenure at the snapshot.
    assert 3 not in frame.index
    assert set(frame.index) == {1, 2}


def test_recency_and_window_counts(frame):
    # Member 1: last visit 2026-03-25 → recency 6 days.
    assert frame.loc[1, "recency_days"] == 6.0
    # Visits: 03-25 (7d), + 03-10 (30d), 02-10 falls in the prior-30d bucket.
    assert frame.loc[1, "visits_7d"] == 1
    assert frame.loc[1, "visits_30d"] == 2
    assert frame.loc[1, "visits_90d"] == 3
    # Trend: 2 visits in last 30d minus 1 visit in the 30d before that.
    assert frame.loc[1, "visit_trend_30d"] == 1.0


def test_future_checkin_never_leaks(frame):
    # The 2026-04-05 check-in is after the snapshot; if it leaked, member 1
    # would have visits_90d == 4 and recency 0 at a later date.
    assert frame.loc[1, "visits_90d"] == 3


def test_recency_capped_for_dormant_member(frame):
    # Member 2's last visit is 2026-01-15 → 75 days; within cap.
    assert frame.loc[2, "recency_days"] == 75.0


def test_payment_window_counts(frame):
    # Member 1: one failed (03-01) + one late (02-01) inside the 90d window.
    assert frame.loc[1, "payment_failures_90d"] == 1
    assert frame.loc[1, "late_payments_90d"] == 1
    assert frame.loc[2, "payment_failures_90d"] == 0


def test_label_horizon(frame):
    # Member 2 cancels 2026-04-20 — inside the 30-day horizon after 03-31.
    assert frame.loc[2, TARGET] == 1
    assert frame.loc[1, TARGET] == 0


def test_class_ratio(frame):
    # Member 1's 90d visits: 03-25 (class), 03-10 (not), 02-10 (not) → 1/3.
    assert frame.loc[1, "class_ratio_90d"] == pytest.approx(1 / 3)


def test_no_nulls_and_schema(frame):
    validate_feature_frame(frame.reset_index())
    assert list(frame.reset_index()[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS


def test_validate_feature_frame_catches_nulls(frame):
    bad = frame.reset_index().copy()
    bad.loc[0, "recency_days"] = None
    with pytest.raises(ValueError, match="integrity"):
        validate_feature_frame(bad)


def test_dataset_split_is_temporal(dataset):
    train, test = dataset["train"], dataset["test"]
    assert pd.to_datetime(train["snapshot_date"]).max() < pd.to_datetime(
        test["snapshot_date"]
    ).min()
    # Both classes present in both splits, and prevalence is plausibly imbalanced.
    for split in (train, test):
        rate = split[TARGET].mean()
        assert 0.005 < rate < 0.30
