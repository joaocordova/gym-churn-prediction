"""Point-in-time feature engineering (the "feature store" layer).

Every feature is computed **strictly from data on or before the snapshot
date**; the label is a cancellation in the ``label_horizon_days`` that
follow. Training uses older snapshots, the most recent snapshot is the
temporal test set — the model is always evaluated on the future, never on
shuffled rows of the past.

Feature dictionary (business logic behind each metric) — see also
``docs/feature_dictionary.md``:

===========================  ==================================================
Feature                      Why it predicts churn
===========================  ==================================================
recency_days                 Days since last check-in. The single strongest
                             disengagement signal: lapsed members cancel.
visits_7d / 30d / 90d        Visit frequency at three horizons — habit strength.
visits_per_week_90d          Frequency normalised to a weekly rate.
visit_trend_30d              Visits in the last 30 days minus the 30 days
                             before that. Negative = a member in decline.
active_weeks_ratio_12w       Share of the last 12 weeks with ≥1 visit.
                             Consistency beats intensity for habit formation.
weekly_visits_std_12w        Volatility of the weekly routine.
class_ratio_90d              Group classes create social lock-in; class-goers
                             churn less.
peak_hour_ratio_90d          Share of visits at peak (17:00–20:00) — proxies
                             a fixed routine anchored to the workday.
payment_failures_90d         Failed billing precedes price-driven churn.
late_payments_90d            Softer version of the same signal.
tenure_days                  Early-tenure members are the most fragile.
monthly_fee                  Price sensitivity lever.
plan_type                    Commitment effect: annual < quarterly < monthly.
age / gender / referral      Demographic and acquisition-channel context.
===========================  ==================================================
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from gym_churn.config import AppConfig
from gym_churn.logging_utils import get_logger
from gym_churn.simulation import SimulatedTables, load_raw_tables

log = get_logger(module="features")

TARGET = "churned_within_horizon"

NUMERIC_FEATURES = [
    "tenure_days",
    "recency_days",
    "visits_7d",
    "visits_30d",
    "visits_90d",
    "visits_per_week_90d",
    "visit_trend_30d",
    "active_weeks_ratio_12w",
    "weekly_visits_std_12w",
    "class_ratio_90d",
    "peak_hour_ratio_90d",
    "payment_failures_90d",
    "late_payments_90d",
    "monthly_fee",
    "age",
]

CATEGORICAL_FEATURES = ["plan_type", "gender", "referral_source"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ID_COLUMNS = ["member_id", "snapshot_date"]

RECENCY_CAP_DAYS = 90.0


def _window_counts(
    checkins: pd.DataFrame, snapshot: pd.Timestamp, days: int, start_days: int = 0
) -> pd.Series:
    """Check-in counts per member in the window (snapshot-days, snapshot-start_days]."""
    lo = snapshot - pd.Timedelta(days=days)
    hi = snapshot - pd.Timedelta(days=start_days)
    mask = (checkins["checkin_date"] > lo) & (checkins["checkin_date"] <= hi)
    return checkins.loc[mask].groupby("member_id").size()


def build_snapshot_frame(
    tables: SimulatedTables, snapshot_date: date, config: AppConfig
) -> pd.DataFrame:
    """Compute the full feature matrix + label for one snapshot date."""
    snapshot = pd.Timestamp(snapshot_date)
    horizon = config.snapshots.label_horizon_days
    min_tenure = config.snapshots.min_tenure_days

    members = tables.members
    cancels = tables.cancellations.set_index("member_id")["cancel_date"]

    # --- population: members active at the snapshot with enough tenure ------
    tenure_days = (snapshot - members["join_date"]).dt.days
    cancel_for = members["member_id"].map(cancels)
    active = (tenure_days >= min_tenure) & (cancel_for.isna() | (cancel_for > snapshot))
    base = members.loc[active].copy()
    base["snapshot_date"] = snapshot.date()
    base["tenure_days"] = tenure_days[active].astype(int)
    base = base.set_index("member_id")

    # --- check-in behaviour (point-in-time: only rows <= snapshot) ----------
    past = tables.checkins[tables.checkins["checkin_date"] <= snapshot]

    last_visit = past.groupby("member_id")["checkin_date"].max()
    recency = (snapshot - last_visit).dt.days.astype(float)
    base["recency_days"] = recency.reindex(base.index).fillna(RECENCY_CAP_DAYS)
    base["recency_days"] = base["recency_days"].clip(upper=RECENCY_CAP_DAYS)

    for label, days in (("visits_7d", 7), ("visits_30d", 30), ("visits_90d", 90)):
        base[label] = _window_counts(past, snapshot, days).reindex(base.index).fillna(0).astype(int)

    prior_30 = _window_counts(past, snapshot, 60, start_days=30).reindex(base.index).fillna(0)
    base["visit_trend_30d"] = (base["visits_30d"] - prior_30).astype(float)
    base["visits_per_week_90d"] = base["visits_90d"] / (90.0 / 7.0)

    # Weekly consistency over the trailing 12 weeks.
    twelve_weeks_ago = snapshot - pd.Timedelta(weeks=12)
    recent = past[past["checkin_date"] > twelve_weeks_ago].copy()
    recent["week_bucket"] = (
        (snapshot - recent["checkin_date"]).dt.days // 7
    ).clip(0, 11)
    weekly = (
        recent.groupby(["member_id", "week_bucket"]).size().unstack(fill_value=0)
    )
    weekly = weekly.reindex(columns=range(12), fill_value=0)
    base["active_weeks_ratio_12w"] = (
        (weekly > 0).sum(axis=1).reindex(base.index).fillna(0) / 12.0
    )
    base["weekly_visits_std_12w"] = weekly.std(axis=1, ddof=0).reindex(base.index).fillna(0.0)

    # Visit-mix ratios over the trailing 90 days.
    win_90 = past[past["checkin_date"] > snapshot - pd.Timedelta(days=90)]
    grp = win_90.groupby("member_id")
    total = grp.size()
    base["class_ratio_90d"] = (
        (grp["is_class"].sum() / total).reindex(base.index).fillna(0.0)
    )
    peak = win_90[win_90["hour"].between(17, 20)].groupby("member_id").size()
    base["peak_hour_ratio_90d"] = (
        (peak / total).reindex(base.index).fillna(0.0)
    )

    # --- billing signals (point-in-time) -------------------------------------
    pay_win = tables.payments[
        (tables.payments["due_date"] > snapshot - pd.Timedelta(days=90))
        & (tables.payments["due_date"] <= snapshot)
    ]
    by_status = (
        pay_win.groupby(["member_id", "status"]).size().unstack(fill_value=0)
    )
    for col, source in (("payment_failures_90d", "failed"), ("late_payments_90d", "late")):
        series = by_status[source] if source in by_status else pd.Series(dtype=int)
        base[col] = series.reindex(base.index).fillna(0).astype(int)

    # --- label: cancellation inside the forward horizon ----------------------
    horizon_end = snapshot + pd.Timedelta(days=horizon)
    cancel_series = cancel_for[active]
    cancel_series.index = base.index
    base[TARGET] = (
        cancel_series.notna()
        & (cancel_series > snapshot)
        & (cancel_series <= horizon_end)
    ).astype(int)

    frame = base.reset_index()[ID_COLUMNS + FEATURE_COLUMNS + [TARGET]]
    log.info(
        "Snapshot {snap}: {rows} active members, churn rate {rate:.2%}",
        snap=str(snapshot_date), rows=len(frame), rate=frame[TARGET].mean(),
    )
    return frame


def build_dataset(config: AppConfig, tables: SimulatedTables | None = None) -> dict[str, pd.DataFrame]:
    """Build train/test frames across all snapshots and persist them.

    Returns ``{"train": ..., "test": ...}``. Train = all snapshots except the
    most recent; test = the most recent snapshot (strictly out-of-time).
    """
    tables = tables or load_raw_tables(config)

    frames = {
        snap: build_snapshot_frame(tables, snap, config)
        for snap in config.snapshots.dates
    }
    train = pd.concat(
        [frames[s] for s in config.snapshots.train_dates], ignore_index=True
    )
    test = frames[config.snapshots.test_snapshot]

    processed = Path(config.paths.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    train.to_csv(processed / "train.csv", index=False)
    test.to_csv(processed / "test.csv", index=False)

    sample_dir = Path(config.paths.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample = test.sample(n=min(200, len(test)), random_state=config.project.random_seed)
    sample.to_csv(sample_dir / "scoring_sample.csv", index=False)

    log.info(
        "Dataset written: train={tr} rows / test={te} rows -> {dir}",
        tr=len(train), te=len(test), dir=str(processed),
    )
    return {"train": train, "test": test}


def load_processed(config: AppConfig) -> dict[str, pd.DataFrame]:
    processed = Path(config.paths.processed_dir)
    out: dict[str, pd.DataFrame] = {}
    for split in ("train", "test"):
        path = processed / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run the features pipeline first "
                "(python -m gym_churn.cli features)"
            )
        out[split] = pd.read_csv(path, parse_dates=["snapshot_date"])
    return out


def validate_feature_frame(frame: pd.DataFrame, require_target: bool = True) -> None:
    """Integrity gate for the processed feature matrix."""
    problems: list[str] = []
    expected = FEATURE_COLUMNS + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        if frame[FEATURE_COLUMNS].isna().any().any():
            nulls = frame[FEATURE_COLUMNS].isna().sum()
            problems.append(f"nulls present: {nulls[nulls > 0].to_dict()}")
        if (frame["recency_days"] < 0).any():
            problems.append("negative recency_days")
        if require_target and not set(frame[TARGET].unique()) <= {0, 1}:
            problems.append("target is not binary")
    if problems:
        raise ValueError("Feature frame failed integrity checks: " + "; ".join(problems))
