"""Synthetic data simulator for a multi-location gym chain.

Real membership data is proprietary, so this module generates a dataset that
mirrors the *shape and failure modes* of production gym data. The generative
story (documented in ``docs/data_generation.md``):

* Each member has a latent **engagement** level that drives how often they
  visit, and a latent **payment reliability** that drives billing failures.
* Churn follows a monthly hazard model: low engagement, monthly (no-commitment)
  plans and billing failures all raise the hazard.
* **~75% of churners disengage gradually** — their visits decay over the final
  6–10 weeks. This is the recoverable signal the model must learn.
* **~25% churn abruptly** (relocation, health) with no behavioural warning.
  These are intentionally near-unpredictable so evaluation metrics stay
  honest — a simulator without irreducible noise produces fantasy AUCs.

Outputs four raw CSVs (members, checkins, payments, cancellations) plus a
``manifest.json`` with row counts and generation parameters for auditability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gym_churn.config import AppConfig
from gym_churn.logging_utils import get_logger
from gym_churn.schemas import validate_dataset

log = get_logger(module="simulation")

LOCATIONS = ["downtown", "riverside", "northgate", "midtown"]
REFERRALS = ["instagram", "friend", "walk_in", "corporate", "website"]
REFERRAL_P = [0.28, 0.30, 0.16, 0.11, 0.15]

PLANS = ["monthly", "quarterly", "annual"]
PLAN_P = [0.55, 0.25, 0.20]
PLAN_FEE_MEAN = {"monthly": 66.0, "quarterly": 58.0, "annual": 49.0}
# Monthly cancellation hazard by plan — commitment reduces churn.
PLAN_BASE_HAZARD = {"monthly": 0.052, "quarterly": 0.028, "annual": 0.013}

ABRUPT_CHURN_SHARE = 0.20  # churners with no behavioural warning signs


@dataclass(frozen=True)
class SimulatedTables:
    members: pd.DataFrame
    checkins: pd.DataFrame
    payments: pd.DataFrame
    cancellations: pd.DataFrame


def _seasonal_multiplier(week_starts: pd.DatetimeIndex) -> np.ndarray:
    """New-year resolution spike, December slump, mild summer dip."""
    factor = np.ones(len(week_starts))
    factor[week_starts.month == 1] = 1.30
    factor[week_starts.month == 12] = 0.80
    factor[week_starts.month == 6] = 0.95
    return factor


def simulate(config: AppConfig) -> SimulatedTables:
    sim = config.simulation
    rng = np.random.default_rng(config.project.random_seed)
    n = sim.n_members

    window_start = pd.Timestamp(sim.window_start)
    window_end = pd.Timestamp(sim.window_end)

    # ------------------------------------------------------------------ #
    # Members: latent traits first, observable attributes second.        #
    # ------------------------------------------------------------------ #
    engagement = rng.beta(2.0, 2.2, n)              # drives visit frequency
    fail_propensity = rng.beta(1.2, 30.0, n)        # drives payment failures
    class_propensity = rng.beta(2.0, 5.0, n)        # share of visits that are classes
    evening_person = rng.random(n) < 0.55           # preferred check-in window

    plan = rng.choice(PLANS, size=n, p=PLAN_P)
    fee = np.round(
        np.array([PLAN_FEE_MEAN[p] for p in plan]) + rng.normal(0, 6.0, n), 2
    ).clip(25.0, 180.0)

    max_join = window_end - pd.Timedelta(days=45)
    join_offsets = rng.integers(0, 700, n)  # up to ~2 years of history
    join_dates = pd.to_datetime(max_join) - pd.to_timedelta(join_offsets, unit="D")

    members = pd.DataFrame(
        {
            "member_id": np.arange(1, n + 1),
            "join_date": join_dates.normalize(),
            "plan_type": plan,
            "monthly_fee": fee,
            "age": rng.normal(34, 11, n).clip(16, 75).astype(int),
            "gender": rng.choice(
                ["female", "male", "other"], size=n, p=[0.47, 0.50, 0.03]
            ),
            "home_location": rng.choice(LOCATIONS, size=n),
            "referral_source": rng.choice(REFERRALS, size=n, p=REFERRAL_P),
        }
    )

    # ------------------------------------------------------------------ #
    # Churn: month-by-month hazard model over the observation window.    #
    # ------------------------------------------------------------------ #
    base_hazard = np.array([PLAN_BASE_HAZARD[p] for p in plan])
    hazard = base_hazard * (2.1 - 1.8 * engagement) * (1.0 + 9.0 * fail_propensity)
    hazard = hazard.clip(0.002, 0.35)

    month_starts = pd.date_range(window_start, window_end, freq="MS")
    cancel_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    for month_start in month_starts:
        month_end = month_start + pd.offsets.MonthEnd(0)
        eligible = (
            pd.isna(cancel_date)
            & (members["join_date"].values <= (month_start - pd.Timedelta(days=21)))
        )
        churns_now = eligible & (rng.random(n) < hazard)
        days_in_month = (month_end - month_start).days + 1
        offsets = rng.integers(0, days_in_month, n)
        candidate = month_start.to_datetime64() + offsets * np.timedelta64(1, "D")
        cancel_date = np.where(churns_now, candidate, cancel_date)

    churned = ~pd.isna(cancel_date)
    abrupt = churned & (rng.random(n) < ABRUPT_CHURN_SHARE)

    # ------------------------------------------------------------------ #
    # Check-ins: weekly Poisson counts with seasonality and, for gradual #
    # churners, a linear engagement decay over their final weeks.        #
    # ------------------------------------------------------------------ #
    week_starts = pd.date_range(
        window_start - pd.Timedelta(days=window_start.weekday()),
        window_end,
        freq="W-MON",
    )
    n_weeks = len(week_starts)
    seasonal = _seasonal_multiplier(week_starts)

    base_rate = 0.3 + 5.0 * engagement
    rate = base_rate[:, None] * seasonal[None, :]  # (members, weeks)

    week_grid = week_starts.values[None, :]
    joined_mask = week_grid >= members["join_date"].values[:, None]
    active_end = np.where(churned, cancel_date, window_end.to_datetime64())
    active_mask = joined_mask & (week_grid <= active_end[:, None])
    rate = rate * active_mask

    decay_weeks = rng.integers(8, 15, n)  # gradual churners fade over 8–14 weeks
    weeks_to_cancel = (
        (active_end[:, None] - week_grid) / np.timedelta64(7, "D")
    ).astype(float)
    gradual = (churned & ~abrupt)[:, None]
    in_decay = gradual & (weeks_to_cancel >= 0) & (weeks_to_cancel < decay_weeks[:, None])
    # Convex decay (exponent > 1): disengagement is already pronounced weeks
    # before the cancellation lands — the window a 30-day model can act on.
    ratio = np.clip(weeks_to_cancel / decay_weeks[:, None], 0.0, 1.0)
    decay_factor = np.where(in_decay, 0.03 + 0.97 * ratio**1.5, 1.0)
    rate = rate * decay_factor

    counts = rng.poisson(rate.clip(0, 20))
    member_idx, week_idx = np.nonzero(counts)
    reps = counts[member_idx, week_idx]
    flat_member = np.repeat(members["member_id"].values[member_idx], reps)
    flat_week = np.repeat(week_starts.values[week_idx], reps)

    day_offset = rng.integers(0, 7, len(flat_member))
    checkin_dates = flat_week + day_offset * np.timedelta64(1, "D")

    flat_evening = np.repeat(evening_person[member_idx], reps)
    hours = np.where(
        flat_evening,
        rng.normal(18.5, 1.5, len(flat_member)),
        rng.normal(7.5, 1.8, len(flat_member)),
    ).round().clip(5, 23).astype(int)

    flat_class_p = np.repeat(class_propensity[member_idx], reps)
    checkins = pd.DataFrame(
        {
            "member_id": flat_member,
            "checkin_date": pd.to_datetime(checkin_dates).normalize(),
            "hour": hours,
            "is_class": rng.random(len(flat_member)) < flat_class_p,
        }
    )
    # Clip stray same-week check-ins that would land after a cancellation.
    end_by_member = pd.Series(active_end, index=members["member_id"].values)
    checkins = checkins[
        checkins["checkin_date"].values
        <= end_by_member.loc[checkins["member_id"]].values
    ].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Payments: monthly billing from join date while the member is       #
    # active; failures driven by the latent reliability trait.           #
    # ------------------------------------------------------------------ #
    payment_rows: list[pd.DataFrame] = []
    for k in range(0, 26):  # covers the full observation window
        due = members["join_date"] + pd.DateOffset(months=k)
        in_window = (due >= window_start) & (due <= window_end)
        still_active = due.values <= active_end
        mask = (in_window & still_active).values
        if not mask.any():
            continue
        m_fail = fail_propensity[mask]
        u = rng.random(mask.sum())
        status = np.where(
            u < m_fail * 3.0, "failed", np.where(u < m_fail * 3.0 + 0.06, "late", "paid")
        )
        payment_rows.append(
            pd.DataFrame(
                {
                    "member_id": members["member_id"].values[mask],
                    "due_date": due[mask].dt.normalize(),
                    "amount": members["monthly_fee"].values[mask],
                    "status": status,
                }
            )
        )
    payments = (
        pd.concat(payment_rows, ignore_index=True)
        .sort_values(["member_id", "due_date"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ #
    # Cancellations: reasons correlate with the causal driver.           #
    # ------------------------------------------------------------------ #
    churn_ids = members["member_id"].values[churned]
    churn_cancel = pd.to_datetime(cancel_date[churned])
    churn_abrupt = abrupt[churned]
    churn_fail = fail_propensity[churned]

    reasons = np.empty(len(churn_ids), dtype=object)
    u = rng.random(len(churn_ids))
    for i in range(len(churn_ids)):
        if churn_abrupt[i]:
            reasons[i] = "relocation" if u[i] < 0.5 else ("health" if u[i] < 0.8 else "other")
        elif churn_fail[i] > 0.06:
            reasons[i] = "price" if u[i] < 0.7 else "low_usage"
        else:
            reasons[i] = (
                "low_usage" if u[i] < 0.62 else ("price" if u[i] < 0.82 else "service")
            )

    cancellations = pd.DataFrame(
        {
            "member_id": churn_ids,
            "cancel_date": churn_cancel.normalize(),
            "reason": reasons,
        }
    ).sort_values("cancel_date").reset_index(drop=True)

    log.info(
        "Simulated dataset: {m} members, {c} check-ins, {p} payments, {x} cancellations",
        m=len(members), c=len(checkins), p=len(payments), x=len(cancellations),
    )
    return SimulatedTables(members, checkins, payments, cancellations)


def run_simulation(config: AppConfig) -> SimulatedTables:
    """Simulate, validate against the data contracts, and persist raw CSVs."""
    tables = simulate(config)

    validate_dataset(
        tables.members, tables.checkins, tables.payments, tables.cancellations
    )
    log.info("All raw tables passed their data contracts")

    raw_dir = Path(config.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("members", "checkins", "payments", "cancellations"):
        frame: pd.DataFrame = getattr(tables, name)
        frame.to_csv(raw_dir / f"{name}.csv", index=False)

    manifest = {
        "generator": "gym_churn.simulation",
        "random_seed": config.project.random_seed,
        "n_members": config.simulation.n_members,
        "window": [str(config.simulation.window_start), str(config.simulation.window_end)],
        "row_counts": {
            "members": len(tables.members),
            "checkins": len(tables.checkins),
            "payments": len(tables.payments),
            "cancellations": len(tables.cancellations),
        },
        "overall_churn_rate": round(len(tables.cancellations) / len(tables.members), 4),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Raw data written to {dir}", dir=str(raw_dir))
    return tables


def load_raw_tables(config: AppConfig) -> SimulatedTables:
    """Load previously simulated raw tables, re-validating the contracts."""
    raw_dir = Path(config.paths.raw_dir)
    parse = {
        "members": ["join_date"],
        "checkins": ["checkin_date"],
        "payments": ["due_date"],
        "cancellations": ["cancel_date"],
    }
    frames = {
        name: pd.read_csv(raw_dir / f"{name}.csv", parse_dates=dates)
        for name, dates in parse.items()
    }
    validate_dataset(**frames)
    return SimulatedTables(**frames)
