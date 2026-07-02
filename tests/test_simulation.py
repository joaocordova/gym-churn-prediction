"""Properties the simulator must guarantee."""

import numpy as np
import pandas as pd

from gym_churn.simulation import simulate


def test_simulation_is_deterministic(fast_config):
    a = simulate(fast_config)
    b = simulate(fast_config)
    assert len(a.checkins) == len(b.checkins)
    pd.testing.assert_frame_equal(a.members, b.members)
    pd.testing.assert_frame_equal(a.cancellations, b.cancellations)


def test_no_checkins_after_cancellation(raw_tables):
    cancel = raw_tables.cancellations.set_index("member_id")["cancel_date"]
    joined = raw_tables.checkins.assign(
        cancel_date=raw_tables.checkins["member_id"].map(cancel)
    ).dropna(subset=["cancel_date"])
    assert (joined["checkin_date"] <= joined["cancel_date"]).all()


def test_no_payments_after_cancellation(raw_tables):
    cancel = raw_tables.cancellations.set_index("member_id")["cancel_date"]
    joined = raw_tables.payments.assign(
        cancel_date=raw_tables.payments["member_id"].map(cancel)
    ).dropna(subset=["cancel_date"])
    assert (joined["due_date"] <= joined["cancel_date"]).all()


def test_checkins_within_window(fast_config, raw_tables):
    start = pd.Timestamp(fast_config.simulation.window_start)
    end = pd.Timestamp(fast_config.simulation.window_end)
    dates = raw_tables.checkins["checkin_date"]
    # weeks are aligned to Mondays, so allow the few days before window start
    assert dates.min() >= start - pd.Timedelta(days=7)
    assert dates.max() <= end


def test_churn_rate_is_realistic(raw_tables):
    rate = len(raw_tables.cancellations) / len(raw_tables.members)
    assert 0.10 < rate < 0.60  # realistic gym annualised churn territory


def test_gradual_churners_disengage_before_cancelling(raw_tables):
    """The core learnable signal: churners' final-month usage collapses
    relative to their own baseline (on average, across the population)."""
    cancels = raw_tables.cancellations
    checkins = raw_tables.checkins.merge(
        cancels[["member_id", "cancel_date"]], on="member_id"
    )
    days_before = (checkins["cancel_date"] - checkins["checkin_date"]).dt.days
    final_month = (days_before <= 30).groupby(checkins["member_id"]).sum()
    prior_month = ((days_before > 30) & (days_before <= 60)).groupby(
        checkins["member_id"]
    ).sum()
    both = pd.concat([final_month.rename("final"), prior_month.rename("prior")], axis=1)
    both = both[both["prior"] > 0]
    assert both["final"].mean() < 0.75 * both["prior"].mean()


def test_manifest_written(fast_config, raw_tables):
    manifest = fast_config.paths.raw_dir / "manifest.json"
    assert manifest.exists()
