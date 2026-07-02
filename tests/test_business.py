"""The financial math must be exactly right — it is the number executives see."""

import numpy as np
import pytest

from gym_churn.business import campaign_outcome, optimal_threshold, profit_curve
from gym_churn.config import load_config


@pytest.fixture(scope="module")
def business():
    return load_config().business


def test_campaign_outcome_hand_computed(business):
    # 4 members: two true churners with the top scores, threshold 0.5 flags 2.
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.3, 0.1])
    out = campaign_outcome(y_true, y_prob, business, threshold=0.5)

    assert out["n_flagged"] == 2
    assert out["true_churners_flagged"] == 2
    assert out["precision"] == 1.0
    expected_revenue = business.save_rate * 2 * business.value_per_save
    assert out["retained_revenue"] == pytest.approx(expected_revenue, abs=0.01)
    assert out["campaign_cost"] == pytest.approx(2 * business.contact_cost)
    assert out["expected_profit"] == pytest.approx(
        expected_revenue - 2 * business.contact_cost, abs=0.01
    )


def test_member_level_fees_are_used(business):
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])
    fees = np.array([100.0, 10.0])
    out = campaign_outcome(y_true, y_prob, business, threshold=0.5, fees=fees)
    assert out["retained_revenue"] == pytest.approx(
        business.save_rate * 100.0 * business.retention_months, abs=0.01
    )


def test_top_fraction_targeting(business):
    y_true = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.linspace(0.95, 0.05, 10)
    out = campaign_outcome(y_true, y_prob, business, top_fraction=0.1)
    assert out["n_flagged"] == 1
    assert out["true_churners_flagged"] == 1  # the churner has the top score


def test_requires_exactly_one_targeting_mode(business):
    y = np.array([0, 1])
    p = np.array([0.2, 0.8])
    with pytest.raises(ValueError):
        campaign_outcome(y, p, business)
    with pytest.raises(ValueError):
        campaign_outcome(y, p, business, threshold=0.5, top_fraction=0.1)


def test_optimal_threshold_maximises_profit(business):
    rng = np.random.default_rng(0)
    y_prob = rng.beta(1.2, 8, 3000)
    y_true = (rng.random(3000) < y_prob).astype(int)  # perfectly calibrated world
    threshold, best = optimal_threshold(y_true, y_prob, business)
    curve = profit_curve(y_true, y_prob, business)
    assert best["expected_profit"] == pytest.approx(curve["expected_profit"].max())
    assert 0.0 <= threshold <= 1.0


def test_flagging_nobody_costs_nothing(business):
    y = np.array([0, 1, 0])
    p = np.array([0.1, 0.2, 0.3])
    out = campaign_outcome(y, p, business, threshold=0.99)
    assert out["n_flagged"] == 0
    assert out["campaign_cost"] == 0.0
    assert out["expected_profit"] == 0.0
