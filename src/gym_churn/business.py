"""Financial translation layer — probabilities in, dollars out.

The campaign economics implemented here (all levers live in
``configs/config.yaml`` under ``business:``):

* Contacting a flagged member costs ``contact_cost`` (staff time + incentive).
* A correctly flagged churner is retained with probability ``save_rate``.
* A retained member keeps paying their fee for ``retention_months`` more
  months, so one save is worth ``fee × retention_months``.

Profit of a campaign = saved revenue − outreach cost. This is the objective
the decision threshold is optimised against, and the language the model's
value is reported in — a hiring gym manager does not buy ROC-AUC, they buy
retained recurring revenue.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gym_churn.config import AppConfig, BusinessConfig
from gym_churn.logging_utils import get_logger

log = get_logger(module="business")


def campaign_outcome(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    business: BusinessConfig,
    threshold: float | None = None,
    top_fraction: float | None = None,
    fees: np.ndarray | None = None,
) -> dict:
    """Economics of contacting members above a threshold or in the top-k%.

    Exactly one of ``threshold`` / ``top_fraction`` must be provided. When
    member-level ``fees`` are given, saved revenue uses each member's actual
    fee; otherwise the configured average fee.
    """
    if (threshold is None) == (top_fraction is None):
        raise ValueError("provide exactly one of threshold or top_fraction")

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if top_fraction is not None:
        n_flagged = max(1, int(round(len(y_prob) * top_fraction)))
        flagged = np.zeros(len(y_prob), dtype=bool)
        flagged[np.argsort(-y_prob)[:n_flagged]] = True
        threshold = float(np.sort(y_prob)[-n_flagged])
    else:
        flagged = y_prob >= threshold

    n_flagged = int(flagged.sum())
    true_churners_flagged = int((y_true[flagged] == 1).sum())

    member_value = (
        np.asarray(fees) * business.retention_months
        if fees is not None
        else np.full(len(y_true), business.value_per_save)
    )
    saved_revenue = float(
        business.save_rate * member_value[flagged & (y_true == 1)].sum()
    )
    cost = float(n_flagged * business.contact_cost)

    return {
        "threshold": float(threshold),
        "n_scored": int(len(y_true)),
        "n_flagged": n_flagged,
        "flagged_fraction": round(n_flagged / max(len(y_true), 1), 4),
        "true_churners_flagged": true_churners_flagged,
        "precision": round(true_churners_flagged / max(n_flagged, 1), 4),
        "recall": round(true_churners_flagged / max(int(y_true.sum()), 1), 4),
        "expected_saves": round(business.save_rate * true_churners_flagged, 1),
        "retained_revenue": round(saved_revenue, 2),
        "campaign_cost": round(cost, 2),
        "expected_profit": round(saved_revenue - cost, 2),
        "roi": round((saved_revenue - cost) / cost, 2) if cost > 0 else float("inf"),
    }


def profit_curve(
    y_true: np.ndarray, y_prob: np.ndarray, business: BusinessConfig
) -> pd.DataFrame:
    """Expected profit as a function of the decision threshold."""
    thresholds = np.unique(np.quantile(y_prob, np.linspace(0.0, 0.995, 200)))
    rows = [
        campaign_outcome(y_true, y_prob, business, threshold=float(t))
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, business: BusinessConfig
) -> tuple[float, dict]:
    """Threshold that maximises expected campaign profit."""
    curve = profit_curve(y_true, y_prob, business)
    best = curve.loc[curve["expected_profit"].idxmax()]
    return float(best["threshold"]), best.to_dict()


def build_business_report(
    test: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    config: AppConfig,
) -> dict:
    """Full financial narrative for the temporal test snapshot, persisted to
    ``assets/business_impact.json``."""
    business = config.business
    y_true = test["churned_within_horizon"].to_numpy()
    fees = test["monthly_fee"].to_numpy()

    monthly_revenue = float(fees.sum())
    revenue_walking_out = float(fees[y_true == 1].sum())

    at_threshold = campaign_outcome(
        y_true, y_prob, business, threshold=threshold, fees=fees
    )
    top_k = campaign_outcome(
        y_true, y_prob, business, top_fraction=business.top_fraction, fees=fees
    )

    # Baseline: same budget spent on randomly chosen members. Expected
    # precision equals the base churn rate.
    base_rate = float(y_true.mean())
    n_random = top_k["n_flagged"]
    random_saves = business.save_rate * base_rate * n_random
    random_revenue = random_saves * float(fees.mean()) * business.retention_months
    random_cost = n_random * business.contact_cost
    random_profit = random_revenue - random_cost

    report = {
        "snapshot": str(config.snapshots.test_snapshot),
        "currency": business.currency,
        "portfolio": {
            "active_members": int(len(test)),
            "monthly_recurring_revenue": round(monthly_revenue, 2),
            "observed_churn_rate_30d": round(base_rate, 4),
            "monthly_revenue_lost_to_churn": round(revenue_walking_out, 2),
        },
        "campaign_at_optimal_threshold": at_threshold,
        "campaign_top_decile": top_k,
        "random_targeting_baseline": {
            "n_flagged": n_random,
            "expected_profit": round(random_profit, 2),
        },
        "model_uplift_vs_random": round(
            top_k["expected_profit"] - random_profit, 2
        ),
        "assumptions": {
            "save_rate": business.save_rate,
            "retention_months": business.retention_months,
            "contact_cost": business.contact_cost,
        },
    }

    assets = Path(config.paths.assets_dir)
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "business_impact.json").write_text(json.dumps(report, indent=2))
    log.info(
        "Business report: top-decile campaign profit {p} {c} "
        "({u} uplift vs random targeting)",
        p=report["campaign_top_decile"]["expected_profit"],
        c=business.currency,
        u=report["model_uplift_vs_random"],
    )
    return report
