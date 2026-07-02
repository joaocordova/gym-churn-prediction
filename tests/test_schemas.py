"""Data contracts must accept clean data and reject every corruption mode."""

import pandas as pd
import pytest

from gym_churn.schemas import (
    DataContractError,
    MEMBERS_CONTRACT,
    ScoringRequest,
    validate_dataset,
)


def test_micro_tables_pass_contracts(micro_tables):
    validate_dataset(
        micro_tables.members,
        micro_tables.checkins,
        micro_tables.payments,
        micro_tables.cancellations,
    )


def test_rejects_null_values(micro_tables):
    members = micro_tables.members.copy()
    members.loc[0, "monthly_fee"] = None
    with pytest.raises(DataContractError, match="null"):
        MEMBERS_CONTRACT.validate(members)


def test_rejects_duplicate_member_ids(micro_tables):
    members = pd.concat([micro_tables.members, micro_tables.members.iloc[[0]]])
    with pytest.raises(DataContractError, match="duplicate"):
        MEMBERS_CONTRACT.validate(members)


def test_rejects_out_of_range_fee(micro_tables):
    members = micro_tables.members.copy()
    members.loc[0, "monthly_fee"] = -10.0
    with pytest.raises(DataContractError, match="monthly_fee"):
        MEMBERS_CONTRACT.validate(members)


def test_rejects_unknown_plan(micro_tables):
    members = micro_tables.members.copy()
    members.loc[0, "plan_type"] = "platinum_forever"
    with pytest.raises(DataContractError, match="plan"):
        MEMBERS_CONTRACT.validate(members)


def test_rejects_empty_frame(micro_tables):
    with pytest.raises(DataContractError, match="empty"):
        MEMBERS_CONTRACT.validate(micro_tables.members.iloc[0:0])


def test_rejects_orphan_checkins(micro_tables):
    checkins = micro_tables.checkins.copy()
    checkins.loc[0, "member_id"] = 999
    with pytest.raises(DataContractError, match="unknown member_ids"):
        validate_dataset(
            micro_tables.members, checkins,
            micro_tables.payments, micro_tables.cancellations,
        )


def test_rejects_cancel_before_join(micro_tables):
    cancellations = micro_tables.cancellations.copy()
    cancellations.loc[0, "cancel_date"] = pd.Timestamp("2020-01-01")
    with pytest.raises(DataContractError, match="cancel_date"):
        validate_dataset(
            micro_tables.members, micro_tables.checkins,
            micro_tables.payments, cancellations,
        )


def test_scoring_request_accepts_valid_payload():
    ScoringRequest(
        tenure_days=200, recency_days=3.0, visits_7d=2, visits_30d=9,
        visits_90d=30, visits_per_week_90d=2.3, visit_trend_30d=-1.0,
        active_weeks_ratio_12w=0.75, weekly_visits_std_12w=1.1,
        class_ratio_90d=0.2, peak_hour_ratio_90d=0.5,
        payment_failures_90d=0, late_payments_90d=1, monthly_fee=62.0,
        age=31, plan_type="monthly", gender="female", referral_source="friend",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("recency_days", -1.0),
        ("active_weeks_ratio_12w", 1.4),
        ("age", 130),
        ("plan_type", "unknown_plan"),
        ("monthly_fee", 0.0),
    ],
)
def test_scoring_request_rejects_bad_values(field, value):
    payload = dict(
        tenure_days=200, recency_days=3.0, visits_7d=2, visits_30d=9,
        visits_90d=30, visits_per_week_90d=2.3, visit_trend_30d=-1.0,
        active_weeks_ratio_12w=0.75, weekly_visits_std_12w=1.1,
        class_ratio_90d=0.2, peak_hour_ratio_90d=0.5,
        payment_failures_90d=0, late_payments_90d=1, monthly_fee=62.0,
        age=31, plan_type="monthly", gender="female", referral_source="friend",
    )
    payload[field] = value
    with pytest.raises(Exception):
        ScoringRequest(**payload)
