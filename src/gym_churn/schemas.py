"""Data contracts for every table that enters or leaves the system.

Two layers of defence:

1. **Row contracts** — Pydantic models (one per raw table) that encode types,
   ranges and enumerations. Because validating a million check-in rows one by
   one is wasteful, row contracts are applied to a deterministic sample of
   each frame (``ROW_SAMPLE_SIZE``) to catch value-level violations.
2. **Frame contracts** — vectorised checks that run on the *entire* frame:
   required columns, null policy, uniqueness, numeric ranges and
   cross-table referential integrity.

Any violation raises :class:`DataContractError` with a full list of failures,
so a bad extract stops the pipeline before any compute is spent.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Callable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

ROW_SAMPLE_SIZE = 2_000  # rows per table spot-checked with the Pydantic contract


class DataContractError(ValueError):
    """Raised when a frame violates its declared contract."""

    def __init__(self, table: str, failures: list[str]):
        self.table = table
        self.failures = failures
        bullet_list = "\n".join(f"  - {failure}" for failure in failures)
        super().__init__(f"Data contract violated for '{table}':\n{bullet_list}")


# ---------------------------------------------------------------------------
# Enumerations shared across the system
# ---------------------------------------------------------------------------


class PlanType(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class Gender(str, Enum):
    FEMALE = "female"
    MALE = "male"
    OTHER = "other"


class ReferralSource(str, Enum):
    INSTAGRAM = "instagram"
    FRIEND = "friend"
    WALK_IN = "walk_in"
    CORPORATE = "corporate"
    WEBSITE = "website"


class PaymentStatus(str, Enum):
    PAID = "paid"
    LATE = "late"
    FAILED = "failed"


class CancellationReason(str, Enum):
    LOW_USAGE = "low_usage"
    PRICE = "price"
    RELOCATION = "relocation"
    SERVICE = "service"
    HEALTH = "health"
    OTHER = "other"


# ---------------------------------------------------------------------------
# Row contracts (one per raw table)
# ---------------------------------------------------------------------------


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemberRecord(_Row):
    member_id: int = Field(ge=1)
    join_date: date
    plan_type: PlanType
    monthly_fee: float = Field(gt=0, lt=500)
    age: int = Field(ge=14, le=90)
    gender: Gender
    home_location: str = Field(min_length=1)
    referral_source: ReferralSource


class CheckinRecord(_Row):
    member_id: int = Field(ge=1)
    checkin_date: date
    hour: int = Field(ge=5, le=23)  # gym operating hours
    is_class: bool


class PaymentRecord(_Row):
    member_id: int = Field(ge=1)
    due_date: date
    amount: float = Field(gt=0, lt=500)
    status: PaymentStatus


class CancellationRecord(_Row):
    member_id: int = Field(ge=1)
    cancel_date: date
    reason: CancellationReason


# ---------------------------------------------------------------------------
# Inference contract (what the scoring API / dashboard accepts)
# ---------------------------------------------------------------------------


class ScoringRequest(_Row):
    """A single member feature vector submitted for scoring.

    Mirrors the output of the feature pipeline; ranges are deliberately
    generous (they bound physically possible values, not the training
    distribution — drift detection is a monitoring concern, not a schema one).
    """

    tenure_days: int = Field(ge=0, le=10_000)
    recency_days: float = Field(ge=0, le=365)
    visits_7d: int = Field(ge=0, le=40)
    visits_30d: int = Field(ge=0, le=150)
    visits_90d: int = Field(ge=0, le=450)
    visits_per_week_90d: float = Field(ge=0, le=35)
    visit_trend_30d: float = Field(ge=-150, le=150)
    active_weeks_ratio_12w: float = Field(ge=0, le=1)
    weekly_visits_std_12w: float = Field(ge=0, le=30)
    class_ratio_90d: float = Field(ge=0, le=1)
    peak_hour_ratio_90d: float = Field(ge=0, le=1)
    payment_failures_90d: int = Field(ge=0, le=12)
    late_payments_90d: int = Field(ge=0, le=12)
    monthly_fee: float = Field(gt=0, lt=500)
    age: int = Field(ge=14, le=90)
    plan_type: PlanType
    gender: Gender
    referral_source: ReferralSource


# ---------------------------------------------------------------------------
# Frame contracts
# ---------------------------------------------------------------------------


class FrameContract:
    """Vectorised, whole-frame validation for one raw table."""

    def __init__(
        self,
        name: str,
        row_model: type[_Row],
        required_columns: list[str],
        unique_key: list[str] | None = None,
        checks: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] | None = None,
    ):
        self.name = name
        self.row_model = row_model
        self.required_columns = required_columns
        self.unique_key = unique_key
        self.checks = checks or []

    def validate(self, frame: pd.DataFrame) -> None:
        failures: list[str] = []

        missing = [c for c in self.required_columns if c not in frame.columns]
        if missing:
            raise DataContractError(self.name, [f"missing columns: {missing}"])

        if frame.empty:
            raise DataContractError(self.name, ["frame is empty"])

        nulls = frame[self.required_columns].isna().sum()
        for column, count in nulls[nulls > 0].items():
            failures.append(f"column '{column}' has {count} null values")

        if self.unique_key is not None:
            dupes = int(frame.duplicated(subset=self.unique_key).sum())
            if dupes:
                failures.append(f"{dupes} duplicate rows on key {self.unique_key}")

        for description, predicate in self.checks:
            bad = int((~predicate(frame)).sum())
            if bad:
                failures.append(f"{bad} rows violate: {description}")

        failures.extend(self._spot_check_rows(frame))

        if failures:
            raise DataContractError(self.name, failures)

    def _spot_check_rows(self, frame: pd.DataFrame) -> list[str]:
        """Validate a deterministic sample of rows against the Pydantic model."""
        sample = frame.head(ROW_SAMPLE_SIZE // 2)
        if len(frame) > len(sample):
            tail = frame.tail(ROW_SAMPLE_SIZE - len(sample))
            sample = pd.concat([sample, tail])
        failures: list[str] = []
        for record in sample.to_dict(orient="records"):
            try:
                self.row_model(**record)
            except Exception as exc:  # noqa: BLE001 — collect, then fail loudly
                failures.append(f"row contract: {exc}")
                if len(failures) >= 5:
                    failures.append("... (further row failures suppressed)")
                    break
        return failures


MEMBERS_CONTRACT = FrameContract(
    name="members",
    row_model=MemberRecord,
    required_columns=[
        "member_id", "join_date", "plan_type", "monthly_fee",
        "age", "gender", "home_location", "referral_source",
    ],
    unique_key=["member_id"],
    checks=[
        ("monthly_fee in (0, 500)", lambda f: (f["monthly_fee"] > 0) & (f["monthly_fee"] < 500)),
        ("age in [14, 90]", lambda f: f["age"].between(14, 90)),
        ("plan_type is a known plan", lambda f: f["plan_type"].isin([p.value for p in PlanType])),
    ],
)

CHECKINS_CONTRACT = FrameContract(
    name="checkins",
    row_model=CheckinRecord,
    required_columns=["member_id", "checkin_date", "hour", "is_class"],
    checks=[
        ("hour within operating hours [5, 23]", lambda f: f["hour"].between(5, 23)),
    ],
)

PAYMENTS_CONTRACT = FrameContract(
    name="payments",
    row_model=PaymentRecord,
    required_columns=["member_id", "due_date", "amount", "status"],
    checks=[
        ("amount in (0, 500)", lambda f: (f["amount"] > 0) & (f["amount"] < 500)),
        ("status is a known status", lambda f: f["status"].isin([s.value for s in PaymentStatus])),
    ],
)

CANCELLATIONS_CONTRACT = FrameContract(
    name="cancellations",
    row_model=CancellationRecord,
    required_columns=["member_id", "cancel_date", "reason"],
    unique_key=["member_id"],
    checks=[
        ("reason is a known reason", lambda f: f["reason"].isin([r.value for r in CancellationReason])),
    ],
)


def validate_dataset(
    members: pd.DataFrame,
    checkins: pd.DataFrame,
    payments: pd.DataFrame,
    cancellations: pd.DataFrame,
) -> None:
    """Validate all raw tables plus cross-table referential integrity."""
    MEMBERS_CONTRACT.validate(members)
    CHECKINS_CONTRACT.validate(checkins)
    PAYMENTS_CONTRACT.validate(payments)
    CANCELLATIONS_CONTRACT.validate(cancellations)

    known = set(members["member_id"])
    failures: list[str] = []
    for name, frame in (
        ("checkins", checkins),
        ("payments", payments),
        ("cancellations", cancellations),
    ):
        orphans = int((~frame["member_id"].isin(known)).sum())
        if orphans:
            failures.append(f"{name}: {orphans} rows reference unknown member_ids")

    joined = cancellations.merge(members[["member_id", "join_date"]], on="member_id")
    bad_dates = int(
        (pd.to_datetime(joined["cancel_date"]) <= pd.to_datetime(joined["join_date"])).sum()
    )
    if bad_dates:
        failures.append(f"cancellations: {bad_dates} cancel_date on/before join_date")

    if failures:
        raise DataContractError("dataset", failures)
