# Feature dictionary

Every feature is computed **point-in-time**: only data on or before the
snapshot date is used. The label (`churned_within_horizon`) is a cancellation
in the 30 days *after* the snapshot. Population: members active at the
snapshot with ≥ 30 days of tenure.

## Engagement (check-in behaviour)

| Feature | Window | Definition | Business logic |
|---|---|---|---|
| `recency_days` | 90d cap | Days since last check-in | The single strongest disengagement signal — lapsed members cancel. Capped at 90 so "never visited recently" is a stable value. |
| `visits_7d` / `visits_30d` / `visits_90d` | 7/30/90d | Check-in counts | Habit strength at three horizons. |
| `visits_per_week_90d` | 90d | `visits_90d / 12.86` | Frequency normalised to a weekly rate. |
| `visit_trend_30d` | 60d | visits last 30d − visits prior 30d | **Momentum.** A negative trend is a member in decline even if absolute usage is still decent. |
| `active_weeks_ratio_12w` | 12w | Share of last 12 weeks with ≥ 1 visit | Consistency beats intensity for habit formation; a 2×/week regular is safer than a binge-and-vanish member with the same total. |
| `weekly_visits_std_12w` | 12w | Std-dev of weekly visit counts | Routine volatility. |
| `class_ratio_90d` | 90d | Share of visits that are group classes | Classes create social lock-in; class-goers churn less. |
| `peak_hour_ratio_90d` | 90d | Share of visits 17:00–20:00 | Proxies a fixed routine anchored to the workday. |

## Billing

| Feature | Window | Definition | Business logic |
|---|---|---|---|
| `payment_failures_90d` | 90d | Count of failed payments | Failed billing precedes price-driven churn — often the first hard signal. |
| `late_payments_90d` | 90d | Count of late payments | Softer version of the same signal. |
| `monthly_fee` | — | Member's monthly price | Price-sensitivity lever; also used member-level in the profit math. |

## Membership & demographics

| Feature | Definition | Business logic |
|---|---|---|
| `tenure_days` | Days since joining | Early-tenure members are the most fragile; churn hazard falls with habit age. |
| `plan_type` | monthly / quarterly / annual | Commitment effect: annual < quarterly < monthly churn. |
| `age`, `gender` | Demographics | Context; weak but non-zero signal. |
| `referral_source` | instagram / friend / walk_in / corporate / website | Acquisition-channel quality — friend-referred members stick, walk-ins are volatile. |

## Label

| Column | Definition |
|---|---|
| `churned_within_horizon` | 1 if the member's `cancel_date` falls in `(snapshot, snapshot + 30 days]`, else 0. |

## Snapshot design

```
2026-03-31 ──► features from history ≤ 03-31, label from (03-31, 04-30]  ─┐ train
2026-04-30 ──► features from history ≤ 04-30, label from (04-30, 05-30]  ─┘
2026-05-31 ──► features from history ≤ 05-31, label from (05-31, 06-30]  ── test (out-of-time)
```

The same member appears in several snapshots, so **every** split — CV folds,
validation holdout — is grouped by `member_id` to prevent identity leakage.
