# Synthetic data: generative assumptions

Real membership data is proprietary, so the project ships a simulator
(`gym_churn/simulation.py`) that reproduces the *shape and failure modes* of
production gym data rather than a toy dataset. Everything below is encoded in
code and reproducible under `project.random_seed`.

## The behavioural model

Each of the 12,000 members draws latent traits:

| Trait | Distribution | Drives |
|---|---|---|
| `engagement` | Beta(2.0, 2.2) | Weekly visit rate: `0.3 + 5.0 × engagement` |
| `fail_propensity` | Beta(1.2, 30) | Payment failure probability and price-driven churn |
| `class_propensity` | Beta(2.0, 5.0) | Share of visits that are group classes |
| `evening_person` | Bernoulli(0.55) | Check-in hour profile (evening vs morning peak) |

Visits are weekly Poisson counts with **seasonality** (January +30%
resolution spike, December −20%, June −5%) over a 9-month observation window
(2025-10 → 2026-06).

## Churn mechanics

Monthly cancellation hazard:

```
hazard = plan_base × (2.1 − 1.8 × engagement) × (1 + 9 × fail_propensity)
plan_base:  monthly 5.2% · quarterly 2.8% · annual 1.3%   (commitment effect)
```

Two churn archetypes:

- **Gradual (~80%)** — visits decay convexly (`0.03 + 0.97 × r^1.5`) over the
  final **8–14 weeks** before the cancellation. This is the recoverable
  signal a 30-day-horizon model can act on.
- **Abrupt (~20%)** — relocation/health/other; usage stays normal until the
  cancellation. **Intentionally near-unpredictable.** A simulator without
  irreducible noise produces fantasy AUCs; this one keeps the ceiling honest
  (test ROC-AUC lands in the high 0.80s, like real churn problems).

Cancellation *reasons* correlate with the causal driver (high fail-propensity
→ "price", low engagement → "low_usage", abrupt → "relocation"/"health").

## Billing

Monthly billing on the join-date anniversary while active. Payment status is
sampled from `fail_propensity` (≈3× multiplier on failures, ~6% late rate) —
so billing trouble both *predicts* and *causes* churn, exactly the
correlation structure a real model must exploit.

## What the simulator guarantees (tested in `tests/test_simulation.py`)

- Deterministic under the seed.
- No check-ins or payments after a member's cancellation date.
- Churners' final-month usage is measurably below their own baseline.
- Overall churn lands in the realistic 10–60% annualised band.
- All four tables pass the Pydantic/frame data contracts on write and read.
