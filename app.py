"""Gym Churn Command Center — interactive scoring & explainability dashboard.

Run with::

    streamlit run app.py

Three views:

1. **Score a member** — enter a member's behaviour, get a calibrated churn
   probability, a risk tier, the outreach recommendation and a SHAP
   explanation of *why*.
2. **Portfolio radar** — the scored test-month portfolio: risk-tier mix,
   the ranked outreach list and expected campaign economics.
3. **Model report** — headline metrics and the interactive evaluation plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from gym_churn.config import get_config
from gym_churn.explain import member_waterfall_figure
from gym_churn.features import FEATURE_COLUMNS
from gym_churn.plotting import SERIES, apply_theme
from gym_churn.predict import ChurnScorer

st.set_page_config(page_title="Gym Churn Command Center", layout="wide")

TIER_COLORS = {"low": SERIES[1], "medium": SERIES[2], "high": SERIES[5]}


@st.cache_resource
def load_scorer() -> ChurnScorer:
    return ChurnScorer.load(get_config().paths.models_dir)


@st.cache_data
def load_sample() -> pd.DataFrame:
    config = get_config()
    return pd.read_csv(Path(config.paths.sample_dir) / "scoring_sample.csv")


@st.cache_data
def load_json(name: str) -> dict:
    config = get_config()
    path = Path(config.paths.assets_dir) / name
    return json.loads(path.read_text()) if path.exists() else {}


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    color = SERIES[5] if probability >= threshold else (
        SERIES[2] if probability >= threshold / 2 else SERIES[1]
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": color, "thickness": 0.35},
                "threshold": {
                    "line": {"color": "#0b0b0b", "width": 2},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
            title={"text": "30-day churn probability"},
        )
    )
    return apply_theme(fig, height=300, margin=dict(l=30, r=30, t=60, b=10))


def main() -> None:
    config = get_config()
    try:
        scorer = load_scorer()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run the pipeline first:\n\n"
            "```\npython -m gym_churn.cli all\n```"
        )
        st.stop()

    st.title("Gym Churn Command Center")
    st.caption(
        f"Model: **{scorer.metadata['model_name']}** · trained "
        f"{scorer.metadata['trained_at'][:10]} · profit-optimal threshold "
        f"**{scorer.threshold:.2f}** · calibrated: "
        f"**{'yes' if scorer.metadata['calibrated'] else 'no'}**"
    )

    tab_score, tab_portfolio, tab_model = st.tabs(
        ["Score a member", "Portfolio radar", "Model report"]
    )

    sample = load_sample()

    # ------------------------------------------------------------------ #
    with tab_score:
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.subheader("Member profile")
            preset = st.selectbox(
                "Start from a real member (test snapshot)",
                ["— manual entry —"] + [f"member #{i}" for i in sample["member_id"].head(30)],
            )
            row = (
                sample[sample["member_id"] == int(preset.split("#")[1])].iloc[0]
                if preset != "— manual entry —"
                else None
            )

            def default(col: str, fallback):
                return type(fallback)(row[col]) if row is not None else fallback

            tenure_days = st.slider("Tenure (days)", 30, 900, default("tenure_days", 180))
            recency_days = st.slider("Days since last visit", 0, 90, int(default("recency_days", 7.0)))
            visits_30d = st.slider("Visits — last 30 days", 0, 40, default("visits_30d", 8))
            visits_90d = st.slider("Visits — last 90 days", 0, 120, default("visits_90d", 26))
            visit_trend_30d = st.slider(
                "Visit trend (last 30d minus prior 30d)", -30, 30, int(default("visit_trend_30d", 0.0))
            )
            active_weeks = st.slider(
                "Active weeks — last 12 (share)", 0.0, 1.0, float(default("active_weeks_ratio_12w", 0.7))
            )
            class_ratio = st.slider(
                "Share of visits that are classes", 0.0, 1.0, float(default("class_ratio_90d", 0.2))
            )
            payment_failures = st.slider(
                "Failed payments — last 90 days", 0, 4, default("payment_failures_90d", 0)
            )
            monthly_fee = st.number_input(
                "Monthly fee", 25.0, 180.0, float(default("monthly_fee", 62.0))
            )
            age = st.slider("Age", 16, 75, default("age", 34))
            plan_type = st.selectbox(
                "Plan", ["monthly", "quarterly", "annual"],
                index=["monthly", "quarterly", "annual"].index(default("plan_type", "monthly")),
            )
            gender = st.selectbox(
                "Gender", ["female", "male", "other"],
                index=["female", "male", "other"].index(default("gender", "female")),
            )
            referral = st.selectbox(
                "Acquisition channel",
                ["instagram", "friend", "walk_in", "corporate", "website"],
                index=["instagram", "friend", "walk_in", "corporate", "website"].index(
                    default("referral_source", "friend")
                ),
            )

        member = pd.DataFrame(
            [
                {
                    "tenure_days": tenure_days,
                    "recency_days": float(recency_days),
                    "visits_7d": int(round(visits_30d / 4)),
                    "visits_30d": visits_30d,
                    "visits_90d": visits_90d,
                    "visits_per_week_90d": visits_90d / (90 / 7),
                    "visit_trend_30d": float(visit_trend_30d),
                    "active_weeks_ratio_12w": active_weeks,
                    "weekly_visits_std_12w": float(
                        default("weekly_visits_std_12w", 1.2) if row is not None else 1.2
                    ),
                    "class_ratio_90d": class_ratio,
                    "peak_hour_ratio_90d": float(
                        default("peak_hour_ratio_90d", 0.4) if row is not None else 0.4
                    ),
                    "payment_failures_90d": payment_failures,
                    "late_payments_90d": int(
                        default("late_payments_90d", 0) if row is not None else 0
                    ),
                    "monthly_fee": monthly_fee,
                    "age": age,
                    "plan_type": plan_type,
                    "gender": gender,
                    "referral_source": referral,
                }
            ]
        )

        with right:
            scored = scorer.score_frame(member)
            probability = float(scored["churn_probability"].iloc[0])
            tier = scored["risk_tier"].iloc[0]

            st.plotly_chart(
                probability_gauge(probability, scorer.threshold),
                use_container_width=True,
            )
            badge = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}[tier]
            action = (
                "**Action:** add to this month's retention campaign — the expected "
                "value of outreach is positive at this score."
                if probability >= scorer.threshold
                else "**Action:** no outreach — contact cost exceeds expected saved revenue."
            )
            st.markdown(f"### {badge}")
            st.markdown(action)

            contributions = scorer.explain_frame(member).iloc[0]
            st.plotly_chart(
                member_waterfall_figure(
                    contributions, member.iloc[0][FEATURE_COLUMNS], probability
                ),
                use_container_width=True,
            )

    # ------------------------------------------------------------------ #
    with tab_portfolio:
        scored_all = scorer.score_frame(sample[FEATURE_COLUMNS], validate=False).assign(
            member_id=sample["member_id"].values
        )
        business = load_json("business_impact.json")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Members scored", f"{len(scored_all):,}")
        c2.metric("Flagged for outreach", int(scored_all["flagged_for_outreach"].sum()))
        c3.metric(
            "Mean churn probability", f"{scored_all['churn_probability'].mean():.1%}"
        )
        if business:
            c4.metric(
                "Top-decile campaign profit",
                f"${business['campaign_top_decile']['expected_profit']:,.0f}",
                help="Expected profit of contacting the top 10% riskiest members "
                "(see assets/business_impact.json for assumptions).",
            )

        tier_counts = (
            scored_all["risk_tier"].value_counts().reindex(["low", "medium", "high"]).fillna(0)
        )
        fig = go.Figure(
            go.Bar(
                x=tier_counts.index.str.title(),
                y=tier_counts.values,
                marker_color=[TIER_COLORS[t] for t in tier_counts.index],
                hovertemplate="%{x}: %{y} members<extra></extra>",
            )
        )
        st.plotly_chart(
            apply_theme(fig, title="Risk tier mix (sample of test snapshot)", height=340),
            use_container_width=True,
        )

        st.subheader("Ranked outreach list")
        top = scored_all.sort_values("churn_probability", ascending=False).head(15)
        st.dataframe(
            top[
                ["member_id", "churn_probability", "risk_tier", "recency_days",
                 "visits_30d", "visit_trend_30d", "payment_failures_90d",
                 "plan_type", "monthly_fee"]
            ].style.format({"churn_probability": "{:.1%}", "monthly_fee": "${:.0f}"}),
            use_container_width=True,
        )

    # ------------------------------------------------------------------ #
    with tab_model:
        performance = load_json("model_performance.json")
        if performance:
            metrics = performance["metrics"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
            c3.metric("F1 @ threshold", f"{metrics['f1']:.3f}")
            c4.metric("Recall @ threshold", f"{metrics['recall']:.3f}")
            c5.metric("Lift @ top 10%", f"{metrics['lift_at_10pct']:.1f}×")
            st.caption(
                f"All metrics on the out-of-time test snapshot "
                f"({performance['test_snapshot']}) — a month the model never saw."
            )

        assets_dir = Path(config.paths.assets_dir)
        plot_names = [
            "pr_curve", "roc_curve", "calibration_curve", "confusion_matrix",
            "gains_lift", "profit_curve", "score_distribution", "cohort_churn",
            "shap_importance", "shap_beeswarm",
        ]
        available = [n for n in plot_names if (assets_dir / f"{n}.html").exists()]
        choice = st.selectbox("Interactive evaluation plots", available)
        if choice:
            components.html(
                (assets_dir / f"{choice}.html").read_text(encoding="utf-8"),
                height=620, scrolling=False,
            )


if __name__ == "__main__":
    main()
