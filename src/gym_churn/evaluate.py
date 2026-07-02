"""Model evaluation: honest metrics on the temporal test set + the full
gallery of interactive Plotly reports saved under ``assets/``.

Every figure is generated from the *out-of-time* test snapshot — the model
has never seen this month. Charts produced:

* ``roc_curve`` / ``pr_curve`` — discrimination (PR is the headline for a
  ~5% positive class; ROC flatters imbalanced problems).
* ``calibration_curve`` — do predicted probabilities match reality?
* ``confusion_matrix`` — at the profit-optimal threshold, not 0.5.
* ``gains_lift`` — cumulative gains + lift by decile: "call the top 10%,
  capture X% of churners".
* ``profit_curve`` — expected campaign profit vs threshold (business view).
* ``score_distribution`` — churner vs non-churner score separation.
* ``cohort_churn`` — churn rate by tenure cohort × plan type heatmap.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from gym_churn.business import build_business_report, profit_curve
from gym_churn.config import AppConfig
from gym_churn.features import TARGET, load_processed, validate_feature_frame
from gym_churn.logging_utils import get_logger
from gym_churn.plotting import (
    GRID,
    INK_SECONDARY,
    MUTED,
    SEQUENTIAL_BLUES,
    SERIES,
    save_figure,
    themed_figure,
)

log = get_logger(module="evaluate")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    order = np.argsort(-y_prob)
    top10 = order[: max(1, int(0.10 * len(y_prob)))]
    base_rate = float(np.mean(y_true))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision_at_10pct": float(np.mean(np.asarray(y_true)[top10])),
        "lift_at_10pct": float(np.mean(np.asarray(y_true)[top10]) / base_rate)
        if base_rate > 0
        else 0.0,
        "base_churn_rate": base_rate,
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def roc_figure(y_true, y_prob) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig = themed_figure(
        title=f"ROC curve — AUC {auc:.3f} (out-of-time test)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, mode="lines", name="Model",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Chance",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    return fig


def pr_figure(y_true, y_prob, threshold: float) -> go.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    base = float(np.mean(y_true))
    fig = themed_figure(
        title=f"Precision–recall tradeoff — PR-AUC {ap:.3f}",
        xaxis_title="Recall (churners caught)",
        yaxis_title="Precision (flags that are right)",
    )
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision, mode="lines", name="Model",
            line=dict(color=SERIES[0], width=2),
            hovertemplate="Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=base, line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text=f"Base rate {base:.1%}", annotation_font_color=MUTED,
    )
    # Mark the deployed operating point.
    idx = int(np.argmin(np.abs(thresholds - threshold))) if len(thresholds) else 0
    fig.add_trace(
        go.Scatter(
            x=[recall[idx]], y=[precision[idx]], mode="markers+text",
            name="Deployed threshold", text=[f"t = {threshold:.2f}"],
            textposition="top right", textfont=dict(color=INK_SECONDARY),
            marker=dict(color=SERIES[5], size=11, line=dict(color="#fcfcfb", width=2)),
            hovertemplate=(
                f"Threshold {threshold:.3f}<br>Recall %{{x:.3f}}"
                "<br>Precision %{y:.3f}<extra></extra>"
            ),
        )
    )
    return fig


def calibration_figure(y_true, y_prob, n_bins: int = 10) -> go.Figure:
    bins = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    frame = pd.DataFrame({"bin": idx, "prob": y_prob, "y": y_true})
    grouped = frame.groupby("bin").agg(mean_prob=("prob", "mean"), rate=("y", "mean"), n=("y", "size"))
    fig = themed_figure(
        title="Calibration — predicted probability vs observed churn rate",
        xaxis_title="Mean predicted probability (per decile bin)",
        yaxis_title="Observed churn rate",
    )
    fig.add_trace(
        go.Scatter(
            x=[0, grouped["mean_prob"].max() * 1.05],
            y=[0, grouped["mean_prob"].max() * 1.05],
            mode="lines", name="Perfect calibration",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grouped["mean_prob"], y=grouped["rate"], mode="lines+markers",
            name="Model", line=dict(color=SERIES[0], width=2),
            marker=dict(size=9),
            customdata=grouped["n"].to_numpy(),
            hovertemplate=(
                "Predicted %{x:.3f}<br>Observed %{y:.3f}"
                "<br>%{customdata} members<extra></extra>"
            ),
        )
    )
    return fig


def confusion_figure(y_true, y_prob, threshold: float) -> go.Figure:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    labels = ["Stays", "Churns"]
    fig = themed_figure(
        title=f"Confusion matrix at profit-optimal threshold ({threshold:.2f})",
        xaxis_title="Predicted", yaxis_title="Actual",
    )
    fig.add_trace(
        go.Heatmap(
            z=matrix, x=labels, y=labels,
            colorscale=[[0, SEQUENTIAL_BLUES[0]], [1, SEQUENTIAL_BLUES[-1]]],
            showscale=False,
            text=[[f"{v:,}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=18),
            hovertemplate="Actual %{y} / Predicted %{x}: %{z:,}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def gains_lift_figure(y_true, y_prob) -> go.Figure:
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_prob))
    sorted_y = y_true[order]
    n = len(sorted_y)
    fractions = np.arange(1, 21) / 20.0  # 5% steps
    gains, lifts = [], []
    total_pos = sorted_y.sum()
    for f in fractions:
        k = max(1, int(round(f * n)))
        captured = sorted_y[:k].sum()
        gains.append(captured / max(total_pos, 1))
        lifts.append((captured / k) / max(y_true.mean(), 1e-9))

    fig = themed_figure(
        title="Cumulative gains — churners captured by contacting the top X%",
        xaxis_title="Share of members contacted (ranked by model score)",
        yaxis_title="Share of actual churners captured",
    )
    fig.add_trace(
        go.Scatter(
            x=fractions, y=gains, mode="lines+markers", name="Model",
            line=dict(color=SERIES[0], width=2), marker=dict(size=7),
            customdata=np.array(lifts).reshape(-1, 1),
            hovertemplate=(
                "Contact top %{x:.0%}<br>Capture %{y:.1%} of churners"
                "<br>Lift %{customdata[0]:.2f}×<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random targeting",
            line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip",
        )
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


def profit_figure(y_true, y_prob, config: AppConfig) -> go.Figure:
    curve = profit_curve(np.asarray(y_true), np.asarray(y_prob), config.business)
    symbol = config.business.currency_symbol
    fig = themed_figure(
        title="Expected campaign profit vs decision threshold",
        xaxis_title="Decision threshold (flag members above this churn probability)",
        yaxis_title=f"Expected profit ({config.business.currency})",
    )
    fig.add_trace(
        go.Scatter(
            x=curve["threshold"], y=curve["expected_profit"], mode="lines",
            name="Expected profit", line=dict(color=SERIES[1], width=2),
            customdata=curve[["n_flagged", "precision"]].to_numpy(),
            hovertemplate=(
                "Threshold %{x:.3f}<br>Profit " + symbol + "%{y:,.0f}"
                "<br>%{customdata[0]} members flagged"
                "<br>Precision %{customdata[1]:.1%}<extra></extra>"
            ),
        )
    )
    best = curve.loc[curve["expected_profit"].idxmax()]
    fig.add_trace(
        go.Scatter(
            x=[best["threshold"]], y=[best["expected_profit"]],
            mode="markers+text", name="Optimal threshold",
            text=[f"{symbol}{best['expected_profit']:,.0f}"],
            textposition="top center", textfont=dict(color=INK_SECONDARY),
            marker=dict(color=SERIES[5], size=11, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="Optimal threshold %{x:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    return fig


def score_distribution_figure(y_true, y_prob) -> go.Figure:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig = themed_figure(
        title="Score separation — churners vs retained members",
        xaxis_title="Predicted churn probability",
        yaxis_title="Share of group",
        barmode="overlay",
    )
    for label, mask, color in (
        ("Retained", y_true == 0, SERIES[0]),
        ("Churned", y_true == 1, SERIES[5]),
    ):
        fig.add_trace(
            go.Histogram(
                x=y_prob[mask], name=label, histnorm="probability",
                nbinsx=40, opacity=0.62, marker_color=color,
                hovertemplate=label + ": %{y:.1%} at score %{x:.2f}<extra></extra>",
            )
        )
    return fig


def cohort_figure(full: pd.DataFrame) -> go.Figure:
    frame = full.copy()
    tenure_bins = [0, 60, 120, 180, 270, 365, 550, 10_000]
    tenure_labels = ["0–2 mo", "2–4 mo", "4–6 mo", "6–9 mo", "9–12 mo", "12–18 mo", "18+ mo"]
    frame["tenure_cohort"] = pd.cut(
        frame["tenure_days"], bins=tenure_bins, labels=tenure_labels, right=False
    )
    pivot = (
        frame.pivot_table(
            index="plan_type", columns="tenure_cohort",
            values=TARGET, aggfunc="mean", observed=True,
        )
        .reindex(index=["monthly", "quarterly", "annual"])
        .reindex(columns=tenure_labels)
    )
    counts = frame.pivot_table(
        index="plan_type", columns="tenure_cohort",
        values=TARGET, aggfunc="size", observed=True,
    ).reindex(index=pivot.index, columns=pivot.columns)
    fig = themed_figure(
        title="Cohort analysis — 30-day churn rate by tenure × plan",
        xaxis_title="Tenure cohort", yaxis_title="Plan type",
    )
    fig.add_trace(
        go.Heatmap(
            z=pivot.values, x=tenure_labels, y=pivot.index.tolist(),
            colorscale=[[0, SEQUENTIAL_BLUES[0]], [1, SEQUENTIAL_BLUES[-1]]],
            colorbar=dict(title="Churn rate", tickformat=".0%"),
            text=[[f"{v:.1%}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            customdata=counts.values,
            hovertemplate=(
                "%{y} plan, tenure %{x}<br>Churn rate %{z:.1%}"
                "<br>%{customdata:,} member-snapshots<extra></extra>"
            ),
        )
    )
    return fig


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


def run_evaluation(config: AppConfig) -> dict[str, float]:
    """Score the temporal test set, persist metrics, figures and the
    business-impact report."""
    models_dir = Path(config.paths.models_dir)
    model = joblib.load(models_dir / "model.joblib")
    metadata = json.loads((models_dir / "metadata.json").read_text())
    threshold = metadata["decision_threshold"]

    data = load_processed(config)
    test, train = data["test"], data["train"]
    validate_feature_frame(test)

    y_true = test[TARGET].to_numpy()
    y_prob = model.predict_proba(test)

    metrics = compute_metrics(y_true, y_prob, threshold)
    log.info("Test metrics: {m}", m={k: round(v, 4) for k, v in metrics.items()})

    assets, img = config.paths.assets_dir, config.paths.img_dir
    figures = {
        "roc_curve": roc_figure(y_true, y_prob),
        "pr_curve": pr_figure(y_true, y_prob, threshold),
        "calibration_curve": calibration_figure(y_true, y_prob),
        "confusion_matrix": confusion_figure(y_true, y_prob, threshold),
        "gains_lift": gains_lift_figure(y_true, y_prob),
        "profit_curve": profit_figure(y_true, y_prob, config),
        "score_distribution": score_distribution_figure(y_true, y_prob),
        "cohort_churn": cohort_figure(pd.concat([train, test], ignore_index=True)),
    }
    for name, fig in figures.items():
        save_figure(fig, name, assets, img)

    report = {
        "model": metadata["model_name"],
        "test_snapshot": str(config.snapshots.test_snapshot),
        "metrics": metrics,
    }
    Path(assets, "model_performance.json").write_text(json.dumps(report, indent=2))

    build_business_report(test, y_prob, threshold, config)
    return metrics
