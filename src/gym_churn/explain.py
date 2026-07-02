"""SHAP explainability — global drivers and per-member explanations.

The explainer operates on the *transformed* feature space of the winning
pipeline (post one-hot), then aggregates one-hot contributions back to their
original business feature (``plan_type_monthly`` + ``plan_type_annual`` + …
→ ``plan_type``) so that every chart speaks the language of the feature
dictionary, not of the encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from lightgbm import LGBMClassifier

from gym_churn.config import AppConfig
from gym_churn.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, load_processed
from gym_churn.logging_utils import get_logger
from gym_churn.models import CalibratedModel
from gym_churn.plotting import (
    NEGATIVE,
    POSITIVE,
    SEQUENTIAL_BLUES,
    SERIES,
    save_figure,
    themed_figure,
)

log = get_logger(module="explain")

MAX_EXPLAIN_ROWS = 2_500  # SHAP sample size for global plots


class ShapExplainer:
    """SHAP wrapper that understands the project's pipeline artifact."""

    def __init__(self, model: CalibratedModel, background: pd.DataFrame):
        self.model = model
        self.preprocess = model.pipeline.named_steps["preprocess"]
        estimator = model.pipeline.named_steps["model"]
        self.transformed_names = model.transformed_feature_names()

        background_t = self.preprocess.transform(background[FEATURE_COLUMNS])
        if isinstance(estimator, LGBMClassifier):
            self._explainer = shap.TreeExplainer(estimator)
        else:
            self._explainer = shap.LinearExplainer(estimator, background_t)

    def shap_matrix(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Per-row SHAP values in the transformed space + expected value."""
        Xt = self.preprocess.transform(X[FEATURE_COLUMNS])
        explanation = self._explainer(Xt)
        values = explanation.values
        base = explanation.base_values
        if values.ndim == 3:  # (rows, features, classes) — take positive class
            values = values[:, :, 1]
            base = base[:, 1]
        base_value = float(np.atleast_1d(base)[0])
        return values, base_value

    def aggregated(self, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """SHAP values summed back onto the original business features."""
        values, base_value = self.shap_matrix(X)
        raw = pd.DataFrame(values, columns=self.transformed_names, index=X.index)
        out = pd.DataFrame(index=X.index)
        for feature in FEATURE_COLUMNS:
            if feature in CATEGORICAL_FEATURES:
                cols = [c for c in raw.columns if c.startswith(f"{feature}_")]
                out[feature] = raw[cols].sum(axis=1)
            else:
                out[feature] = raw[feature]
        return out, base_value


def global_importance_figure(shap_frame: pd.DataFrame, top_n: int = 15) -> go.Figure:
    importance = shap_frame.abs().mean().sort_values(ascending=True).tail(top_n)
    fig = themed_figure(
        title="Global churn drivers — mean |SHAP| (impact on model output)",
        xaxis_title="Mean absolute SHAP value (log-odds)",
        yaxis_title=None,
        height=520,
    )
    fig.add_trace(
        go.Bar(
            x=importance.values, y=importance.index, orientation="h",
            marker=dict(color=SERIES[0]),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    return fig


def beeswarm_figure(
    shap_frame: pd.DataFrame, X: pd.DataFrame, top_n: int = 10, seed: int = 42
) -> go.Figure:
    """Interactive beeswarm: SHAP value vs (color-coded) feature value."""
    rng = np.random.default_rng(seed)
    top = shap_frame.abs().mean().sort_values(ascending=False).head(top_n).index
    fig = themed_figure(
        title="How feature values push churn risk (SHAP beeswarm)",
        xaxis_title="SHAP value — ← lowers risk | raises risk →",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(top))),
            ticktext=list(top),
            autorange="reversed",
        ),
        height=580,
        showlegend=False,
    )
    for row_pos, feature in enumerate(top):
        values = X[feature]
        if values.dtype == object:
            codes = pd.Categorical(values).codes.astype(float)
        else:
            codes = values.astype(float)
        span = codes.max() - codes.min()
        norm = (codes - codes.min()) / span if span > 0 else np.full(len(codes), 0.5)
        jitter = rng.uniform(-0.28, 0.28, len(values))
        fig.add_trace(
            go.Scattergl(
                x=shap_frame[feature],
                y=row_pos + jitter,
                mode="markers",
                marker=dict(
                    size=5,
                    color=norm,
                    colorscale=[[0, SEQUENTIAL_BLUES[1]], [1, SEQUENTIAL_BLUES[-1]]],
                    opacity=0.75,
                ),
                customdata=values.astype(str),
                hovertemplate=(
                    feature + " = %{customdata}<br>SHAP %{x:.3f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    return fig


def member_waterfall_figure(
    contributions: pd.Series,
    feature_values: pd.Series,
    probability: float,
    top_n: int = 9,
) -> go.Figure:
    """Per-member explanation: top signed SHAP contributions."""
    top = contributions.reindex(
        contributions.abs().sort_values(ascending=False).head(top_n).index
    ).sort_values()
    labels = [
        f"{feature} = {feature_values[feature]:.1f}"
        if isinstance(feature_values[feature], (int, float, np.floating))
        else f"{feature} = {feature_values[feature]}"
        for feature in top.index
    ]
    fig = themed_figure(
        title=f"Why this member scores {probability:.0%} churn risk",
        xaxis_title="Contribution to churn risk (SHAP, log-odds)",
        height=440,
        showlegend=False,
    )
    fig.add_trace(
        go.Bar(
            x=top.values, y=labels, orientation="h",
            marker=dict(color=[POSITIVE if v > 0 else NEGATIVE for v in top.values]),
            hovertemplate="%{y}<br>SHAP %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#c3c2b7", width=1))
    return fig


def run_explain(config: AppConfig) -> None:
    """Generate and persist the global explainability assets."""
    models_dir = Path(config.paths.models_dir)
    model: CalibratedModel = joblib.load(models_dir / "model.joblib")

    data = load_processed(config)
    test = data["test"]
    sample = test.sample(
        n=min(MAX_EXPLAIN_ROWS, len(test)), random_state=config.project.random_seed
    ).reset_index(drop=True)

    explainer = ShapExplainer(model, background=sample)
    shap_frame, base_value = explainer.aggregated(sample)

    assets, img = config.paths.assets_dir, config.paths.img_dir
    save_figure(global_importance_figure(shap_frame), "shap_importance", assets, img)
    save_figure(
        beeswarm_figure(shap_frame, sample, seed=config.project.random_seed),
        "shap_beeswarm", assets, img,
    )

    summary = {
        "base_value_logodds": base_value,
        "n_rows_explained": len(sample),
        "mean_abs_shap": shap_frame.abs().mean().sort_values(ascending=False).round(5).to_dict(),
    }
    Path(assets, "shap_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Explainability assets written to {dir}", dir=str(assets))
