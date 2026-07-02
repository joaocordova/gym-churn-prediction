"""Inference service: the single entry point every consumer scores through.

``ChurnScorer`` wraps the persisted model artifact + metadata and gives the
Streamlit app, batch jobs and tests one identical code path:

* input rows are validated against the :class:`~gym_churn.schemas.ScoringRequest`
  contract (bad payloads fail loudly, never silently mis-score),
* probabilities come from the calibrated model,
* the deployed profit-optimal threshold and risk tiers ride along,
* per-member SHAP explanations are available on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gym_churn.explain import ShapExplainer
from gym_churn.features import FEATURE_COLUMNS
from gym_churn.logging_utils import get_logger
from gym_churn.models import CalibratedModel
from gym_churn.schemas import ScoringRequest

log = get_logger(module="predict")


class ChurnScorer:
    def __init__(self, model: CalibratedModel, metadata: dict):
        self.model = model
        self.metadata = metadata
        self.threshold = float(metadata["decision_threshold"])
        self._explainer: ShapExplainer | None = None

    @classmethod
    def load(cls, models_dir: str | Path) -> "ChurnScorer":
        models_dir = Path(models_dir)
        model_path = models_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found — train a model first "
                "(python -m gym_churn.cli train)"
            )
        model = joblib.load(model_path)
        metadata = json.loads((models_dir / "metadata.json").read_text())
        return cls(model, metadata)

    # ------------------------------------------------------------------ #

    def validate(self, frame: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Scoring input is missing features: {missing}")
        for record in frame[FEATURE_COLUMNS].to_dict(orient="records"):
            ScoringRequest(**record)  # raises pydantic.ValidationError on bad rows

    def risk_tier(self, probability: np.ndarray) -> np.ndarray:
        tiers = np.where(
            probability >= self.threshold,
            "high",
            np.where(probability >= self.threshold / 2, "medium", "low"),
        )
        return tiers

    def score_frame(self, frame: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Score members; returns the input plus probability / flag / tier."""
        if validate:
            self.validate(frame)
        out = frame.copy()
        prob = self.model.predict_proba(frame[FEATURE_COLUMNS])
        out["churn_probability"] = prob
        out["flagged_for_outreach"] = (prob >= self.threshold).astype(int)
        out["risk_tier"] = self.risk_tier(prob)
        return out

    # ------------------------------------------------------------------ #

    def _get_explainer(self, background: pd.DataFrame) -> ShapExplainer:
        if self._explainer is None:
            self._explainer = ShapExplainer(self.model, background)
        return self._explainer

    def explain_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-member SHAP contributions on the original business features."""
        explainer = self._get_explainer(frame)
        contributions, _ = explainer.aggregated(frame)
        return contributions
