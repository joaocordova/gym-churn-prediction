"""Model definitions: preprocessing, candidate estimators, calibration wrapper.

Two candidate families are trained and compared on the same grouped CV:

* ``logistic`` — scaled + one-hot logistic regression. The interpretable
  baseline every gradient-boosted model must beat to justify its complexity.
* ``lightgbm`` — gradient-boosted trees with a randomized hyperparameter
  search. Handles non-linearities (e.g. recency × tenure interactions).

The exported artifact is a :class:`CalibratedModel`: the winning pipeline plus
an optional Platt (sigmoid) calibration layer fitted on a grouped validation
split — kept only when it improves the Brier score, because a churn
probability is a business input (expected-value math), not just a ranking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gym_churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    # Keep feature names flowing through the pipeline (SHAP + LightGBM both
    # consume them, and it silences fit/predict feature-name mismatches).
    return transformer.set_output(transform="pandas")


def candidate_specs(random_seed: int) -> dict[str, dict]:
    """Name -> {pipeline, param_distributions} for the model bake-off."""
    return {
        "logistic": {
            "pipeline": Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=random_seed,
                        ),
                    ),
                ]
            ),
            "param_distributions": {
                "model__C": loguniform(1e-3, 10.0),
            },
        },
        "lightgbm": {
            "pipeline": Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=False)),
                    (
                        "model",
                        LGBMClassifier(
                            objective="binary",
                            random_state=random_seed,
                            verbose=-1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            "param_distributions": {
                "model__n_estimators": randint(150, 600),
                "model__learning_rate": loguniform(0.01, 0.15),
                "model__num_leaves": randint(15, 64),
                "model__min_child_samples": randint(20, 120),
                "model__subsample": uniform(0.7, 0.3),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__reg_lambda": loguniform(1e-3, 10.0),
            },
        },
    }


class SigmoidCalibrator:
    """Platt scaling: logistic regression on the logit of the raw score."""

    def __init__(self):
        self._lr = LogisticRegression(max_iter=1000)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        self._lr.fit(self._logit(raw_prob), y)
        return self

    def transform(self, raw_prob: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._logit(raw_prob))[:, 1]


class CalibratedModel:
    """The deployable artifact: winning pipeline + optional calibration layer."""

    def __init__(self, pipeline: Pipeline, calibrator: SigmoidCalibrator | None = None):
        self.pipeline = pipeline
        self.calibrator = calibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.pipeline.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            raw = self.calibrator.transform(raw)
        return raw

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def estimator_name(self) -> str:
        return type(self.pipeline.named_steps["model"]).__name__

    def transformed_feature_names(self) -> list[str]:
        return list(
            self.pipeline.named_steps["preprocess"].get_feature_names_out()
        )
