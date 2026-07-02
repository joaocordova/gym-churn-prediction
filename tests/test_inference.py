"""End-to-end integration: the trained artifact must score, explain and
report exactly as the production entry points do."""

import json

import numpy as np
import pandas as pd
import pytest

from gym_churn.features import FEATURE_COLUMNS, TARGET, load_processed
from gym_churn.predict import ChurnScorer


@pytest.fixture(scope="module")
def scorer(trained_pipeline):
    return ChurnScorer.load(trained_pipeline.paths.models_dir)


@pytest.fixture(scope="module")
def test_frame(trained_pipeline):
    return load_processed(trained_pipeline)["test"]


def test_metadata_is_complete(scorer):
    meta = scorer.metadata
    assert meta["model_name"] in {"logistic", "lightgbm"}
    assert 0.0 < meta["decision_threshold"] < 1.0
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert "validation_metrics" in meta
    assert set(meta["candidates"]) == {"logistic", "lightgbm"}


def test_scores_are_valid_probabilities(scorer, test_frame):
    scored = scorer.score_frame(test_frame.head(300), validate=False)
    prob = scored["churn_probability"]
    assert prob.between(0, 1).all()
    assert (scored["flagged_for_outreach"] == (prob >= scorer.threshold)).all()
    assert set(scored["risk_tier"].unique()) <= {"low", "medium", "high"}


def test_model_beats_random_ranking(scorer, test_frame):
    """Sanity floor: on out-of-time data the model must rank churners above
    non-churners far better than chance."""
    from sklearn.metrics import roc_auc_score

    scored = scorer.score_frame(test_frame, validate=False)
    auc = roc_auc_score(test_frame[TARGET], scored["churn_probability"])
    assert auc > 0.70


def test_validation_rejects_missing_feature(scorer, test_frame):
    broken = test_frame.head(5).drop(columns=["recency_days"])
    with pytest.raises(ValueError, match="missing features"):
        scorer.score_frame(broken)


def test_validation_rejects_out_of_contract_values(scorer, test_frame):
    bad = test_frame.head(3).copy()
    bad.loc[bad.index[0], "age"] = 300
    with pytest.raises(Exception):
        scorer.score_frame(bad)


def test_explanations_align_with_features(scorer, test_frame):
    rows = test_frame.head(20)
    contributions = scorer.explain_frame(rows)
    assert list(contributions.columns) == FEATURE_COLUMNS
    assert len(contributions) == 20
    assert np.isfinite(contributions.to_numpy()).all()


def test_experiment_was_tracked(trained_pipeline):
    index = trained_pipeline.paths.experiments_dir / "index.jsonl"
    assert index.exists()
    runs = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    assert any(r["name"] == "train" and r["status"] == "completed" for r in runs)
    last = runs[-1]
    assert "lightgbm_val_pr_auc" in last["metrics"]
    assert "selected_model" in last["params"]


def test_evaluation_produces_assets_and_metrics(trained_pipeline):
    from gym_churn.evaluate import run_evaluation

    metrics = run_evaluation(trained_pipeline)
    assert metrics["roc_auc"] > 0.70
    assert metrics["pr_auc"] > metrics["base_churn_rate"]

    assets = trained_pipeline.paths.assets_dir
    for name in (
        "roc_curve", "pr_curve", "calibration_curve", "confusion_matrix",
        "gains_lift", "profit_curve", "score_distribution", "cohort_churn",
    ):
        assert (assets / f"{name}.html").exists(), name
    assert (assets / "model_performance.json").exists()

    business = json.loads((assets / "business_impact.json").read_text())
    assert business["campaign_top_decile"]["n_flagged"] > 0
    assert business["model_uplift_vs_random"] > 0
