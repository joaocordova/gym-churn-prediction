"""Training pipeline: grouped hyperparameter search, model bake-off,
probability calibration and profit-optimal threshold selection.

Design decisions that matter:

* **Grouped everything.** The same member appears in multiple monthly
  snapshots, so every split (CV folds, validation holdout) is grouped by
  ``member_id``. A random row split would leak member identity across folds
  and inflate every metric.
* **PR-AUC as the search objective.** With a ~5% positive class, ROC-AUC
  saturates and hides real differences; average precision does not.
* **Calibration is earned, not assumed.** Platt scaling is fitted on one half
  of the validation split and kept only if it improves the Brier score on the
  other half — probabilities feed expected-value math downstream.
* **The decision threshold is a business decision.** It is chosen to maximise
  expected campaign profit (save-rate × retained revenue − outreach cost),
  not a symmetric 0.5.

Every run is recorded by the file-based experiment tracker under
``experiments/`` (params, metrics, artifacts, environment).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV

from gym_churn import __version__
from gym_churn.business import optimal_threshold
from gym_churn.config import AppConfig
from gym_churn.evaluate import compute_metrics
from gym_churn.features import (
    FEATURE_COLUMNS,
    TARGET,
    load_processed,
    validate_feature_frame,
)
from gym_churn.logging_utils import get_logger
from gym_churn.models import CalibratedModel, SigmoidCalibrator, candidate_specs
from gym_churn.tracking import ExperimentTracker

log = get_logger(module="train")


def run_training(config: AppConfig) -> dict:
    seed = config.project.random_seed
    train = load_processed(config)["train"]
    validate_feature_frame(train)

    X = train[FEATURE_COLUMNS]
    y = train[TARGET].to_numpy()
    groups = train["member_id"].to_numpy()

    # Grouped fit/validation split — the validation members never appear in
    # any training fold.
    fit_idx, val_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=config.training.validation_fraction,
            random_state=seed,
        ).split(X, y, groups)
    )
    X_fit, y_fit, g_fit = X.iloc[fit_idx], y[fit_idx], groups[fit_idx]
    X_val, y_val, g_val = X.iloc[val_idx], y[val_idx], groups[val_idx]

    tracker = ExperimentTracker(config.paths.experiments_dir)
    with tracker.start_run(
        "train", tags={"stage": "training", "package_version": __version__}
    ) as run:
        run.log_params(
            {
                "n_train_rows": len(fit_idx),
                "n_val_rows": len(val_idx),
                "train_churn_rate": round(float(y_fit.mean()), 4),
                "cv_folds": config.training.cv_folds,
                "n_search_iter": config.training.n_search_iter,
                "scoring": config.training.scoring,
                "random_seed": seed,
                "features": FEATURE_COLUMNS,
            }
        )

        # ------------------------- model bake-off ------------------------- #
        results: dict[str, dict] = {}
        for name, spec in candidate_specs(seed).items():
            n_iter = config.training.n_search_iter if name == "lightgbm" else 10
            search = RandomizedSearchCV(
                spec["pipeline"],
                spec["param_distributions"],
                n_iter=n_iter,
                scoring=config.training.scoring,
                cv=GroupKFold(n_splits=config.training.cv_folds),
                random_state=seed,
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_fit, y_fit, groups=g_fit)
            val_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
            val_ap = float(average_precision_score(y_val, val_prob))
            results[name] = {
                "estimator": search.best_estimator_,
                "val_pr_auc": val_ap,
                "cv_pr_auc": float(search.best_score_),
                "best_params": {
                    k.removeprefix("model__"): v for k, v in search.best_params_.items()
                },
                "val_prob": val_prob,
            }
            run.log_metrics(
                {f"{name}_cv_pr_auc": search.best_score_, f"{name}_val_pr_auc": val_ap}
            )
            log.info(
                "{name}: CV PR-AUC {cv:.4f} | validation PR-AUC {val:.4f}",
                name=name, cv=search.best_score_, val=val_ap,
            )

        best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
        best = results[best_name]
        run.log_params({"selected_model": best_name, "best_params": best["best_params"]})

        # ------------------------- calibration --------------------------- #
        # Fit Platt scaling on one grouped half of the validation split and
        # keep it only if it improves Brier on the *other* half.
        cal_idx, assess_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed).split(
                X_val, y_val, g_val
            )
        )
        # Degenerate halves (a class missing — possible on very small datasets)
        # fall back to the full validation split and skip calibration.
        halves_ok = (
            len(np.unique(y_val[cal_idx])) == 2
            and len(np.unique(y_val[assess_idx])) == 2
        )
        if not halves_ok:
            assess_idx = np.arange(len(y_val))
            log.warning(
                "Validation halves are single-class; skipping calibration and "
                "using the full validation split for thresholding"
            )
        calibrator = None
        calibrated = False
        raw_assess = best["val_prob"][assess_idx]
        if config.training.calibration == "auto" and halves_ok:
            candidate = SigmoidCalibrator().fit(best["val_prob"][cal_idx], y_val[cal_idx])
            brier_raw = brier_score_loss(y_val[assess_idx], raw_assess)
            brier_cal = brier_score_loss(
                y_val[assess_idx], candidate.transform(raw_assess)
            )
            if brier_cal < brier_raw:
                calibrator = candidate
                calibrated = True
            run.log_metrics({"brier_raw": brier_raw, "brier_calibrated": brier_cal})
            log.info(
                "Calibration {verdict} (Brier {raw:.5f} -> {cal:.5f})",
                verdict="kept" if calibrated else "rejected",
                raw=brier_raw, cal=brier_cal,
            )

        model = CalibratedModel(pipeline=best["estimator"], calibrator=calibrator)

        # -------------------- profit-optimal threshold -------------------- #
        assess_prob = model.predict_proba(X_val.iloc[assess_idx])
        threshold, operating_point = optimal_threshold(
            y_val[assess_idx], assess_prob, config.business
        )
        val_metrics = compute_metrics(y_val[assess_idx], assess_prob, threshold)
        run.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        run.log_dict("operating_point", operating_point)

        # ----------------------------- persist ---------------------------- #
        models_dir = Path(config.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.joblib"
        joblib.dump(model, model_path)

        metadata = {
            "model_name": best_name,
            "estimator": model.estimator_name,
            "package_version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run.run_id,
            "decision_threshold": threshold,
            "calibrated": calibrated,
            "best_params": best["best_params"],
            "feature_columns": FEATURE_COLUMNS,
            "validation_metrics": val_metrics,
            "candidates": {
                name: {"cv_pr_auc": r["cv_pr_auc"], "val_pr_auc": r["val_pr_auc"]}
                for name, r in results.items()
            },
        }
        (models_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )
        run.log_artifact(model_path)
        run.log_artifact(models_dir / "metadata.json")
        log.info(
            "Selected {name} (threshold {t:.3f}) -> {path}",
            name=best_name, t=threshold, path=str(model_path),
        )

    return metadata
