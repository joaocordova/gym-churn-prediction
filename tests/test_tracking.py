"""The experiment tracker must persist everything it is told, atomically."""

import json

from gym_churn.tracking import ExperimentTracker


def test_run_persists_params_metrics_and_artifacts(tmp_path):
    tracker = ExperimentTracker(tmp_path / "experiments")
    artifact = tmp_path / "model.txt"
    artifact.write_text("weights")

    with tracker.start_run("unit", tags={"suite": "pytest"}) as run:
        run.log_params({"lr": 0.05, "n_estimators": 300})
        run.log_metrics({"pr_auc": 0.41})
        run.log_dict("extra", {"note": "hello"})
        run.log_artifact(artifact)

    params = json.loads((run.dir / "params.json").read_text())
    metrics = json.loads((run.dir / "metrics.json").read_text())
    meta = json.loads((run.dir / "meta.json").read_text())

    assert params == {"lr": 0.05, "n_estimators": 300}
    assert metrics == {"pr_auc": 0.41}
    assert meta["status"] == "completed"
    assert meta["tags"] == {"suite": "pytest"}
    assert (run.artifacts_dir / "model.txt").read_text() == "weights"
    assert (run.artifacts_dir / "extra.json").exists()


def test_index_accumulates_history(tmp_path):
    tracker = ExperimentTracker(tmp_path / "experiments")
    with tracker.start_run("first") as run:
        run.log_metrics({"m": 1.0})
    with tracker.start_run("second") as run:
        run.log_metrics({"m": 2.0})

    history = tracker.history()
    assert [h["name"] for h in history] == ["first", "second"]
    assert history[1]["metrics"]["m"] == 2.0


def test_exception_marks_run_failed(tmp_path):
    tracker = ExperimentTracker(tmp_path / "experiments")
    try:
        with tracker.start_run("boom") as run:
            raise RuntimeError("training exploded")
    except RuntimeError:
        pass
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["status"] == "failed"
