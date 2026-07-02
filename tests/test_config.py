"""The config layer must fail fast on anything invalid."""

import pytest
import yaml

from gym_churn.config import REPO_ROOT, load_config


def _default_payload() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "config.yaml").read_text())


def _write(tmp_path, payload) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return str(path)


def test_default_config_loads():
    config = load_config()
    assert config.project.name == "gym-churn-prediction"
    assert config.snapshots.test_snapshot == max(config.snapshots.dates)
    assert config.paths.raw_dir.is_absolute()


def test_train_dates_exclude_test_snapshot():
    config = load_config()
    assert config.snapshots.test_snapshot not in config.snapshots.train_dates
    assert len(config.snapshots.train_dates) == len(config.snapshots.dates) - 1


def test_value_per_save_math():
    config = load_config()
    business = config.business
    assert business.value_per_save == pytest.approx(
        business.avg_monthly_fee * business.retention_months
    )


def test_rejects_test_snapshot_not_in_dates(tmp_path):
    payload = _default_payload()
    payload["snapshots"]["test_snapshot"] = "2030-01-01"
    with pytest.raises(ValueError, match="test_snapshot"):
        load_config(_write(tmp_path, payload))


def test_rejects_non_latest_test_snapshot(tmp_path):
    payload = _default_payload()
    payload["snapshots"]["test_snapshot"] = payload["snapshots"]["dates"][0]
    with pytest.raises(ValueError, match="most recent"):
        load_config(_write(tmp_path, payload))


def test_rejects_unknown_keys(tmp_path):
    payload = _default_payload()
    payload["project"]["typo_key"] = 1
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, payload))


def test_rejects_invalid_business_values(tmp_path):
    payload = _default_payload()
    payload["business"]["save_rate"] = 1.5
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, payload))


def test_rejects_reversed_simulation_window(tmp_path):
    payload = _default_payload()
    payload["simulation"]["window_start"] = "2027-01-01"
    with pytest.raises(ValueError, match="window_end"):
        load_config(_write(tmp_path, payload))


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does/not/exist.yaml")
