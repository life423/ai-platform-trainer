"""
Tests for the training-data pipeline: row checks, merging, crash recovery, and
the guard that stops training from replacing a good model with a blank one.
"""

import json
import os

import pytest
import torch

from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.ai.training.missile_dataset import row_problem
from ai_platform_trainer.ai.training.train_missile_model import MissileTrainer
from ai_platform_trainer.utils.data_validator_and_trainer import DataValidatorAndTrainer


def _row(action: float = 5.0, collided: bool = False, value: float = 0.1) -> dict:
    return {
        "schema": 2,
        "features": [value] * 8 + [0.0],
        "missile_action": action,
        "missile_collision": collided,
        "timestamp": 1000,
    }


def _legacy_row() -> dict:
    return {
        "player_x": 100.0,
        "player_y": 100.0,
        "enemy_x": 300.0,
        "enemy_y": 300.0,
        "missile_x": 120.0,
        "missile_y": 120.0,
        "missile_angle": 0.785,
        "missile_action": 5.0,
        "missile_collision": False,
        "timestamp": 1000,
    }


def _write_json(path, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def paths(tmp_path):
    return {
        "master": str(tmp_path / "data" / "raw" / "training_data.json"),
        "backups": str(tmp_path / "data" / "backups"),
        "session": str(tmp_path / "data" / "raw" / "_training_session.json"),
        "model": str(tmp_path / "models" / "missile_model.pth"),
    }


@pytest.fixture
def validator(paths):
    return DataValidatorAndTrainer(
        training_data_path=paths["master"],
        missile_model_path=paths["model"],
        backup_dir=paths["backups"],
    )


# Row checks


def test_good_row_has_no_problem():
    assert row_problem(_row()) is None


def test_legacy_row_is_rejected():
    assert "schema" in row_problem(_legacy_row())


def test_bool_label_is_rejected():
    assert row_problem(_row(action=True)) is not None


def test_nan_feature_is_rejected():
    row = _row()
    row["features"][2] = float("nan")
    assert row_problem(row) is not None


def test_wrong_feature_count_is_rejected():
    row = _row()
    row["features"] = row["features"][:8]
    assert row_problem(row) is not None


def test_validator_rejects_a_batch_with_one_bad_row(validator):
    valid, message = validator.validate_data_format([_row(), _legacy_row()])

    assert not valid
    assert "Data point 1" in message


# Merging


def test_commit_creates_the_master_file_when_missing(validator, paths):
    assert validator.commit_samples([_row(), _row(action=-2.0)])

    assert len(_read_json(paths["master"])) == 2


def test_commit_appends_and_backs_up_the_previous_file(validator, paths):
    _write_json(paths["master"], [_row(value=0.5)])

    assert validator.commit_samples([_row(value=0.2)])

    assert len(_read_json(paths["master"])) == 2
    backups = os.listdir(paths["backups"])
    assert len(backups) == 1
    assert len(_read_json(os.path.join(paths["backups"], backups[0]))) == 1


def test_commit_leaves_an_unreadable_master_unchanged(validator, paths):
    os.makedirs(os.path.dirname(paths["master"]), exist_ok=True)
    with open(paths["master"], "w", encoding="utf-8") as f:
        f.write("{not json")

    assert not validator.commit_samples([_row()])

    with open(paths["master"], "r", encoding="utf-8") as f:
        assert f.read() == "{not json"


def test_commit_leaves_no_temporary_file(validator, paths):
    validator.commit_samples([_row()])

    assert not os.path.exists(paths["master"] + ".tmp")


def test_commit_rejects_bad_samples_without_touching_the_master(validator, paths):
    _write_json(paths["master"], [_row()])

    assert not validator.commit_samples([_legacy_row()])

    assert len(_read_json(paths["master"])) == 1


# Crash recovery


def test_recovery_merges_and_removes_the_session_file(validator, paths):
    _write_json(paths["session"], [_row(), _row(value=0.3)])

    recovered = validator.recover_session_file(paths["session"])

    assert recovered == 2
    assert not os.path.exists(paths["session"])
    assert len(_read_json(paths["master"])) == 2


def test_recovery_keeps_the_session_when_the_master_is_unreadable(validator, paths):
    os.makedirs(os.path.dirname(paths["master"]), exist_ok=True)
    with open(paths["master"], "w", encoding="utf-8") as f:
        f.write("{not json")
    _write_json(paths["session"], [_row()])

    assert validator.recover_session_file(paths["session"]) == 0

    assert os.path.exists(paths["session"])


def test_recovery_keeps_an_unreadable_session_file(validator, paths):
    os.makedirs(os.path.dirname(paths["session"]), exist_ok=True)
    with open(paths["session"], "w", encoding="utf-8") as f:
        f.write("{half written")

    assert validator.recover_session_file(paths["session"]) == 0
    assert os.path.exists(paths["session"])


def test_recovery_with_no_session_file_does_nothing(validator, paths):
    assert validator.recover_session_file(paths["session"]) == 0


# Retraining guard


def _write_model(path: str) -> bytes:
    """Write a real model file and return its bytes, to check it is left alone."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(MissileModel().state_dict(), path)
    with open(path, "rb") as f:
        return f.read()


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_trainer_refuses_a_dataset_with_no_usable_rows(paths):
    _write_json(paths["master"], [_legacy_row()])
    original = _write_model(paths["model"])

    with pytest.raises(ValueError, match="No usable training rows"):
        MissileTrainer(
            filename=paths["master"], epochs=1, model_save_path=paths["model"]
        )

    assert _read_bytes(paths["model"]) == original


def test_retraining_failure_is_reported_and_keeps_the_model(validator, paths):
    _write_json(paths["master"], [_legacy_row()])
    original = _write_model(paths["model"])

    assert not validator.train_missile_model()

    assert _read_bytes(paths["model"]) == original


def test_training_saves_the_model_without_leaving_a_temporary_file(validator, paths):
    _write_json(paths["master"], [_row(), _row(action=-3.0, collided=True)])

    assert validator.train_missile_model()

    assert os.path.exists(paths["model"])
    assert not os.path.exists(paths["model"] + ".tmp")
    model = MissileModel()
    model.load_state_dict(torch.load(paths["model"]))
