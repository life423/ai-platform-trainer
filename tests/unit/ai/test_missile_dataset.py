"""
Unit tests for MissileDataset.

Tests the initialization, data loading, and item retrieval functionality
of the MissileDataset class.
"""

import json
import math

import pytest  # noqa: F401 - Used for fixture decorators and approx
import torch

from ai_platform_trainer.ai.training.missile_dataset import MissileDataset


def _good_row(action: float = 5.0, collided: bool = False) -> dict:
    return {
        "schema": 2,
        "features": [0.1, 0.1, 0.3, 0.3, 0.12, 0.12, 0.785, 0.25, 0.0],
        "missile_action": action,
        "missile_collision": collided,
        "timestamp": 1000,
    }


class TestMissileDataset:
    """Test suite for MissileDataset class and related functions."""

    def test_dataset_initialization(self, sample_training_json_path):
        """Test dataset initialization with JSON file."""
        dataset = MissileDataset(json_file=sample_training_json_path)

        assert len(dataset.data) == 2
        assert dataset.data[0]["schema"] == 2
        assert dataset.skipped == 0

    def test_dataset_length(self, sample_training_json_path):
        """Test __len__ method returns correct count."""
        dataset = MissileDataset(json_file=sample_training_json_path)
        assert len(dataset) == 2

    def test_dataset_getitem(self, sample_training_json_path):
        """Test __getitem__ method returns expected tensors."""
        dataset = MissileDataset(json_file=sample_training_json_path)

        state, action, weight = dataset[0]

        # Check types
        assert isinstance(state, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(weight, torch.Tensor)

        # Check shapes
        assert state.shape == (9,)
        assert action.shape == (1,)
        assert weight.shape == (1,)

        # Check dtypes
        assert state.dtype == torch.float32
        assert action.dtype == torch.float32
        assert weight.dtype == torch.float32

        # The state is the stored feature vector, unchanged
        assert state[0].item() == pytest.approx(0.1)  # player_x
        assert state[6].item() == pytest.approx(0.785)  # heading
        assert state[8].item() == pytest.approx(0.0)  # always zero at play time

        # Check action value
        assert action[0].item() == pytest.approx(5.0)

        # Check weight based on collision status (false = 1.0)
        assert weight[0].item() == pytest.approx(1.0)

    def test_collision_weights(self, sample_training_json_path):
        """Test that collision samples get higher weights."""
        dataset = MissileDataset(json_file=sample_training_json_path)

        _, _, weight_non_collision = dataset[0]  # non-collision sample
        _, _, weight_collision = dataset[1]  # collision sample

        assert weight_collision.item() > weight_non_collision.item()
        assert weight_collision.item() == pytest.approx(2.0)
        assert weight_non_collision.item() == pytest.approx(1.0)

    def test_collision_flag_is_not_a_feature(self, sample_training_json_path):
        """The collision outcome only changes the weight, never the model input."""
        dataset = MissileDataset(json_file=sample_training_json_path)
        state_no_hit, _, _ = dataset[0]
        state_hit, _, _ = dataset[1]

        assert state_hit[8].item() == pytest.approx(0.0)
        assert state_no_hit[8].item() == pytest.approx(0.0)

    def test_skips_legacy_and_bad_rows(self, tmp_path):
        """Only schema-2 rows with finite values are trained on."""
        legacy = {
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
        nan_row = _good_row()
        nan_row["features"][3] = float("nan")
        bool_label_row = _good_row(action=True)  # bool is not a number

        rows = [_good_row(), legacy, nan_row, bool_label_row]
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps(rows), encoding="utf-8")

        dataset = MissileDataset(json_file=str(path))

        assert len(dataset) == 1
        assert dataset.skipped == 3
        assert all(math.isfinite(v) for v in dataset[0][0].tolist())
