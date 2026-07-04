"""
Unit tests for MissileDataset.

Tests the initialization, data loading, and item retrieval functionality
of the MissileDataset class.
"""
import pytest  # noqa: F401 - Used for fixture decorators and approx
import torch

from ai_platform_trainer.ai.training.missile_dataset import (
    MissileDataset,
    calculate_distance,
)


class TestMissileDataset:
    """Test suite for MissileDataset class and related functions."""

    def test_calculate_distance(self):
        """Test that distance calculation works correctly."""
        # Test with simple case: points along x-axis
        assert calculate_distance(0.0, 0.0, 3.0, 0.0) == 3.0

        # Test with simple case: points along y-axis
        assert calculate_distance(0.0, 0.0, 0.0, 4.0) == 4.0

        # Test with Pythagorean triple (3,4,5)
        assert calculate_distance(0.0, 0.0, 3.0, 4.0) == 5.0

        # Test with negative coordinates
        assert calculate_distance(-1.0, -1.0, 2.0, 3.0) == 5.0

        # Test with floating point coordinates
        assert abs(calculate_distance(1.5, 2.5, 4.5, 6.5) - 5.0) < 1e-10

    def test_dataset_initialization(self, sample_training_json_path):
        """Test dataset initialization with JSON file."""
        # Initialize dataset with test JSON file
        dataset = MissileDataset(json_file=sample_training_json_path)

        # Check that data is loaded correctly
        assert len(dataset.data) == 2
        assert "player_x" in dataset.data[0]
        assert "missile_collision" in dataset.data[1]

    def test_dataset_length(self, sample_training_json_path):
        """Test __len__ method returns correct count."""
        dataset = MissileDataset(json_file=sample_training_json_path)
        assert len(dataset) == 2

    def test_dataset_getitem(self, sample_training_json_path):
        """Test __getitem__ method returns expected tensors."""
        dataset = MissileDataset(json_file=sample_training_json_path)

        # Get the first item
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

        # Check specific values from the state tensor
        assert state[0].item() == pytest.approx(100.0)  # player_x
        assert state[1].item() == pytest.approx(100.0)  # player_y
        assert state[6].item() == pytest.approx(0.785)  # missile_angle

        # Check action value
        assert action[0].item() == pytest.approx(5.0)

        # Check weight based on collision status (false = 1.0)
        assert weight[0].item() == pytest.approx(1.0)

    def test_collision_weights(self, sample_training_json_path):
        """Test that collision samples get higher weights."""
        dataset = MissileDataset(json_file=sample_training_json_path)

        # Get both items
        _, _, weight_non_collision = dataset[0]  # non-collision sample
        _, _, weight_collision = dataset[1]  # collision sample

        # Check weights
        assert weight_collision.item() > weight_non_collision.item()
        assert weight_collision.item() == pytest.approx(2.0)
        assert weight_non_collision.item() == pytest.approx(1.0)

    def test_calculated_distance(self, sample_training_json_path):
        """Test that distance is calculated and included correctly in state tensor."""
        dataset = MissileDataset(json_file=sample_training_json_path)
        state, _, _ = dataset[0]

        # Calculate expected distance
        expected_distance = calculate_distance(
            dataset.data[0]["missile_x"],
            dataset.data[0]["missile_y"],
            dataset.data[0]["enemy_x"],
            dataset.data[0]["enemy_y"],
        )

        # Check distance in state tensor (index 7)
        assert state[7].item() == pytest.approx(expected_distance)
