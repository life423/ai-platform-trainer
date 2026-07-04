"""
Global pytest fixtures for AI Platform Trainer tests.

This module provides shared test fixtures that can be used across
all test modules, including mock data, model instances, and utility helpers.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from ai_platform_trainer.ai.models.simple_missile_model import SimpleMissileModel


@pytest.fixture
def sample_training_data() -> List[Dict[str, Any]]:
    """
    Create a small sample of training data for testing.

    Returns:
        List of dictionaries with game state data
    """
    return [
        {
            "player_x": 100.0,
            "player_y": 100.0,
            "enemy_x": 300.0,
            "enemy_y": 300.0,
            "missile_x": 120.0,
            "missile_y": 120.0,
            "missile_angle": 0.785,  # 45 degrees in radians
            "missile_action": 5.0,
            "missile_collision": False,
        },
        {
            "player_x": 150.0,
            "player_y": 150.0,
            "enemy_x": 350.0,
            "enemy_y": 350.0,
            "missile_x": 170.0,
            "missile_y": 170.0,
            "missile_angle": 0.785,
            "missile_action": -2.0,
            "missile_collision": True,
        },
    ]


@pytest.fixture
def sample_training_json_path(
    tmp_path: Path, sample_training_data: List[Dict[str, Any]]
) -> str:
    """
    Create a temporary JSON file with sample training data.

    Args:
        tmp_path: pytest fixture providing a temporary directory
        sample_training_data: sample data to write to the file

    Returns:
        Path to the created JSON file
    """
    json_path = tmp_path / "test_training_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_training_data, f)
    return str(json_path)


@pytest.fixture
def simple_missile_model() -> SimpleMissileModel:
    """
    Create a SimpleMissileModel instance with deterministic weights.

    Returns:
        Initialized SimpleMissileModel with preset weights
    """
    model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize with deterministic weights
    for param in model.parameters():
        param.data.fill_(0.01)

    return model


@pytest.fixture
def mock_game_state() -> Dict[str, Any]:
    """
    Create a mock game state for testing.

    Returns:
        Dictionary with game state information
    """
    return {
        "player": {"x": 100, "y": 100},
        "enemy": {"x": 300, "y": 300},
        "missiles": [
            {
                "pos": {"x": 120, "y": 120},
                "vx": 5.0,
                "vy": 5.0,
                "speed": 7.07,  # sqrt(5^2 + 5^2)
                "last_action": 0.0,
            }
        ],
    }
