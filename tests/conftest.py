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

from ai_platform_trainer.ai.models.missile_model import MissileModel


@pytest.fixture
def sample_training_data() -> List[Dict[str, Any]]:
    """
    Create a small sample of training data for testing.

    Returns:
        List of schema-2 rows: the nine model features as the game builds
        them, plus the label and collision flag
    """
    return [
        {
            "schema": 2,
            "features": [0.1, 0.1, 0.3, 0.3, 0.12, 0.12, 0.785, 0.25, 0.0],
            "missile_action": 5.0,
            "missile_collision": False,
            "timestamp": 1000,
        },
        {
            "schema": 2,
            "features": [0.15, 0.15, 0.35, 0.35, 0.17, 0.17, 0.785, 0.2, 0.0],
            "missile_action": -2.0,
            "missile_collision": True,
            "timestamp": 1016,
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
def missile_model() -> MissileModel:
    """
    Create a MissileModel instance with deterministic weights.

    Returns:
        Initialized MissileModel with preset weights
    """
    model = MissileModel(input_size=9, hidden_size=64, output_size=1)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize with deterministic weights
    for param in model.parameters():
        param.data.fill_(0.01)

    return model
