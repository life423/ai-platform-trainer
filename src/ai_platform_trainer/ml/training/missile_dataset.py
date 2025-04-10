"""
MissileDataset: Dataset class for training missile trajectory prediction models.

This module provides a PyTorch Dataset implementation for loading and preprocessing
game state data for missile trajectory prediction.
"""
import json
import math
from typing import Tuple

import torch
from torch.utils.data import Dataset


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        x1: X-coordinate of first point
        y1: Y-coordinate of first point
        x2: X-coordinate of second point
        y2: Y-coordinate of second point

    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class MissileDataset(Dataset):
    """
    Dataset for missile trajectory prediction training.

    Loads game state data from a JSON file and provides tensor representations
    suitable for training a missile trajectory prediction model.
    """

    def __init__(self, json_file: str) -> None:
        """
        Initialize the dataset by loading data from a JSON file.

        Args:
            json_file: Path to the JSON file containing training data
        """
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample by index.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple containing (state, action, weight) tensors where:
            - state: Game state features
            - action: Target missile action
            - weight: Sample importance weight
        """
        entry = self.data[idx]
        collision_val = 1.0 if entry["missile_collision"] else 0.0

        # Calculate distance on the fly
        dist = calculate_distance(
            entry["missile_x"], entry["missile_y"], entry["enemy_x"], entry["enemy_y"]
        )

        state = [
            entry["player_x"],
            entry["player_y"],
            entry["enemy_x"],
            entry["enemy_y"],
            entry["missile_x"],
            entry["missile_y"],
            entry["missile_angle"],
            dist,
            collision_val,
        ]
        action = [entry["missile_action"]]
        weight = 2.0 if collision_val == 1.0 else 1.0

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor([weight], dtype=torch.float32),  # Wrap in list to make shape (1,)
        )
