"""
MissileDataset: Dataset class for training missile trajectory prediction models.

Rows are read in the schema-2 format written by TrainingMode: the nine model
features are stored exactly as the game builds them (see ai/missile_features.py).
Older rows stored raw pixel fields, which the game never feeds the model, so
they are skipped rather than trained on.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ai_platform_trainer.ai.missile_features import FEATURE_COUNT

# Bump this when the row layout or the meaning of the features changes.
SCHEMA_VERSION = 2


def _is_finite_number(value: Any) -> bool:
    """True for ints and floats that are not bools, NaN or infinity."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def row_problem(row: Dict[str, Any]) -> Optional[str]:
    """Why a row cannot be trained on, or None if it can."""
    if row.get("schema") != SCHEMA_VERSION:
        return f"schema is {row.get('schema')!r}, expected {SCHEMA_VERSION}"

    features = row.get("features")
    if not isinstance(features, list) or len(features) != FEATURE_COUNT:
        return f"features must be a list of {FEATURE_COUNT} numbers"
    for i, value in enumerate(features):
        if not _is_finite_number(value):
            return f"feature {i} is not a finite number: {value!r}"

    action = row.get("missile_action")
    if not _is_finite_number(action):
        return f"missile_action is not a finite number: {action!r}"
    if not isinstance(row.get("missile_collision"), bool):
        return "missile_collision must be true or false"
    timestamp = row.get("timestamp")
    if not isinstance(timestamp, int) or isinstance(timestamp, bool):
        return "timestamp must be an integer"
    return None


def is_usable_row(row: Dict[str, Any]) -> bool:
    """True if a row can be trained on."""
    return row_problem(row) is None


class MissileDataset(Dataset):
    """
    Dataset for missile trajectory prediction training.

    Loads game state data from a JSON file and keeps only the rows that can
    be trained on. The count of skipped rows is kept on `skipped`.
    """

    def __init__(self, json_file: str) -> None:
        """
        Initialize the dataset by loading data from a JSON file.

        Args:
            json_file: Path to the JSON file containing training data
        """
        with open(json_file, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        self.data = [row for row in raw if is_usable_row(row)]
        self.skipped = len(raw) - len(self.data)
        if self.skipped:
            logging.warning(
                f"MissileDataset skipped {self.skipped} of {len(raw)} rows "
                f"that are not schema {SCHEMA_VERSION} or contain bad values."
            )

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
            - state: The nine model features, in model order
            - action: Target missile action (degrees of turn)
            - weight: Sample importance weight (collisions count double)
        """
        entry = self.data[idx]
        collided = bool(entry["missile_collision"])

        return (
            torch.tensor(entry["features"], dtype=torch.float32),
            torch.tensor([entry["missile_action"]], dtype=torch.float32),
            torch.tensor([2.0 if collided else 1.0], dtype=torch.float32),
        )
