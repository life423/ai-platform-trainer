"""
Data validation, appending, and model retraining utility.

This module provides functionality to:
1. Validate new training data format compatibility
2. Append new data to existing training data, but only when the existing file
   can be read and backed up first
3. Recover samples left behind by a session that never finished
4. Trigger retraining of AI models
"""

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Import training modules
from ai_platform_trainer.ai.training.missile_dataset import row_problem
from ai_platform_trainer.ai.training.train_missile_model import MissileTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataValidatorAndTrainer:
    """
    Handles validation of new training data, appending to existing data,
    and retraining of AI models when appropriate.
    """

    def __init__(
        self,
        training_data_path: str = "data/raw/training_data.json",
        missile_model_path: str = "models/missile_model.pth",
        backup_dir: str = "data/backups",
    ):
        """
        Initialize the validator and trainer.

        Args:
            training_data_path: Path to the main training data JSON file
            missile_model_path: Path to save the missile model
            backup_dir: Directory to store data backups
        """
        self.training_data_path = training_data_path
        self.missile_model_path = missile_model_path
        self.backup_dir = backup_dir

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def validate_data_format(self, new_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate that new data can be trained on.

        Args:
            new_data: List of data points to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not new_data:
            return False, "New data is empty"

        for i, data_point in enumerate(new_data):
            problem = row_problem(data_point)
            if problem is not None:
                return False, f"Data point {i}: {problem}"

        return True, ""

    def backup_existing_data(self) -> bool:
        """
        Copy the existing training data to a timestamped backup.

        The file is copied as bytes rather than parsed, so a damaged file can
        still be backed up before anything else touches it.

        Returns:
            Success status
        """
        if not os.path.exists(self.training_data_path):
            logger.info(f"No existing data file at {self.training_data_path} to backup")
            return True

        try:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup_path = os.path.join(
                self.backup_dir, f"training_data_backup_{stamp}.json"
            )
            shutil.copyfile(self.training_data_path, backup_path)
            logger.info(f"Created backup at {backup_path}")
            return True
        except OSError as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def load_existing_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Load existing training data from file.

        Returns:
            The rows, or [] if there is no file yet. None if the file exists but
            cannot be read as a list, so the caller must not overwrite it.
        """
        if not os.path.exists(self.training_data_path):
            logger.info(f"No existing data file at {self.training_data_path}")
            return []

        try:
            with open(self.training_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(f"Existing training data could not be read: {e}")
            return None

        if not isinstance(data, list):
            logger.error("Existing training data is not a list of rows")
            return None

        logger.info(f"Loaded {len(data)} existing data points")
        return data

    def merge_and_save_data(
        self, existing_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]]
    ) -> bool:
        """
        Merge new data with existing data and save to file.

        The combined rows are written to a temporary file first and then moved
        into place, so an interrupted write cannot leave a half-written dataset.

        Args:
            existing_data: Existing data points
            new_data: New data points to append

        Returns:
            Success status
        """
        tmp_path = self.training_data_path + ".tmp"
        try:
            combined_data = existing_data + new_data
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4)
            os.replace(tmp_path, self.training_data_path)
            logger.info(
                f"Saved {len(combined_data)} entries to {self.training_data_path}"
            )
            return True
        except (OSError, TypeError, ValueError) as e:
            logger.error(f"Error merging and saving data: {e}")
            return False

    def commit_samples(self, new_data: List[Dict[str, Any]]) -> bool:
        """
        Validate new samples and append them to the master dataset.

        Nothing is written unless every step succeeds: the new rows must be
        valid, the existing file must back up and read cleanly.

        Returns:
            True if the samples are in the master dataset.
        """
        valid, error_msg = self.validate_data_format(new_data)
        if not valid:
            logger.error(f"Samples rejected: {error_msg}")
            return False

        if not self.backup_existing_data():
            logger.error("Not merging: the existing data could not be backed up.")
            return False

        existing_data = self.load_existing_data()
        if existing_data is None:
            logger.error("Not merging: the existing data was left unchanged.")
            return False

        return self.merge_and_save_data(existing_data, new_data)

    def recover_session_file(self, session_path: str) -> int:
        """
        Merge samples left behind by a session that did not finish cleanly.

        The file is removed only after its samples are in the master dataset,
        so a failure here never loses data.

        Returns:
            How many samples were recovered (0 if there was nothing to recover).
        """
        if not os.path.exists(session_path):
            return 0

        try:
            with open(session_path, "r", encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, ValueError) as e:
            logger.error(f"Leftover session file {session_path} is unreadable: {e}")
            return 0

        if not isinstance(rows, list) or not rows:
            os.remove(session_path)
            return 0

        if not self.commit_samples(rows):
            logger.error(f"Leftover samples were kept in {session_path}")
            return 0

        os.remove(session_path)
        logger.info(f"Recovered {len(rows)} samples from an unfinished session")
        return len(rows)

    def train_missile_model(self) -> bool:
        """
        Train the missile trajectory prediction model.

        Returns:
            Success status
        """
        try:
            logger.info("Starting missile model training...")
            trainer = MissileTrainer(
                filename=self.training_data_path,
                model_save_path=self.missile_model_path,
            )
            trainer.run_training()
            model_path = self.missile_model_path
            logger.info(f"Missile model training completed and saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error training missile model: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    logger.info("This module is meant to be imported, not run directly.")
