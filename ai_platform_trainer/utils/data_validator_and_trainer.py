"""
Data validation, appending, and model retraining utility.

This module provides functionality to:
1. Validate new training data format compatibility
2. Append new data to existing training data
3. Trigger retraining of AI models
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Tuple

# Import training modules
from ai_platform_trainer.ai.training.train_missile_model import MissileTrainer
from ai_platform_trainer.ai.training.train_enemy_rl import train_rl_agent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidatorAndTrainer:
    """
    Handles validation of new training data, appending to existing data,
    and retraining of AI models when appropriate.
    """

    # Define expected data schema for validation
    EXPECTED_SCHEMA = {
        "player_x": float,
        "player_y": float,
        "enemy_x": float,
        "enemy_y": float,
        "missile_x": float,
        "missile_y": float,
        "missile_angle": float,
        "missile_collision": bool,
        "missile_action": float,
        "timestamp": int
    }

    def __init__(
        self,
        training_data_path: str = "data/raw/training_data.json",
        missile_model_path: str = "models/missile_model.pth",
        enemy_model_path: str = "models/enemy_rl",
        backup_dir: str = "data/backups"
    ):
        """
        Initialize the validator and trainer.

        Args:
            training_data_path: Path to the main training data JSON file
            missile_model_path: Path to save the missile model
            enemy_model_path: Directory to save enemy RL model
            backup_dir: Directory to store data backups
        """
        self.training_data_path = training_data_path
        self.missile_model_path = missile_model_path
        self.enemy_model_path = enemy_model_path
        self.backup_dir = backup_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def validate_data_format(self, new_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate that new data matches the expected format.

        Args:
            new_data: List of data points to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not new_data:
            return False, "New data is empty"
        
        # Check each data point for required fields and types
        for i, data_point in enumerate(new_data):
            # Check for missing keys
            for key in self.EXPECTED_SCHEMA:
                if key not in data_point:
                    return False, f"Data point {i} missing required key: {key}"
            
            # Check for correct types
            for key, expected_type in self.EXPECTED_SCHEMA.items():
                actual_value = data_point[key]
                # Handle None values (decide if they should be allowed)
                if actual_value is None:
                    expected_name = expected_type.__name__
                    return False, f"Data point {i}: {key} is None but should be {expected_name}"
                
                # Type checking with special handling for numeric types
                if expected_type in (float, int):
                    if not isinstance(actual_value, (float, int)):
                        type_name = type(actual_value).__name__
                        expected = expected_type.__name__
                        msg = f"Data point {i}: {key} has type {type_name} but should be {expected}"
                        return False, msg
                elif not isinstance(actual_value, expected_type):
                    type_name = type(actual_value).__name__
                    expected = expected_type.__name__
                    msg = f"Data point {i}: {key} has type {type_name} but should be {expected}"
                    return False, msg
        
        return True, ""

    def backup_existing_data(self) -> bool:
        """
        Create a backup of the existing training data.

        Returns:
            Success status
        """
        if not os.path.exists(self.training_data_path):
            path = self.training_data_path
            logger.info(f"No existing data file at {path} to backup")
            return True
        
        try:
            # Create a timestamped backup filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_filename = f"training_data_backup_{timestamp}.json"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Copy the file
            with open(self.training_data_path, 'r') as source:
                data = json.load(source)
                with open(backup_path, 'w') as target:
                    json.dump(data, target, indent=4)
            
            logger.info(f"Created backup at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return False

    def load_existing_data(self) -> List[Dict[str, Any]]:
        """
        Load existing training data from file.

        Returns:
            List of data points or empty list if file doesn't exist
        """
        if not os.path.exists(self.training_data_path):
            path = self.training_data_path
            logger.info(f"No existing data file at {path}")
            return []
        
        try:
            with open(self.training_data_path, 'r') as f:
                data = json.load(f)
                count = len(data)
                logger.info(f"Loaded {count} existing data points")
                return data
        except json.JSONDecodeError:
            logger.error("Existing data file is not valid JSON. Creating new file.")
            return []
        except Exception as e:
            error = str(e)
            logger.error(f"Error loading existing data: {error}")
            return []

    def merge_and_save_data(self, existing_data: List[Dict[str, Any]], new_data: List[Dict[str, Any]]) -> bool:
        """
        Merge new data with existing data and save to file.

        Args:
            existing_data: Existing data points
            new_data: New data points to append

        Returns:
            Success status
        """
        try:
            # Combine the data
            combined_data = existing_data + new_data
            total = len(combined_data)
            logger.info(f"Combined data now has {total} entries")
            
            # Save to file
            with open(self.training_data_path, 'w') as f:
                json.dump(combined_data, f, indent=4)
            
            data_path = self.training_data_path
            msg = f"Successfully saved combined data to {data_path}"
            logger.info(msg)
            return True
        except Exception as e:
            logger.error(f"Error merging and saving data: {e}")
            return False

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
                model_save_path=self.missile_model_path
            )
            trainer.run_training()
            model_path = self.missile_model_path
            logger.info(f"Missile model training completed and saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error training missile model: {e}")
            return False

    def train_enemy_model(self) -> bool:
        """
        Train the enemy RL model.

        Returns:
            Success status
        """
        try:
            # Use a reduced number of timesteps for faster training
            # This can be configured based on available time
            logger.info("Starting enemy RL model training...")
            timesteps = 100000  # Adjust as needed
            
            model = train_rl_agent(
                total_timesteps=timesteps,
                save_path=self.enemy_model_path,
                log_path="logs/enemy_rl",
                headless=True
            )
            
            if model:
                model_path = self.enemy_model_path
                logger.info(f"Enemy RL model training completed and saved to {model_path}")
                return True
            else:
                logger.error("Enemy RL training returned None")
                return False
        except Exception as e:
            logger.error(f"Error training enemy RL model: {e}")
            return False

    def process_new_data(self, new_data: List[Dict[str, Any]]) -> bool:
        """
        Main workflow: validate new data, append to existing data, and retrain models.

        Args:
            new_data: New training data to process

        Returns:
            Overall success status
        """
        # Step 1: Validate data format
        valid, error_msg = self.validate_data_format(new_data)
        if not valid:
            msg = f"Data validation failed: {error_msg}"
            logger.error(msg)
            return False
        
        logger.info(f"Validated {len(new_data)} new data points")
        
        # Step 2: Backup existing data
        if not self.backup_existing_data():
            logger.warning("Proceeding without backup")
        
        # Step 3: Load existing data
        existing_data = self.load_existing_data()
        
        # Step 4: Merge and save the combined data
        if not self.merge_and_save_data(existing_data, new_data):
            return False
        
        # Step 5: Retrain models
        missile_success = self.train_missile_model()
        enemy_success = self.train_enemy_model()
        
        return missile_success and enemy_success


# Convenience function for external use
def process_new_training_data(new_data: List[Dict[str, Any]]) -> bool:
    """
    Validate, append, and retrain models with new training data.

    Args:
        new_data: New training data to process

    Returns:
        Success status
    """
    processor = DataValidatorAndTrainer()
    return processor.process_new_data(new_data)


if __name__ == "__main__":
    # Example usage
    logger.info("This module is meant to be imported, not run directly.")
