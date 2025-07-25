"""
Clean Data Logger

This module provides simplified data logging functionality.
"""
import json
import os
import logging
from typing import Any, Dict, List

from config.paths import paths


class DataLogger:
    """
    Handles data collection and saving for training.
    """

    def __init__(self):
        """Initialize the DataLogger."""
        self.data: List[Dict[str, Any]] = []
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(paths.TRAINING_DATA_FILE), exist_ok=True)
        
        # Load existing data if it exists
        self._load_existing_data()
        
        logging.info(f"DataLogger initialized with {len(self.data)} existing data points")

    def _load_existing_data(self):
        """Load existing training data if it exists."""
        if os.path.exists(paths.TRAINING_DATA_FILE):
            try:
                with open(paths.TRAINING_DATA_FILE, "r") as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        self.data = existing_data
                    else:
                        self.data = []
                        logging.warning("Existing data file format invalid, starting fresh")
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not load existing data: {e}")
                self.data = []

    def log(self, data_point: Dict[str, Any]) -> None:
        """
        Add a data point to the collection.

        Args:
            data_point: Dictionary containing data to log
        """
        self.data.append(data_point)

    def save(self) -> bool:
        """
        Save the collected data to the JSON file.
        
        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(paths.TRAINING_DATA_FILE, "w") as f:
                json.dump(self.data, f, indent=2)
            logging.debug(f"Saved {len(self.data)} data points to {paths.TRAINING_DATA_FILE}")
            return True
        except IOError as e:
            logging.error(f"Error saving data to {paths.TRAINING_DATA_FILE}: {e}")
            return False

    def clear(self):
        """Clear all collected data."""
        self.data.clear()
        logging.info("Data logger cleared")

    def get_data_count(self) -> int:
        """Get the number of data points collected."""
        return len(self.data)