import json
import os
from typing import Any, Dict, List

class DataLogger:
    """
    Handles appending data points in memory and periodically writing them to a JSON file.
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the DataLogger with the given filename. If file exists, it is removed
        and replaced with an empty JSON file.

        :param filename: path to the JSON file where data will be logged
        """
        self.filename = filename
        self.data: List[Dict[str, Any]] = []

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Remove existing file
        if os.path.isfile(self.filename):
            try:
                os.remove(self.filename)
            except OSError as e:
                print(f"Error deleting existing file {self.filename}: {e}")

        # Create an empty JSON file
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error creating file {self.filename}: {e}")

    def log(self, data_point: Dict[str, Any]) -> None:
        """
        Add a data point to the internal list of logged data.

        :param data_point: dictionary containing data to log
        """
        self.data.append(data_point)

    def save(self) -> None:
        """
        Write the logged data to the JSON file.
        """
        try:
            with open(self.filename, "w") as f:
                json.dump(self.data, f, indent=4)
        except IOError as e:
            print(f"Error saving data to {self.filename}: {e}")