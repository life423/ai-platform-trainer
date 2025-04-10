import json
import os
from typing import Dict

DEFAULT_SETTINGS = {
    "fullscreen": False,
    "width": 1280,
    "height": 720,
}

def load_settings(config_path: str = "settings.json") -> Dict[str, bool | int]:
    """
    Load settings from a JSON file. If the file doesn't exist, returns a default dict.
    """
    if os.path.isfile(config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: Dict[str, bool | int], config_path: str = "settings.json") -> None:
    """
    Save the settings dictionary to a JSON file.
    """
    with open(config_path, "w") as file:
        json.dump(settings, file, indent=4)