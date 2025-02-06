import json
import os
import logging

DEFAULT_SETTINGS = {"fullscreen": False, "width": 1280, "height": 720}

def load_settings(config_path: str = "settings.json") -> dict:
    """
    Loads settings from a JSON file. If the file doesn't exist,
    returns a default settings dict. If JSON parse fails,
    logs an error and returns defaults.
    """
    if not os.path.isfile(config_path):
        logging.warning(f"Settings file not found: {config_path}, using defaults.")
        return DEFAULT_SETTINGS.copy()

    try:
        with open(config_path, "r") as file:
            data = json.load(file)
            return data
    except (json.JSONDecodeError, OSError) as e:
        logging.error(f"Failed to load settings from {config_path}: {e}")
        return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict, config_path: str = "settings.json") -> None:
    """
    Saves the settings dictionary to a JSON file with indentation.
    """
    try:
        with open(config_path, "w") as file:
            json.dump(settings, file, indent=4)
        logging.info(f"Settings saved to {config_path}")
    except OSError as e:
        logging.error(f"Could not save settings to {config_path}: {e}")