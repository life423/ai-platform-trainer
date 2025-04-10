# file: ai_platform_trainer/core/config_manager.py
"""
Configuration management system for the AI Platform Trainer.
Provides centralized access to configuration values with validation and schema checking.
"""
import json
import os
import logging
from typing import Any, Dict, Optional

# Schema definition for configuration validation
CONFIG_SCHEMA = {
    "display": {
        "width": {"type": int, "required": True, "default": 800},
        "height": {"type": int, "required": True, "default": 600},
        "fullscreen": {"type": bool, "required": True, "default": False},
        "window_title": {"type": str, "required": True, "default": "Pixel Pursuit"},
        "frame_rate": {"type": int, "required": True, "default": 60},
        "screen_size": {"type": list, "required": False}
    },
    "gameplay": {
        "respawn_delay": {"type": int, "required": True, "default": 1000},
        "player_step": {"type": int, "required": True, "default": 3},
        "player_size": {"type": int, "required": True, "default": 20},
        "enemy_step": {"type": int, "required": True, "default": 2},
        "enemy_size": {"type": int, "required": True, "default": 20},
        "missile_step": {"type": int, "required": True, "default": 4},
        "missile_size": {"type": int, "required": True, "default": 5},
        "wall_margin": {"type": int, "required": True, "default": 50},
        "min_distance": {"type": int, "required": True, "default": 100}
    },
    "ai": {
        "missile_model_path": {
            "type": str,
            "required": True,
            "default": "models/missile_model.pth"
        },
        "enemy_model_path": {"type": str, "required": True, "default": "models/enemy_ai_model.pth"},
        "input_size": {"type": int, "required": True, "default": 5},
        "hidden_size": {"type": int, "required": True, "default": 64},
        "output_size": {"type": int, "required": True, "default": 2},
        "random_speed_factor_min": {"type": float, "required": True, "default": 0.8},
        "random_speed_factor_max": {"type": float, "required": True, "default": 1.2},
        "enemy_min_speed": {"type": int, "required": True, "default": 2}
    },
    "paths": {
        "data_path": {"type": str, "required": True, "default": "data/raw/training_data.json"}
    }
}

# Default configuration values derived from the schema
DEFAULT_CONFIG = {
    "display": {
        "width": 800,
        "height": 600,
        "fullscreen": False,
        "window_title": "Pixel Pursuit",
        "frame_rate": 60,
        "screen_size": [800, 600]
    },
    "gameplay": {
        "respawn_delay": 1000,
        "player_step": 3,
        "player_size": 20,
        "enemy_step": 2,
        "enemy_size": 20,
        "missile_step": 4,
        "missile_size": 5,
        "wall_margin": 50,
        "min_distance": 100
    },
    "ai": {
        "missile_model_path": "models/missile_model.pth",
        "enemy_model_path": "models/enemy_ai_model.pth",
        "input_size": 5,
        "hidden_size": 64,
        "output_size": 2,
        "random_speed_factor_min": 0.8,
        "random_speed_factor_max": 1.2,
        "enemy_min_speed": 2
    },
    "paths": {
        "data_path": "data/raw/training_data.json"
    }
}

# Generate the default config values from the schema
for section, fields in CONFIG_SCHEMA.items():
    if section not in DEFAULT_CONFIG:
        DEFAULT_CONFIG[section] = {}

    for field, metadata in fields.items():
        if "default" in metadata and field not in DEFAULT_CONFIG[section]:
            DEFAULT_CONFIG[section][field] = metadata["default"]


class ValidationError(Exception):
    """Exception raised for configuration validation errors."""


class ConfigManager:
    """
    Configuration manager for the AI Platform Trainer.
    Provides centralized access to configuration values from various sources with validation.
    """

    def __init__(self, config_file: str = "config.json") -> None:
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.user_config_file = "user_settings.json"
        self.config: Dict[str, Any] = {}

        # Load configuration from default
        self.config.update(DEFAULT_CONFIG)

        # Load configuration from file
        self._load_from_file()

        # Load user settings that override the main config
        self._load_user_settings()

        # Validate the configuration
        self._validate_config()

        # Calculate derived values (like screen_size)
        self._calculate_derived_values()

        logging.info("Configuration manager initialized")

    def _load_from_file(self) -> None:
        """Load configuration from file."""
        self._load_config_from_file(self.config_file)

    def _load_user_settings(self) -> None:
        """Load user settings that override the main config."""
        self._load_config_from_file(self.user_config_file)

    def _load_config_from_file(self, file_path: str) -> None:
        """
        Load configuration from the specified file.

        Args:
            file_path: Path to the configuration file to load
        """
        if not os.path.exists(file_path):
            logging.info(f"Configuration file {file_path} not found. Skipping.")
            return

        try:
            with open(file_path, "r") as f:
                file_config = json.load(f)

            # Update config with values from file
            # This will recursively update nested dictionaries
            self._deep_update(self.config, file_config)
            logging.info(f"Configuration loaded from {file_path}")
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing configuration from {file_path}: {e}")
        except Exception as e:
            logging.error(f"Error loading configuration from {file_path}: {e}")

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.

        Args:
            target: The dictionary to update
            source: The dictionary to update from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # If both target and source have a dict at this key, recursively update
                self._deep_update(target[key], value)
            else:
                # Otherwise, just update the value
                target[key] = value

    def _validate_config(self) -> None:
        """
        Validate the configuration against the schema.
        Ensures that required fields exist and have the correct type.
        """
        for section, fields in CONFIG_SCHEMA.items():
            if section not in self.config:
                self.config[section] = {}
                logging.warning(f"Configuration section '{section}' is missing. Using defaults.")

            for field, metadata in fields.items():
                # Check if required field exists
                if metadata.get("required", False) and field not in self.config[section]:
                    if "default" in metadata:
                        self.config[section][field] = metadata["default"]
                        msg = (f"Required field '{section}.{field}' is missing. "
                              f"Using default value: {metadata['default']}")
                        logging.warning(msg)
                    else:
                        raise ValidationError(f"Required field '{section}.{field}' is missing")

                # Check type if the field exists
                if field in self.config[section]:
                    expected_type = metadata.get("type")
                    if expected_type and not isinstance(self.config[section][field], expected_type):
                        # Special case for lists
                        if expected_type == list and isinstance(self.config[section][field], (list, tuple)):
                            continue

                        # Try to convert the value to the expected type
                        try:
                            self.config[section][field] = expected_type(self.config[section][field])
                            logging.warning(
                                f"Field '{section}.{field}' has been converted to {expected_type.__name__}"
                            )
                        except (ValueError, TypeError):
                            error_msg = (f"Field '{section}.{field}' has incorrect type. "
                                        f"Expected {expected_type.__name__}, got "
                                        f"{type(self.config[section][field]).__name__}")
                            logging.error(error_msg)
                            # Use default if available
                            if "default" in metadata:
                                self.config[section][field] = metadata["default"]
                                default_msg = f"Using default value for '{section}.{field}': {metadata['default']}"
                                logging.warning(default_msg)

    def _calculate_derived_values(self) -> None:
        """Calculate derived configuration values."""
        # Set screen_size based on width and height if not already set
        if "screen_size" not in self.config["display"] or not self.config["display"]["screen_size"]:
            self.config["display"]["screen_size"] = [
                self.config["display"]["width"],
                self.config["display"]["height"]
            ]

    def save(self, save_to_user_settings: bool = False) -> None:
        """Save configuration to file."""
        try:
            # Save to the appropriate file
            target_file = self.user_config_file if save_to_user_settings else self.config_file

            with open(target_file, "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuration saved to {target_file}")
        except Exception as e:
            logging.error(f"Error saving configuration to {target_file}: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            key_path: Dot-separated path to the configuration value (e.g., "display.width")
            default: Default value to return if the key path is not found

        Returns:
            The configuration value, or the default value if not found
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]

        return value

    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value by key path.

        Args:
            key_path: Dot-separated path to the configuration value (e.g., "display.width")
            value: The value to set
        """
        keys = key_path.split(".")
        config = self.config

        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            elif not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]

        # Set the value
        config[keys[-1]] = value


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: str = "config.json") -> ConfigManager:
    """
    Get the singleton ConfigManager instance.

    Args:
        config_file: Path to the configuration file

    Returns:
        The ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager
