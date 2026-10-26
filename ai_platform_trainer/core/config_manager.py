"""The player's saved settings. Right now that is only fullscreen, kept in config.json.

Window size, speeds, sizes and timers belong to the arena (arena/config.py) and to
gameplay/config.py, so they are not configurable here.
"""
import copy
import json
import logging
import os
from typing import Any, Dict, Optional

DEFAULT_SETTINGS: Dict[str, Any] = {"display": {"fullscreen": False}}

_TRUE_STRINGS = {"true", "1", "yes", "on"}
_FALSE_STRINGS = {"false", "0", "no", "off"}


def _as_bool(value: Any, default: bool) -> bool:
    """Read a saved fullscreen value. Strings like "false" are false, not truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    logging.warning(f"Ignoring unreadable fullscreen setting {value!r}")
    return default


class ConfigManager:
    """Loads and saves the settings file. Unknown keys are kept as they are."""

    def __init__(self, config_file: str = "config.json") -> None:
        self.config_file = config_file
        self.config: Dict[str, Any] = copy.deepcopy(DEFAULT_SETTINGS)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, encoding="utf-8") as fh:
                saved = json.load(fh)
        except (OSError, json.JSONDecodeError) as error:
            logging.error(f"Could not read {self.config_file}: {error}")
            return
        display = saved.get("display", {}) if isinstance(saved, dict) else {}
        fullscreen = display.get("fullscreen", DEFAULT_SETTINGS["display"]["fullscreen"])
        self.config["display"]["fullscreen"] = _as_bool(
            fullscreen, DEFAULT_SETTINGS["display"]["fullscreen"]
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """The value at a dotted path such as "display.fullscreen", or default."""
        value: Any = self.config
        for key in key_path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set the value at a dotted path, creating sections as needed."""
        keys = key_path.split(".")
        node = self.config
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def save(self) -> None:
        """Write the settings file."""
        try:
            with open(self.config_file, "w", encoding="utf-8") as fh:
                json.dump(self.config, fh, indent=4)
            logging.info(f"Settings saved to {self.config_file}")
        except OSError as error:
            logging.error(f"Could not save settings to {self.config_file}: {error}")


_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_file: str = "config.json") -> ConfigManager:
    """The shared ConfigManager. The first call decides which file it uses."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager
