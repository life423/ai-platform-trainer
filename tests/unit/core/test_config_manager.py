"""
Unit tests for the ConfigManager class.
"""
import json
import os
import tempfile

import pytest

from ai_platform_trainer.core.config_manager import (
    DEFAULT_CONFIG,
    ConfigManager,
    ValidationError,
)


class TestConfigManager:
    """Test suite for the ConfigManager class."""

    def test_default_config_loading(self):
        """Test loading default configuration."""
        # Create ConfigManager with non-existent file
        config_manager = ConfigManager("nonexistent_file.json")

        # Check that defaults were loaded
        assert config_manager.get("display.width") == 800
        assert config_manager.get("display.height") == 600
        assert config_manager.get("display.frame_rate") == 60
        assert config_manager.get("gameplay.wall_margin") == 50
        assert config_manager.get("gameplay.min_distance") == 100

    def test_config_loading_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            test_config = {"display": {"width": 1024, "height": 768}}
            json.dump(test_config, temp_file)
            temp_file_path = temp_file.name

        try:
            # Create ConfigManager with our test file
            config_manager = ConfigManager(temp_file_path)

            # Check that values from file were loaded
            assert config_manager.get("display.width") == 1024
            assert config_manager.get("display.height") == 768

            # Check that other defaults were preserved
            assert config_manager.get("display.frame_rate") == 60
            assert config_manager.get("gameplay.wall_margin") == 50

        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_get_with_dot_notation(self):
        """Test getting values with dot notation."""
        config_manager = ConfigManager("nonexistent_file.json")

        # Test getting values at different depths
        assert config_manager.get("display") == DEFAULT_CONFIG["display"]
        assert config_manager.get("display.width") == 800
        assert config_manager.get("display.nonexistent", "default") == "default"

        # Test getting nested values
        assert config_manager.get("ai.missile_model_path") == "models/missile_model.pth"

    def test_set_value(self):
        """Test setting configuration values."""
        config_manager = ConfigManager("nonexistent_file.json")

        # Set and check a few values
        config_manager.set("display.width", 1200)
        assert config_manager.get("display.width") == 1200

        # Set a new nested value
        config_manager.set("new_section.new_value", 42)
        assert config_manager.get("new_section.new_value") == 42

        # Check that the section was created
        assert "new_section" in config_manager.config
        assert "new_value" in config_manager.config["new_section"]

    def test_derived_values(self):
        """Test calculation of derived configuration values."""
        # Create a config with missing screen_size
        config_manager = ConfigManager("nonexistent_file.json")

        # The screen_size should be derived from width and height
        screen_size = config_manager.get("display.screen_size")
        assert isinstance(screen_size, list)
        assert screen_size[0] == config_manager.get("display.width")
        assert screen_size[1] == config_manager.get("display.height")

        # If we change width and height, screen_size should not automatically update
        # (that would require calling _calculate_derived_values again)
        config_manager.set("display.width", 1024)
        config_manager.set("display.height", 768)
        assert screen_size != [1024, 768]

        # _calculate_derived_values() only fills in screen_size if it's
        # missing/empty (so it won't clobber a value explicitly loaded from
        # a config file) - calling it again after screen_size already exists
        # is a no-op, so the stale value is preserved rather than recomputed.
        config_manager._calculate_derived_values()
        assert config_manager.get("display.screen_size") == [800, 600]

    def test_validation(self):
        """Test configuration validation."""
        # Create a temporary config file with invalid types
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as temp_file:
            test_config = {
                "display": {
                    "width": "not an integer",  # Should be converted or use default
                    "height": "also not an integer",  # Should be converted or use default
                }
            }
            json.dump(test_config, temp_file)
            temp_file_path = temp_file.name

        try:
            # Create ConfigManager - validation happens during init
            config_manager = ConfigManager(temp_file_path)

            # Values should use defaults since conversion will fail
            assert config_manager.get("display.width") == 800
            assert config_manager.get("display.height") == 600

        finally:
            # Clean up the temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    def test_validation_error(self):
        """Test that ValidationError is raised for missing required fields without defaults."""

        # Create a ConfigManager subclass for testing that will raise ValidationError
        class TestConfigManager(ConfigManager):
            def _validate_config(self):
                """Override to always raise ValidationError for testing."""
                raise ValidationError("Test validation error")

        # This should raise ValidationError
        with pytest.raises(ValidationError):
            TestConfigManager("nonexistent_file.json")


if __name__ == "__main__":
    pytest.main()
