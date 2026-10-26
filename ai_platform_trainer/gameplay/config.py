"""
Configuration constants for the AI Platform Trainer.

This module provides a centralized location for all configuration constants.
"""


class Config:
    """Configuration constants for the game."""

    # Display settings
    WINDOW_TITLE = "Pixel Pursuit"
    SCREEN_SIZE = (1280, 720)
    FRAME_RATE = 60

    # Colors
    COLOR_BACKGROUND = (135, 206, 235)  # Light blue

    # Paths


# Create a singleton instance for easy import
config = Config()
