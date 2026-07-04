"""
Configuration constants for the AI Platform Trainer.

This module provides a centralized location for all configuration constants.
"""
import os


class Config:
    """Configuration constants for the game."""

    # Display settings
    WINDOW_TITLE = "AI Platform Trainer"
    SCREEN_SIZE = (1280, 720)
    FRAME_RATE = 60

    # Game settings
    PLAYER_SIZE = 50
    ENEMY_SIZE = 50
    MISSILE_SIZE = 10
    PLAYER_SPEED = 5
    ENEMY_SPEED = 5
    MISSILE_SPEED = 5

    # Colors
    COLOR_BACKGROUND = (135, 206, 235)  # Light blue
    COLOR_PLAYER = (0, 0, 139)  # Dark blue
    COLOR_ENEMY = (139, 0, 0)  # Dark red
    COLOR_MISSILE = (255, 255, 0)  # Yellow

    # Paths
    ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "training_data.json")
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "enemy_ai_model.pth")
    MISSILE_MODEL_PATH = os.path.join(ROOT_DIR, "models", "missile_model.pth")

    # Game mechanics
    RESPAWN_DELAY = 1000  # milliseconds
    MIN_SPAWN_DISTANCE = 300  # pixels
    WALL_MARGIN = 50  # pixels
    MIN_DISTANCE = 200  # minimum distance between entities

    # Training settings
    TRAINING_EPISODES = 1000
    SAVE_INTERVAL = 100


# Create a singleton instance for easy import
config = Config()
