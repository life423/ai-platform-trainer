"""
Game settings and configuration constants.

This module provides centralized game configuration management.
"""
import os


class GameSettings:
    """Game configuration constants."""
    
    # Display settings
    WINDOW_TITLE = "AI Platform Trainer"
    SCREEN_SIZE = (1280, 720)
    FRAME_RATE = 60
    
    # Game mechanics
    PLAYER_SIZE = 50
    ENEMY_SIZE = 50
    MISSILE_SIZE = 10
    PLAYER_SPEED = 5
    ENEMY_SPEED = 5
    MISSILE_SPEED = 5
    
    # Colors (light blue theme)
    COLOR_BACKGROUND = (135, 206, 235)  # Light blue
    COLOR_PLAYER = (0, 0, 139)          # Dark blue
    COLOR_ENEMY = (139, 0, 0)           # Dark red
    COLOR_MISSILE = (255, 255, 0)       # Yellow
    
    # Game mechanics
    RESPAWN_DELAY = 1000  # milliseconds
    MIN_SPAWN_DISTANCE = 300  # pixels
    WALL_MARGIN = 50  # pixels
    MIN_DISTANCE = 200  # minimum distance between entities
    
    # Training settings
    TRAINING_EPISODES = 1000
    SAVE_INTERVAL = 100
    AUTO_SAVE_INTERVAL = 30000  # 30 seconds


# Create singleton instance
settings = GameSettings()