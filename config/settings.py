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
    
    # Colors with good contrast ratios (WCAG AA compliant)
    COLOR_BACKGROUND = (135, 206, 235)  # Light blue (#87CEEB)
    COLOR_PLAYER = (25, 25, 112)        # Midnight blue (#191970) - High contrast on light blue
    COLOR_ENEMY = (178, 34, 34)         # Fire brick red (#B22222) - Good contrast, less harsh than pure red
    COLOR_MISSILE = (255, 140, 0)       # Dark orange (#FF8C00) - Better visibility than yellow on light blue
    
    # UI Colors for accessibility
    COLOR_TEXT_PRIMARY = (255, 255, 255)    # White - High contrast on dark backgrounds
    COLOR_TEXT_SECONDARY = (240, 248, 255)  # Alice blue - Softer white
    COLOR_SELECTED = (255, 215, 0)          # Gold (#FFD700) - High contrast selection
    COLOR_NORMAL = (220, 220, 220)          # Light gray - Good readability
    COLOR_DESCRIPTION = (192, 192, 192)     # Silver - Muted but readable
    
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