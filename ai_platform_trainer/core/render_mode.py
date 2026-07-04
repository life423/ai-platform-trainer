"""
Render mode definitions for AI Platform Trainer.

This module defines the rendering modes available in the game,
allowing for headless operation during training.
"""
from enum import Enum, auto


class RenderMode(Enum):
    """Rendering modes for the game."""

    FULL = auto()  # Full rendering with window
    HEADLESS = auto()  # No rendering, console only
