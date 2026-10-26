"""
Screen Context Manager for AI Platform Trainer

This module provides a centralized system for managing screen dimensions and
coordinate normalization across the entire game. It ensures all AI systems,
entities, and gameplay elements work consistently regardless of screen resolution.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ScreenDimensions:
    """Container for screen dimension information."""

    width: int
    height: int
    max_dimension: int = 0

    def __post_init__(self):
        self.max_dimension = max(self.width, self.height)


class ScreenContext:
    """
    Centralized screen context manager for resolution-independent game systems.

    This class provides utilities for normalizing coordinates, scaling elements,
    and ensuring consistent behavior across all screen resolutions.
    """

    _instance: Optional["ScreenContext"] = None

    def __init__(self, width: int, height: int):
        """
        Initialize the screen context.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self.dimensions = ScreenDimensions(width, height)
        self._reference_width = 1280  # Reference resolution for scaling
        self._reference_height = 720
        self._scale_factor = self._calculate_scale_factor()

        logging.info(
            f"ScreenContext initialized: {width}x{height} (scale: {self._scale_factor:.2f})"
        )

    @classmethod
    def initialize(cls, width: int, height: int) -> "ScreenContext":
        """
        Initialize the global screen context instance.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels

        Returns:
            The initialized ScreenContext instance
        """
        cls._instance = cls(width, height)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ScreenContext":
        """
        Get the global screen context instance.

        Returns:
            The current ScreenContext instance

        Raises:
            RuntimeError: If ScreenContext hasn't been initialized
        """
        if cls._instance is None:
            raise RuntimeError("ScreenContext must be initialized before use")
        return cls._instance

    def _calculate_scale_factor(self) -> float:
        """Calculate scale factor relative to reference resolution."""
        width_scale = self.dimensions.width / self._reference_width
        height_scale = self.dimensions.height / self._reference_height
        return min(
            width_scale, height_scale
        )  # Use smaller scale to ensure content fits

    @property
    def width(self) -> int:
        """Get screen width."""
        return self.dimensions.width

    @property
    def height(self) -> int:
        """Get screen height."""
        return self.dimensions.height

    def normalize_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Normalize pixel coordinates to [0,1] range.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            Tuple of (normalized_x, normalized_y) in range [0,1]
        """
        norm_x = x / self.dimensions.width if self.dimensions.width > 0 else 0.0
        norm_y = y / self.dimensions.height if self.dimensions.height > 0 else 0.0
        return norm_x, norm_y

    def create_missile_observation(
        self,
        player_pos: Dict[str, float],
        target_pos: Dict[str, float],
        missile_pos: Dict[str, float],
        missile_velocity: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Create a resolution-independent observation for missile AI.

        Args:
            player_pos: Player position dict with x, y keys
            target_pos: Target position dict with x, y keys
            missile_pos: Missile position dict with x, y keys
            missile_velocity: Missile velocity dict with x, y keys

        Returns:
            Normalized observation dict for missile AI
        """
        # Normalize all positions
        norm_player_x, norm_player_y = self.normalize_position(
            player_pos["x"], player_pos["y"]
        )
        norm_target_x, norm_target_y = self.normalize_position(
            target_pos["x"], target_pos["y"]
        )
        norm_missile_x, norm_missile_y = self.normalize_position(
            missile_pos["x"], missile_pos["y"]
        )

        # Normalize velocities relative to a reference speed
        reference_speed = 10.0  # Reference speed for normalization
        norm_vel_x = missile_velocity["x"] / reference_speed
        norm_vel_y = missile_velocity["y"] / reference_speed

        # Calculate normalized distances
        dx_to_target = norm_target_x - norm_missile_x
        dy_to_target = norm_target_y - norm_missile_y
        distance_to_target = (dx_to_target**2 + dy_to_target**2) ** 0.5

        return {
            "player_x": norm_player_x,
            "player_y": norm_player_y,
            "target_x": norm_target_x,
            "target_y": norm_target_y,
            "missile_x": norm_missile_x,
            "missile_y": norm_missile_y,
            "velocity_x": norm_vel_x,
            "velocity_y": norm_vel_y,
            "distance_to_target": distance_to_target,
        }

    def __str__(self) -> str:
        """String representation of screen context."""
        return (
            f"ScreenContext({self.dimensions.width}x{self.dimensions.height}, "
            f"scale={self._scale_factor:.2f})"
        )
