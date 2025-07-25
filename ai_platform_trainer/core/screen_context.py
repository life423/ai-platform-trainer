"""
Screen Context Manager for AI Platform Trainer

This module provides a centralized system for managing screen dimensions and
coordinate normalization across the entire game. It ensures all AI systems,
entities, and gameplay elements work consistently regardless of screen resolution.
"""
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class ScreenDimensions:
    """Container for screen dimension information."""
    width: int
    height: int
    max_dimension: int = 0
    min_dimension: int = 0
    aspect_ratio: float = 1.0
    
    def __post_init__(self):
        self.max_dimension = max(self.width, self.height)
        self.min_dimension = min(self.width, self.height)
        self.aspect_ratio = self.width / self.height if self.height > 0 else 1.0


class ScreenContext:
    """
    Centralized screen context manager for resolution-independent game systems.
    
    This class provides utilities for normalizing coordinates, scaling elements,
    and ensuring consistent behavior across all screen resolutions.
    """
    
    _instance: Optional['ScreenContext'] = None
    
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
        
        logging.info(f"ScreenContext initialized: {width}x{height} (scale: {self._scale_factor:.2f})")
    
    @classmethod
    def initialize(cls, width: int, height: int) -> 'ScreenContext':
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
    def get_instance(cls) -> 'ScreenContext':
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
    
    @classmethod
    def update_dimensions(cls, width: int, height: int) -> None:
        """
        Update the screen dimensions (e.g., after fullscreen toggle).
        
        Args:
            width: New screen width in pixels
            height: New screen height in pixels
        """
        if cls._instance is None:
            cls.initialize(width, height)
        else:
            cls._instance.dimensions = ScreenDimensions(width, height)
            cls._instance._scale_factor = cls._instance._calculate_scale_factor()
            logging.info(f"ScreenContext updated: {width}x{height} (scale: {cls._instance._scale_factor:.2f})")
    
    def _calculate_scale_factor(self) -> float:
        """Calculate scale factor relative to reference resolution."""
        width_scale = self.dimensions.width / self._reference_width
        height_scale = self.dimensions.height / self._reference_height
        return min(width_scale, height_scale)  # Use smaller scale to ensure content fits
    
    @property
    def width(self) -> int:
        """Get screen width."""
        return self.dimensions.width
    
    @property
    def height(self) -> int:
        """Get screen height."""
        return self.dimensions.height
    
    @property
    def max_dimension(self) -> int:
        """Get the larger of width or height."""
        return self.dimensions.max_dimension
    
    @property
    def scale_factor(self) -> float:
        """Get the current scale factor relative to reference resolution."""
        return self._scale_factor
    
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
    
    def denormalize_position(self, norm_x: float, norm_y: float) -> Tuple[float, float]:
        """
        Convert normalized coordinates back to pixel coordinates.
        
        Args:
            norm_x: Normalized X coordinate [0,1]
            norm_y: Normalized Y coordinate [0,1]
            
        Returns:
            Tuple of (x, y) in pixel coordinates
        """
        x = norm_x * self.dimensions.width
        y = norm_y * self.dimensions.height
        return x, y
    
    def normalize_distance(self, distance: float) -> float:
        """
        Normalize a distance relative to screen size.
        
        Args:
            distance: Distance in pixels
            
        Returns:
            Normalized distance relative to screen diagonal
        """
        diagonal = (self.dimensions.width ** 2 + self.dimensions.height ** 2) ** 0.5
        return distance / diagonal if diagonal > 0 else 0.0
    
    def denormalize_distance(self, norm_distance: float) -> float:
        """
        Convert normalized distance back to pixels.
        
        Args:
            norm_distance: Normalized distance
            
        Returns:
            Distance in pixels
        """
        diagonal = (self.dimensions.width ** 2 + self.dimensions.height ** 2) ** 0.5
        return norm_distance * diagonal
    
    def scale_value(self, value: float, reference_value: float = 1.0) -> float:
        """
        Scale a value based on the current resolution.
        
        Args:
            value: Value to scale
            reference_value: Reference value at reference resolution
            
        Returns:
            Scaled value
        """
        return value * self._scale_factor * reference_value
    
    def get_ai_observation_dims(self) -> Tuple[int, int]:
        """
        Get standardized dimensions for AI observations.
        
        Returns:
            Tuple of (width, height) for AI training consistency
        """
        return self.dimensions.width, self.dimensions.height
    
    def create_missile_observation(self, player_pos: Dict[str, float], 
                                 target_pos: Dict[str, float],
                                 missile_pos: Dict[str, float],
                                 missile_velocity: Dict[str, float]) -> Dict[str, float]:
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
        norm_player_x, norm_player_y = self.normalize_position(player_pos["x"], player_pos["y"])
        norm_target_x, norm_target_y = self.normalize_position(target_pos["x"], target_pos["y"])
        norm_missile_x, norm_missile_y = self.normalize_position(missile_pos["x"], missile_pos["y"])
        
        # Normalize velocities relative to a reference speed
        reference_speed = 10.0  # Reference speed for normalization
        norm_vel_x = missile_velocity["x"] / reference_speed
        norm_vel_y = missile_velocity["y"] / reference_speed
        
        # Calculate normalized distances
        dx_to_target = norm_target_x - norm_missile_x
        dy_to_target = norm_target_y - norm_missile_y
        distance_to_target = (dx_to_target ** 2 + dy_to_target ** 2) ** 0.5
        
        return {
            "player_x": norm_player_x,
            "player_y": norm_player_y,
            "target_x": norm_target_x,
            "target_y": norm_target_y,
            "missile_x": norm_missile_x,
            "missile_y": norm_missile_y,
            "velocity_x": norm_vel_x,
            "velocity_y": norm_vel_y,
            "distance_to_target": distance_to_target
        }
    
    def create_enemy_observation(self, player_pos: Dict[str, float],
                               enemy_pos: Dict[str, float],
                               player_speed: float,
                               additional_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Create a resolution-independent observation for enemy AI.
        
        Args:
            player_pos: Player position dict with x, y keys
            enemy_pos: Enemy position dict with x, y keys
            player_speed: Player movement speed
            additional_data: Optional additional observation data
            
        Returns:
            Normalized observation dict for enemy AI
        """
        # Normalize positions
        norm_player_x, norm_player_y = self.normalize_position(player_pos["x"], player_pos["y"])
        norm_enemy_x, norm_enemy_y = self.normalize_position(enemy_pos["x"], enemy_pos["y"])
        
        # Calculate normalized distance
        dx = norm_player_x - norm_enemy_x
        dy = norm_player_y - norm_enemy_y
        distance = (dx ** 2 + dy ** 2) ** 0.5
        
        # Normalize speed
        reference_speed = 10.0
        norm_speed = player_speed / reference_speed
        
        observation = {
            "player_x": norm_player_x,
            "player_y": norm_player_y,
            "enemy_x": norm_enemy_x,
            "enemy_y": norm_enemy_y,
            "distance": distance,
            "player_speed": norm_speed
        }
        
        # Add any additional normalized data
        if additional_data:
            observation.update(additional_data)
        
        return observation
    
    def get_bounds_for_entity(self, entity_size: int) -> Dict[str, float]:
        """
        Get movement bounds for an entity of given size.
        
        Args:
            entity_size: Size of the entity in pixels
            
        Returns:
            Dict with min_x, max_x, min_y, max_y bounds
        """
        return {
            "min_x": 0.0,
            "max_x": float(self.dimensions.width - entity_size),
            "min_y": 0.0, 
            "max_y": float(self.dimensions.height - entity_size)
        }
    
    def clamp_position(self, x: float, y: float, entity_size: int) -> Tuple[float, float]:
        """
        Clamp a position to stay within screen bounds.
        
        Args:
            x: X position
            y: Y position
            entity_size: Size of the entity
            
        Returns:
            Tuple of (clamped_x, clamped_y)
        """
        bounds = self.get_bounds_for_entity(entity_size)
        clamped_x = max(bounds["min_x"], min(bounds["max_x"], x))
        clamped_y = max(bounds["min_y"], min(bounds["max_y"], y))
        return clamped_x, clamped_y
    
    def __str__(self) -> str:
        """String representation of screen context."""
        return f"ScreenContext({self.dimensions.width}x{self.dimensions.height}, scale={self._scale_factor:.2f})"