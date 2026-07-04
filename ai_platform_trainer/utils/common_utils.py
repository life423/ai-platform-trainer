"""
Common utility functions for the AI Platform Trainer.

This module provides utility functions used across the codebase.
"""
import math
import random
from typing import List, Optional, Tuple


def compute_normalized_direction(
    source_x: float, source_y: float, target_x: float, target_y: float
) -> Tuple[float, float]:
    """
    Compute the normalized direction vector from source to target.

    Args:
        source_x: X coordinate of the source
        source_y: Y coordinate of the source
        target_x: X coordinate of the target
        target_y: Y coordinate of the target

    Returns:
        Tuple of (normalized_x, normalized_y) direction vector
    """
    dx = target_x - source_x
    dy = target_y - source_y

    # Calculate distance
    distance = math.sqrt(dx * dx + dy * dy)

    # Avoid division by zero
    if distance < 0.0001:
        return (0.0, 0.0)

    # Normalize
    return (dx / distance, dy / distance)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        x1: X coordinate of the first point
        y1: Y coordinate of the first point
        x2: X coordinate of the second point
        y2: Y coordinate of the second point

    Returns:
        The distance between the points
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)


def wrap_position(
    x: float, y: float, width: int, height: int, obj_size: int = 0
) -> Tuple[float, float]:
    """
    Wrap a position around the screen boundaries.

    Args:
        x: X coordinate
        y: Y coordinate
        width: Screen width
        height: Screen height
        obj_size: Size of the object (default: 0)

    Returns:
        Tuple of wrapped (x, y) coordinates
    """
    wrapped_x = x
    wrapped_y = y

    if x < -obj_size:
        wrapped_x = width
    elif x > width:
        wrapped_x = -obj_size

    if y < -obj_size:
        wrapped_y = height
    elif y > height:
        wrapped_y = -obj_size

    return wrapped_x, wrapped_y


def find_valid_spawn_position(
    screen_width: int,
    screen_height: int,
    obj_size: int,
    min_distance: int = 200,
    existing_positions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Find a valid spawn position that is at least min_distance away from existing positions.

    Args:
        screen_width: Width of the screen
        screen_height: Height of the screen
        obj_size: Size of the object to spawn
        min_distance: Minimum distance from existing positions
        existing_positions: List of existing positions to avoid

    Returns:
        Tuple of (x, y) coordinates for the spawn position
    """
    if existing_positions is None:
        existing_positions = []

    margin = obj_size * 2
    max_attempts = 100

    for _ in range(max_attempts):
        # Generate a random position within screen bounds with margin
        x = random.randint(margin, screen_width - margin - obj_size)
        y = random.randint(margin, screen_height - margin - obj_size)

        # Check if the position is far enough from existing positions
        valid = True
        for pos_x, pos_y in existing_positions:
            if calculate_distance(x, y, pos_x, pos_y) < min_distance:
                valid = False
                break

        if valid:
            return (x, y)

    # If we couldn't find a valid position after max attempts,
    # just return a random position
    return (
        random.randint(margin, screen_width - margin - obj_size),
        random.randint(margin, screen_height - margin - obj_size),
    )


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp a value between min and max.

    Args:
        value: The value to clamp
        min_value: The minimum allowed value
        max_value: The maximum allowed value

    Returns:
        The clamped value
    """
    return max(min_value, min(value, max_value))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between a and b.

    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0.0 to 1.0)

    Returns:
        Interpolated value
    """
    return a + (b - a) * t
