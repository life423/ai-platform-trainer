import math
import random
from typing import Optional, Tuple

from ai_platform_trainer.gameplay.config import config

def compute_normalized_direction(
    px: float, py: float, ex: float, ey: float
) -> Tuple[float, float]:
    direction_x = px - ex
    direction_y = py - ey
    dist = math.hypot(direction_x, direction_y)
    if dist > 0:
        return direction_x / dist, direction_y / dist
    else:
        return 0.0, 0.0

def find_valid_spawn_position(
    screen_width: int,
    screen_height: int,
    entity_size: int,
    margin: int = config.WALL_MARGIN,
    min_dist: int = config.MIN_DISTANCE,
    other_pos: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    x_min = margin
    x_max = screen_width - entity_size - margin
    y_min = margin
    y_max = screen_height - entity_size - margin

    while True:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)

        if other_pos:
            dist = math.hypot(x - other_pos[0], y - other_pos[1])
            if dist >= min_dist:
                return x, y
        else:
            return x, y

def find_enemy_spawn_position(
    screen_width: int,
    screen_height: int,
    enemy_size: int,
    player_pos: Tuple[float, float],
) -> Tuple[int, int]:
    return find_valid_spawn_position(
        screen_width=screen_width,
        screen_height=screen_height,
        entity_size=enemy_size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=player_pos,
    )
    
def compute_normalized_direction(
    px: float, py: float, ex: float, ey: float
) -> Tuple[float, float]:
    """
    Returns the normalized (dx, dy) direction from enemy to player (or vice versa),
    or (0,0) if both points coincide.
    """
    direction_x = px - ex
    direction_y = py - ey
    dist = math.hypot(direction_x, direction_y)
    if dist > 0:
        return direction_x / dist, direction_y / dist
    else:
        return 0.0, 0.0