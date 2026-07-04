import math
import random
from typing import Optional, Tuple

from ai_platform_trainer.core.config_manager import get_config_manager

# Get the ConfigManager instance
config_manager = get_config_manager()


def find_valid_spawn_position(
    screen_width: int,
    screen_height: int,
    entity_size: int,
    margin: int = None,
    min_dist: int = None,
    other_pos: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int]:
    """
    Returns a random (x, y) within the screen boundaries, ensuring optional
    minimum distance from another position.
    """
    # Use the config values if not provided
    if margin is None:
        margin = config_manager.get("gameplay.wall_margin", 50)
    if min_dist is None:
        min_dist = config_manager.get("gameplay.min_distance", 100)
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
                return (x, y)
        else:
            return (x, y)


def find_enemy_spawn_position(
    screen_width: int,
    screen_height: int,
    enemy_size: int,
    player_pos: Tuple[float, float],
) -> Tuple[int, int]:
    """
    Specialized helper for spawning an enemy, ensuring min distance from the player.
    """
    wall_margin = config_manager.get("gameplay.wall_margin", 50)
    min_distance = config_manager.get("gameplay.min_distance", 100)

    return find_valid_spawn_position(
        screen_width=screen_width,
        screen_height=screen_height,
        entity_size=enemy_size,
        margin=wall_margin,
        min_dist=min_distance,
        other_pos=player_pos,
    )
