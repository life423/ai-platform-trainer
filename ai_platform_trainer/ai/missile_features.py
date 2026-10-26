"""
The supervised missile model's input vector.

Play (SmartMissile) and training (TrainingMode) both build their features
here, so the model is trained on the same numbers it sees in the game.
"""

import math
from typing import Dict, List

from ai_platform_trainer.core.screen_context import ScreenContext

FEATURE_COUNT = 9


def missile_features(
    player_pos: Dict[str, float],
    target_pos: Dict[str, float],
    missile_pos: Dict[str, float],
    velocity: Dict[str, float],
) -> List[float]:
    """
    Return the nine model inputs for one missile, in model order.

    Positions and distance are normalized to the screen by ScreenContext.
    The heading is the missile's direction before this frame's turn is
    applied. The last slot is always 0.0: it used to hold the collision flag,
    which is only known after the outcome, so the game can never supply it.
    """
    observation = ScreenContext.get_instance().create_missile_observation(
        player_pos, target_pos, missile_pos, velocity
    )
    heading = math.atan2(velocity["y"], velocity["x"])
    return [
        observation["player_x"],
        observation["player_y"],
        observation["target_x"],
        observation["target_y"],
        observation["missile_x"],
        observation["missile_y"],
        heading,
        observation["distance_to_target"],
        0.0,
    ]
