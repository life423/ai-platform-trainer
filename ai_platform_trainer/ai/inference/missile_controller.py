"""
Deterministic missile steering, used to label supervised training data.

Play-time guidance lives in entities/smart_missile.py. This module only
provides the controller that the training data collector steers with.
"""
import math
from typing import Dict

from ai_platform_trainer.entities.missile import Missile


def steer_toward(
    missile: "Missile", enemy_pos: Dict[str, float], max_turn_rate: float = 5.0
) -> float:
    """
    Turn the missile straight at the enemy, by at most max_turn_rate degrees.

    This is the deterministic controller that labels training data. It needs
    no model, so data can be recorded before a supervised model exists.

    Returns:
        The turn that was applied, in degrees.
    """
    current_angle = math.atan2(missile.vy, missile.vx)
    target_angle = math.atan2(
        enemy_pos["y"] - missile.pos["y"], enemy_pos["x"] - missile.pos["x"]
    )

    angle_diff = math.degrees(target_angle - current_angle)
    if angle_diff > 180:
        angle_diff -= 360
    elif angle_diff < -180:
        angle_diff += 360

    turn = max(-max_turn_rate, min(max_turn_rate, angle_diff))
    new_angle = current_angle + math.radians(turn)
    missile.vx = missile.speed * math.cos(new_angle)
    missile.vy = missile.speed * math.sin(new_angle)
    missile.last_action = turn
    return turn
