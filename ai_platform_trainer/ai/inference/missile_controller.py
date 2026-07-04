"""
Missile AI Controller module for game missile guidance.

This module provides the AI control logic for in-game missiles,
applying a trained neural network model to determine missile trajectory
adjustments based on current game state.
"""
import logging
import math
from typing import Dict, List, Optional

import torch

from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.entities.missile import Missile


def update_missile_ai(
    missiles: List["Missile"],
    player_pos: Dict[str, float],
    enemy_pos: Optional[Dict[str, float]],
    shared_input_tensor: torch.Tensor,
    missile_model: MissileModel,
    model_blend_factor: float = 0.5,
    max_turn_rate: float = 5.0,
) -> None:
    """
    Update missile trajectories using AI model predictions.

    This function processes all active missiles and adjusts their trajectories
    based on a combination of direct targeting and neural network predictions.

    Args:
        missiles: List of missile objects to update
        player_pos: Dictionary containing player x,y coordinates
        enemy_pos: Dictionary containing enemy x,y coordinates (or None if no enemy)
        shared_input_tensor: Pre-allocated tensor for model input to avoid allocations
        missile_model: Trained neural network model for missile guidance
        model_blend_factor: Weight between model output and direct targeting (0-1)
        max_turn_rate: Maximum degrees per update a missile can turn
    """
    if not missiles:
        return

    if enemy_pos is None:
        logging.debug("No enemy position available for missile guidance")
        return

    for missile in missiles:
        try:
            # Calculate current missile direction
            current_angle = math.atan2(missile.vy, missile.vx)

            # Determine target angle (direct path to enemy or continue current trajectory)
            target_angle = math.atan2(
                enemy_pos["y"] - missile.pos["y"], enemy_pos["x"] - missile.pos["x"]
            )

            # Extract position values for model input
            px, py = player_pos["x"], player_pos["y"]
            ex, ey = enemy_pos["x"], enemy_pos["y"]

            # Calculate distance to target (used as model input feature)
            dist_val = math.hypot(missile.pos["x"] - ex, missile.pos["y"] - ey)

            # Prepare model input tensor with current state
            shared_input_tensor[0] = torch.tensor(
                [
                    px,
                    py,
                    ex,
                    ey,
                    missile.pos["x"],
                    missile.pos["y"],
                    current_angle,
                    dist_val,
                    0.0,
                ]
            )

            # Get AI model prediction (turn rate adjustment)
            with torch.no_grad():
                turn_rate = missile_model(shared_input_tensor).item()

            # Calculate angle difference for direct targeting
            angle_diff = math.degrees(target_angle - current_angle)

            # Normalize angle difference to be between -180 and 180 degrees
            if angle_diff > 180:
                angle_diff -= 360
            elif angle_diff < -180:
                angle_diff += 360

            # Blend between model prediction and direct targeting
            blended_turn_rate = (
                model_blend_factor * turn_rate + (1 - model_blend_factor) * angle_diff
            )

            # Constrain turn rate to prevent unrealistic movement
            constrained_turn_rate = max(
                -max_turn_rate, min(max_turn_rate, blended_turn_rate)
            )

            # Apply the turn and update missile velocity components
            new_angle = current_angle + math.radians(constrained_turn_rate)
            missile.vx = missile.speed * math.cos(new_angle)
            missile.vy = missile.speed * math.sin(new_angle)

            # Store for training data collection. This must be the turn that
            # was actually applied (constrained_turn_rate), not the raw
            # model output - logging the model's own guess as the "correct"
            # label would just train it to imitate itself.
            if hasattr(missile, "last_action"):
                missile.last_action = constrained_turn_rate

        except Exception as e:
            logging.error(f"Error updating missile AI: {e}")
            continue
