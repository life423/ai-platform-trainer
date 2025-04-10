import math
import torch
from typing import List, Dict, Optional

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel

def update_missile_ai(
    missiles: List["Missile"],
    player_pos: Dict[str, float],
    enemy_pos: Optional[Dict[str, float]],
    shared_input_tensor: torch.Tensor,
    missile_model: SimpleMissileModel,
    model_blend_factor: float = 0.5,
    max_turn_rate: float = 5.0,
) -> None:
    
    for missile in missiles:
        current_angle = math.atan2(missile.vy, missile.vx)

        if enemy_pos is None:  # Handle case where enemy is not present
            target_angle = current_angle
        else:
            target_angle = math.atan2(enemy_pos["y"] - missile.pos["y"], enemy_pos["x"] - missile.pos["x"])

        px, py = player_pos["x"], player_pos["y"]
        # If no enemy, use projected missile position as a target to keep it moving forward.
        ex, ey = enemy_pos["x"] if enemy_pos else missile.pos["x"] + missile.vx, enemy_pos["y"] if enemy_pos else missile.pos["y"] + missile.vy  
        dist_val = math.hypot(missile.pos['x'] - ex, missile.pos["y"] - ey)

        shared_input_tensor[0] = torch.tensor([px, py, ex, ey, missile.pos["x"], missile.pos["y"], current_angle, dist_val, 0.0]) 

        with torch.no_grad():
            turn_rate = missile_model(shared_input_tensor).item()


        angle_diff = math.degrees(target_angle - current_angle) 
        blended_turn_rate = model_blend_factor * turn_rate + (1 - model_blend_factor) * angle_diff

        constrained_turn_rate = max(-max_turn_rate, min(max_turn_rate, blended_turn_rate))

        new_angle = current_angle + math.radians(constrained_turn_rate)
        missile.vx = missile.speed * math.cos(new_angle)
        missile.vy = missile.speed * math.sin(new_angle)
        missile.last_action = turn_rate 
