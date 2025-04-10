import math
import torch
from typing import Tuple

def update_enemy_movement(
    enemy,
    player_x: float,
    player_y: float,
    player_speed: float,
    current_time: int
) -> None:
    """
    Handle the enemy's AI-driven movement.
    - 'enemy' is an EnemyPlay instance (so we can access enemy.pos, enemy.model, etc.).
    - 'player_x', 'player_y' is the player's position.
    - 'player_speed' is how fast the player is moving (used to scale enemy speed).
    - 'current_time' might be needed for more advanced logic, but we keep it for parity.
    """

    # If the enemy is not visible, skip
    if not enemy.visible:
        return

    # Construct input state for the model
    dist = math.sqrt((player_x - enemy.pos["x"])**2 + (player_y - enemy.pos["y"])**2)
    state = torch.tensor(
        [[player_x, player_y, enemy.pos["x"], enemy.pos["y"], dist]],
        dtype=torch.float32
    )

    # Inference
    with torch.no_grad():
        action = enemy.model(state)  # shape: [1, 2]

    action_dx, action_dy = action[0].tolist()

    # Normalize
    action_len = math.sqrt(action_dx**2 + action_dy**2)
    if action_len > 0:
        action_dx /= action_len
        action_dy /= action_len
    else:
        action_dx, action_dy = 0.0, 0.0

    # Move enemy at 70% of the player's speed
    speed = player_speed * 0.7
    enemy.pos["x"] += action_dx * speed
    enemy.pos["y"] += action_dy * speed

    # Reintroduce wrap-around logic
    enemy.pos["x"], enemy.pos["y"] = enemy.wrap_position(enemy.pos["x"], enemy.pos["y"])