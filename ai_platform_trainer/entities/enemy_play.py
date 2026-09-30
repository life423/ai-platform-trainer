"""
Enemy entity for play mode.

This module defines the enemy entity used in play mode, which uses AI models
for movement decisions.
"""
import logging
import math
import os
from typing import List, Optional, Tuple

import pygame
import torch

from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.gameplay.config import config

# Display names for the two selectable enemy behaviors, keyed the same way
# as enemy_choice throughout this module and the menu.
ENEMY_CHOICES = {"adaptive": "Adaptive Staged AI", "trained": "Trained AI"}


def is_trained_enemy_available() -> bool:
    """Whether the supervised movement network has been trained at all."""
    return os.path.exists(config.MODEL_PATH)


def create_enemy_play(screen_width: int, screen_height: int) -> "EnemyPlay":
    """EnemyPlay driven by the legacy supervised movement network."""
    model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
    model.eval()
    return EnemyPlay(screen_width, screen_height, model)


class EnemyPlay:
    """
    Enemy entity for play mode with AI-controlled movement.

    This class represents the enemy in play mode, which can be controlled
    by either a neural network or a reinforcement learning model.
    """

    def __init__(
        self, screen_width: int, screen_height: int, model: EnemyMovementModel
    ) -> None:
        """
        Initialize the enemy entity.

        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            model: Neural network model for enemy movement
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.pos = {"x": float(screen_width // 2), "y": float(screen_height // 2)}
        self.speed = 5.0
        self.model = model
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000  # 1 second fade-in

        # Simple counters so this class can report the same "learning
        # stats" shape PlayLearningMode's UI panel expects, regardless of
        # whether it's hosting AdaptiveStagedEnemyAI or this class.
        self.hits_on_player = 0
        self.times_hit_by_missile = 0

    def update_movement(
        self,
        player_x: float,
        player_y: float,
        player_speed: float,
        current_time: int,
        missiles: Optional[List] = None,
    ) -> None:
        """
        Update the enemy's position based on AI model predictions.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time in milliseconds
            missiles: Active missiles (accepted for interface parity with
                AdaptiveStagedEnemyAI; this class's models don't take
                missile positions as input, so it's unused here).
        """
        if not self.visible:
            return

        if self.model is not None:
            self._update_with_nn(player_x, player_y, player_speed)
        else:
            self._update_with_basic_chase(player_x, player_y)

        # Update fade-in effect if active
        if self.fading_in:
            self.update_fade_in(current_time)

    def _update_with_basic_chase(self, player_x: float, player_y: float) -> None:
        """Direct chase, used only if no model at all was loaded."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]

        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 1:  # Avoid division by zero
            move_x = (dx / distance) * self.speed
            move_y = (dy / distance) * self.speed

            self.pos["x"] += move_x
            self.pos["y"] += move_y

            self._wrap_position()

    def on_hit_player(self) -> None:
        """Called when this enemy successfully hits the player."""
        self.hits_on_player += 1

    def on_hit_by_missile(self) -> None:
        """Called when this enemy is hit by a missile."""
        self.times_hit_by_missile += 1

    def get_difficulty_level(self) -> float:
        """Difficulty value (0.0-1.0) for the shared learning-mode UI panel."""
        return 0.75

    def get_learning_stats(self) -> dict:
        """Stats for the shared learning-mode UI panel (see PlayLearningMode)."""
        return {
            "stage": "Trained NN",
            "difficulty": self.get_difficulty_level(),
            "frames": self.hits_on_player + self.times_hit_by_missile,
            "hits": self.hits_on_player,
            "deaths": self.times_hit_by_missile,
            "speed": self.speed,
        }

    def _update_with_nn(
        self, player_x: float, player_y: float, player_speed: float
    ) -> None:
        """
        Update enemy position using the neural network model.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
        """
        # Use ScreenContext for normalization
        screen_context = ScreenContext.get_instance()
        observation = screen_context.create_enemy_observation(
            {"x": player_x, "y": player_y}, self.pos, player_speed
        )

        # Prepare input tensor for the model using normalized values
        model_input = torch.tensor(
            [
                observation["player_x"],
                observation["player_y"],
                observation["enemy_x"],
                observation["enemy_y"],
                observation["distance"],
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        # Get model prediction
        with torch.no_grad():
            movement = self.model(model_input).squeeze(0)
            logging.debug(
                f"Neural network output: "
                f"[{movement[0].item():.3f}, {movement[1].item():.3f}]"
            )

        # Apply movement (scale from [-1,1] to actual pixels)
        move_x = movement[0].item() * self.speed
        move_y = movement[1].item() * self.speed
        logging.debug(f"Scaled movement: dx={move_x:.3f}, dy={move_y:.3f}")

        # Debug: Log enemy movement
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            logging.debug(
                f"Enemy moving: dx={move_x:.2f}, dy={move_y:.2f}, "
                f"toward player at ({player_x:.0f},{player_y:.0f})"
            )

        # Update position
        old_x, old_y = self.pos["x"], self.pos["y"]
        self.pos["x"] += move_x
        self.pos["y"] += move_y

        # Debug: Log position change
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            logging.debug(
                f"Enemy position: ({old_x:.0f},{old_y:.0f}) -> "
                f"({self.pos['x']:.0f},{self.pos['y']:.0f})"
            )

        # Wrap around screen edges
        self._wrap_position()

    def _wrap_position(self) -> None:
        """Wrap the enemy position around screen edges."""
        self.pos["x"], self.pos["y"] = self.wrap_position(self.pos["x"], self.pos["y"])

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Wrap the given coordinates around screen edges.

        Args:
            x: X position to wrap
            y: Y position to wrap

        Returns:
            Tuple of wrapped (x, y) coordinates
        """
        if x < -self.size:
            x = self.screen_width
        elif x > self.screen_width:
            x = -self.size

        if y < -self.size:
            y = self.screen_height
        elif y > self.screen_height:
            y = -self.size

        return x, y

    def set_position(self, x: float, y: float) -> None:
        """
        Set the enemy position.

        Args:
            x: New x position
            y: New y position
        """
        self.pos["x"] = x
        self.pos["y"] = y

    def hide(self) -> None:
        """Hide the enemy (e.g., after being hit)."""
        self.visible = False

    def show(self, current_time: int) -> None:
        """
        Show the enemy with a fade-in effect.

        Args:
            current_time: Current game time in milliseconds
        """
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time

    def update_fade_in(self, current_time: int) -> None:
        """
        Update the fade-in effect.

        Args:
            current_time: Current game time in milliseconds
        """
        if not self.fading_in:
            return

        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)

        self.fade_alpha = int(255 * progress)

        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the enemy on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        if not self.visible:
            return

        if self.fading_in:
            # Create a surface with per-pixel alpha
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.fade_alpha)
            pygame.draw.rect(s, color_with_alpha, (0, 0, self.size, self.size))
            screen.blit(s, (self.pos["x"], self.pos["y"]))
        else:
            pygame.draw.rect(
                screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
            )
