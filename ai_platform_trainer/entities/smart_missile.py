"""
Smart Missile: a homing missile steered by the SAC reinforcement learning model.
"""
import logging
import math
from typing import Any, Dict, Optional

import numpy as np

from ai_platform_trainer.ai.model_paths import MissingModelError
from ai_platform_trainer.arena.observations import (
    SAC_TURN_SCALE_DEG,
    sac_missile_observation,
)
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.missile import Missile


class SmartMissile(Missile):
    """
    Homing missile guided by the trained SAC reinforcement learning model.

    The observation format and turn-rate scale are the ones the model was
    trained with, shared with the arena (see arena.observations), so the game
    and the arena steer missiles the same way.
    """

    def __init__(
        self,
        x: int,
        y: int,
        target_x: float = 0.0,
        target_y: float = 0.0,
        speed: float = 8.0,
        vx: float = 8.0,
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 5000,  # Reduced lifespan to prevent endless circling
        rl_model: Optional[Any] = None,
    ):
        super().__init__(x, y, speed, vx, vy, birth_time, lifespan)

        if rl_model is None:
            raise MissingModelError(
                "A SmartMissile needs the SAC reinforcement learning model. "
                "There is no basic-homing fallback."
            )

        self.rl_model = rl_model

        # Target tracking
        self.target_pos = {"x": target_x, "y": target_y}
        self.last_target_pos = {"x": target_x, "y": target_y}

        logging.info("SmartMissile created with SAC AI")

    def update_with_ai(self, target_pos: Dict[str, float]) -> None:
        """
        Update missile trajectory using AI guidance.

        Args:
            target_pos: Current target position
        """
        if not target_pos:
            logging.warning("update_with_ai called with no target_pos")
            return

        # Update target tracking
        self.last_target_pos = self.target_pos.copy()
        self.target_pos = target_pos.copy()

        self._update_with_rl(target_pos)

        # Actually update the position after AI calculations
        super().update()

    def _update_with_rl(self, target_pos: Dict[str, float]) -> None:
        """Update using the SAC model this missile was given."""
        # Calculate target velocity
        target_vx = target_pos["x"] - self.last_target_pos["x"]
        target_vy = target_pos["y"] - self.last_target_pos["y"]

        observation = self._create_sac_observation(target_pos, target_vx, target_vy)

        # Get action from RL model
        action, _ = self.rl_model.predict(observation, deterministic=True)
        turn_rate = action[0] * SAC_TURN_SCALE_DEG

        # Apply the turn
        self._apply_turn(turn_rate)

    def _apply_turn(self, turn_rate_degrees: float) -> None:
        """Apply a turn to the missile."""
        current_angle = math.atan2(self.vy, self.vx)
        new_angle = current_angle + math.radians(turn_rate_degrees)

        # Update velocity components
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)

        # Update direction for rendering
        self.direction = (math.cos(new_angle), math.sin(new_angle))

    def _create_sac_observation(
        self, target_pos: Dict[str, float], target_vx: float, target_vy: float
    ) -> np.ndarray:
        """
        The input for the trained SAC missile model.

        Built by arena.observations.sac_missile_observation(), the same
        function the arena uses, so the game and the arena cannot drift apart.
        Positions are scaled by the screen size, which is fixed at the world
        size (800x600) the model was trained on.
        """
        screen_context = ScreenContext.get_instance()
        return sac_missile_observation(
            self.pos["x"],
            self.pos["y"],
            self.vx,
            self.vy,
            target_pos["x"],
            target_pos["y"],
            target_vx,
            target_vy,
            screen_context.width,
            screen_context.height,
        )
