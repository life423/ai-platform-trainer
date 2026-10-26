"""
Smart Missile: a homing missile steered by a reinforcement learning model.
"""
import logging
import math
from typing import Any, Dict, Optional

import numpy as np

try:
    from stable_baselines3 import (  # noqa: F401 - import used only to check availability
        PPO,
    )

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

from ai_platform_trainer.ai.model_paths import MissingModelError
from ai_platform_trainer.arena.observations import sac_missile_observation
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.missile import Missile


class SmartMissile(Missile):
    """
    Homing missile guided by a reinforcement learning model (SAC or PPO).

    SAC and PPO were trained with different observation formats and turn-rate
    scales (see _create_sac_observation()/
    _create_ppo_observation() below), so rl_algorithm must be set
    correctly for whichever rl_model is passed in - the two are not
    interchangeable despite both being "an RL model".
    """

    # Degrees per step each RL model's action of 1.0 was actually trained
    # to mean - not interchangeable.
    RL_TURN_SCALES = {"sac": 20.0, "ppo": 15.0}

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
        rl_algorithm: Optional[str] = None,
    ):
        super().__init__(x, y, speed, vx, vy, birth_time, lifespan)

        if rl_model is None:
            raise MissingModelError(
                "A SmartMissile needs a reinforcement learning model (SAC or PPO). "
                "There is no basic-homing fallback."
            )

        if rl_algorithm not in self.RL_TURN_SCALES:
            raise ValueError(f"Unknown RL missile algorithm '{rl_algorithm}'")

        self.rl_model = rl_model
        self.rl_algorithm = rl_algorithm  # "sac" or "ppo"

        # Target tracking
        self.target_pos = {"x": target_x, "y": target_y}
        self.last_target_pos = {"x": target_x, "y": target_y}

        # Homing parameters
        self.prediction_strength = 0.5  # Increased prediction for better interception

        logging.info(f"SmartMissile created with {self.rl_algorithm or 'RL'} AI")

    def update_with_ai(
        self,
        player_pos: Dict[str, float],
        target_pos: Dict[str, float],
    ) -> None:
        """
        Update missile trajectory using AI guidance.

        Args:
            player_pos: Player position for context
            target_pos: Current target position
        """
        if not target_pos:
            logging.warning("update_with_ai called with no target_pos")
            return

        # Update target tracking
        self.last_target_pos = self.target_pos.copy()
        self.target_pos = target_pos.copy()

        self._update_with_rl(player_pos, target_pos)

        # Actually update the position after AI calculations
        super().update()

    def _update_with_rl(
        self, player_pos: Dict[str, float], target_pos: Dict[str, float]
    ) -> None:
        """Update using the RL model (SAC or PPO) this missile was given."""
        # Calculate target velocity
        target_vx = target_pos["x"] - self.last_target_pos["x"]
        target_vy = target_pos["y"] - self.last_target_pos["y"]

        # Each algorithm needs its own observation format and turn-rate
        # scale - they were trained with different conventions, not
        # just different weights for the same inputs.
        if self.rl_algorithm == "ppo":
            observation = self._create_ppo_observation(
                target_pos, target_vx, target_vy
            )
        else:
            observation = self._create_sac_observation(
                target_pos, target_vx, target_vy
            )
        turn_scale = self.RL_TURN_SCALES[self.rl_algorithm]

        # Get action from RL model
        action, _ = self.rl_model.predict(observation, deterministic=True)
        turn_rate = action[0] * turn_scale

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

    def _create_ppo_observation(
        self, target_pos: Dict[str, float], target_vx: float, target_vy: float
    ) -> np.ndarray:
        """
        Create the observation vector for the trained PPO missile model.

        This must exactly match MissileRLEnvironment._get_observation() in
        ai/training/train_missile_rl.py: 10 features using
        ScreenContext.create_missile_observation()'s [0, 1] position
        convention (not SAC's [-1, 1] scale) and a distance computed from
        the difference of those normalized positions (not the raw screen
        diagonal SAC uses) - a different, older convention than SAC's,
        not a subset of it.
        """
        screen_context = ScreenContext.get_instance()
        observation = screen_context.create_missile_observation(
            {"x": 0, "y": 0},  # Player pos not needed for this observation
            target_pos,
            self.pos,
            {"x": self.vx, "y": self.vy},
        )

        target_vx_norm = target_vx / 5.0
        target_vy_norm = target_vy / 5.0

        angle_to_target = math.atan2(
            target_pos["y"] - self.pos["y"], target_pos["x"] - self.pos["x"]
        )
        angle_norm = angle_to_target / math.pi

        return np.array(
            [
                observation["missile_x"],
                observation["missile_y"],
                observation["velocity_x"],
                observation["velocity_y"],
                observation["target_x"],
                observation["target_y"],
                target_vx_norm,
                target_vy_norm,
                observation["distance_to_target"],
                angle_norm,
            ],
            dtype=np.float32,
        )
