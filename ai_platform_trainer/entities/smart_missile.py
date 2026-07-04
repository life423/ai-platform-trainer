"""
Smart Missile with AI homing capabilities.

This module provides an enhanced missile that can use either traditional neural networks
or reinforcement learning models for intelligent homing behavior.
"""
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from stable_baselines3 import (  # noqa: F401 - import used only to check availability
        PPO,
    )

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.missile import Missile


class SmartMissile(Missile):
    """
    Enhanced missile with AI-powered homing capabilities.

    Can use a supervised neural network or an RL model (SAC or PPO) for
    guidance. SAC and PPO were trained with different observation formats
    and turn-rate scales (see _create_sac_observation()/
    _create_ppo_observation() below), so rl_algorithm must be set
    correctly for whichever rl_model is passed in - the two are not
    interchangeable despite both being "an RL model".
    """

    # Degrees per step each RL model's action of 1.0 was actually trained
    # to mean - not interchangeable, and different from the 12.0 used for
    # the supervised/basic-homing turn-rate clamp below.
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
        ai_model: Optional[MissileModel] = None,
        rl_model: Optional[Any] = None,
        rl_algorithm: Optional[str] = None,
        use_rl: bool = False,
    ):
        super().__init__(x, y, speed, vx, vy, birth_time, lifespan)

        # AI components
        self.ai_model = ai_model
        self.rl_model = rl_model
        self.rl_algorithm = rl_algorithm  # "sac" or "ppo"
        self.use_rl = use_rl and STABLE_BASELINES_AVAILABLE and rl_model is not None

        # Target tracking
        self.target_pos = {"x": target_x, "y": target_y}
        self.last_target_pos = {"x": target_x, "y": target_y}

        # Homing parameters
        self.max_turn_rate = 12.0  # Used by supervised NN and basic homing
        self.prediction_strength = 0.5  # Increased prediction for better interception

        # Performance tracking
        self.frames_alive = 0
        self.distance_to_target_history: List[float] = []

        if self.use_rl:
            ai_kind = self.rl_algorithm.upper() if self.rl_algorithm else "RL"
        elif self.ai_model:
            ai_kind = "Neural Network"
        else:
            ai_kind = "Basic"
        logging.info(f"SmartMissile created with {ai_kind} AI")

    def update_with_ai(
        self,
        player_pos: Dict[str, float],
        target_pos: Dict[str, float],
        shared_input_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update missile trajectory using AI guidance.

        Args:
            player_pos: Player position for context
            target_pos: Current target position
            shared_input_tensor: Pre-allocated tensor for efficiency
        """
        if not target_pos:
            logging.warning("update_with_ai called with no target_pos")
            return

        # Update target tracking
        self.last_target_pos = self.target_pos.copy()
        self.target_pos = target_pos.copy()
        self.frames_alive += 1

        # Calculate current distance for performance tracking
        current_distance = self._calculate_distance_to_target()
        self.distance_to_target_history.append(current_distance)

        if self.use_rl and self.rl_model:
            self._update_with_rl(player_pos, target_pos)
        elif self.ai_model:
            self._update_with_neural_network(
                player_pos, target_pos, shared_input_tensor
            )
        else:
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)

        # Actually update the position after AI calculations
        super().update()

    def _update_with_rl(
        self, player_pos: Dict[str, float], target_pos: Dict[str, float]
    ) -> None:
        """Update using the RL model (SAC or PPO) this missile was given."""
        try:
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
            turn_scale = self.RL_TURN_SCALES.get(self.rl_algorithm, self.max_turn_rate)

            # Get action from RL model
            action, _ = self.rl_model.predict(observation, deterministic=True)
            turn_rate = action[0] * turn_scale

            # Apply the turn
            self._apply_turn(turn_rate)

        except Exception as e:
            logging.error(f"Error in RL missile guidance: {e}")
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)

    def _update_with_neural_network(
        self,
        player_pos: Dict[str, float],
        target_pos: Dict[str, float],
        shared_input_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        """Update using traditional neural network model."""
        try:
            # Get screen context for normalization
            screen_context = ScreenContext.get_instance()

            # Create normalized observation
            observation = screen_context.create_missile_observation(
                player_pos, target_pos, self.pos, {"x": self.vx, "y": self.vy}
            )

            current_angle = math.atan2(self.vy, self.vx)

            # Prepare input tensor with normalized values
            if shared_input_tensor is not None:
                input_tensor = shared_input_tensor
                input_tensor[0] = torch.tensor(
                    [
                        observation["player_x"],
                        observation["player_y"],
                        observation["target_x"],
                        observation["target_y"],
                        observation["missile_x"],
                        observation["missile_y"],
                        current_angle,
                        observation["distance_to_target"],
                        0.0,
                    ]
                )
            else:
                input_tensor = torch.tensor(
                    [
                        [
                            observation["player_x"],
                            observation["player_y"],
                            observation["target_x"],
                            observation["target_y"],
                            observation["missile_x"],
                            observation["missile_y"],
                            current_angle,
                            observation["distance_to_target"],
                            0.0,
                        ]
                    ],
                    dtype=torch.float32,
                )

            # Get prediction from neural network
            with torch.no_grad():
                turn_rate = self.ai_model(input_tensor).item()

            # Apply turn rate limits
            turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, turn_rate))
            self._apply_turn(turn_rate)

        except Exception as e:
            logging.error(f"Error in neural network missile guidance: {e}")
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)

    def _update_with_basic_homing(self, target_pos: Dict[str, float]) -> None:
        """Fallback basic homing behavior."""
        # Predict where target will be
        target_vx = target_pos["x"] - self.last_target_pos["x"]
        target_vy = target_pos["y"] - self.last_target_pos["y"]

        predicted_x = target_pos["x"] + target_vx * self.prediction_strength
        predicted_y = target_pos["y"] + target_vy * self.prediction_strength

        # Calculate desired angle
        desired_angle = math.atan2(
            predicted_y - self.pos["y"], predicted_x - self.pos["x"]
        )

        current_angle = math.atan2(self.vy, self.vx)

        # Calculate turn needed
        angle_diff = desired_angle - current_angle

        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Convert to degrees and limit turn rate
        turn_rate = math.degrees(angle_diff)
        turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, turn_rate))

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
        Create the observation vector for the trained SAC missile model.

        This must exactly match MissileSACEnvironment._get_observation() in
        ai/training/train_missile_sac.py (the environment the deployed model
        was actually trained on): 11 features, positions normalized to
        [-1, 1] (not ScreenContext's generic [0, 1] convention), distance
        normalized by raw screen diagonal, plus a relative-angle feature.
        Reusing ScreenContext.create_missile_observation() here would give
        the right shape but the wrong scale on several fields, silently
        feeding the model out-of-distribution inputs instead of raising.
        """
        screen_context = ScreenContext.get_instance()
        width = screen_context.width
        height = screen_context.height
        max_speed = 10.0  # Matches MissileSACEnvironment.max_speed

        # Normalize positions to [-1, 1]
        missile_x_norm = (self.pos["x"] / width) * 2 - 1
        missile_y_norm = (self.pos["y"] / height) * 2 - 1
        target_x_norm = (target_pos["x"] / width) * 2 - 1
        target_y_norm = (target_pos["y"] / height) * 2 - 1

        # Normalize velocities
        missile_vx_norm = self.vx / max_speed
        missile_vy_norm = self.vy / max_speed
        target_vx_norm = target_vx / 5.0
        target_vy_norm = target_vy / 5.0

        # Relative vectors
        dx = target_pos["x"] - self.pos["x"]
        dy = target_pos["y"] - self.pos["y"]
        distance = math.hypot(dx, dy)
        distance_norm = distance / math.hypot(width, height)

        angle_to_target = math.atan2(dy, dx)
        angle_to_target_norm = angle_to_target / math.pi

        # Relative angle: missile heading vs. direction to target
        missile_angle = math.atan2(self.vy, self.vx)
        relative_angle = angle_to_target - missile_angle
        relative_angle_norm = (
            math.atan2(math.sin(relative_angle), math.cos(relative_angle)) / math.pi
        )

        return np.array(
            [
                missile_x_norm,
                missile_y_norm,
                missile_vx_norm,
                missile_vy_norm,
                target_x_norm,
                target_y_norm,
                target_vx_norm,
                target_vy_norm,
                distance_norm,
                angle_to_target_norm,
                relative_angle_norm,
            ],
            dtype=np.float32,
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

    def _calculate_distance_to_target(self) -> float:
        """Calculate current distance to target."""
        dx = self.pos["x"] - self.target_pos["x"]
        dy = self.pos["y"] - self.target_pos["y"]
        return math.sqrt(dx * dx + dy * dy)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this missile."""
        if not self.distance_to_target_history:
            return {"avg_distance": 0.0, "min_distance": 0.0, "improvement": 0.0}

        avg_distance = sum(self.distance_to_target_history) / len(
            self.distance_to_target_history
        )
        min_distance = min(self.distance_to_target_history)

        # Calculate improvement (negative means getting closer)
        if len(self.distance_to_target_history) > 1:
            initial_distance = self.distance_to_target_history[0]
            final_distance = self.distance_to_target_history[-1]
            improvement = (initial_distance - final_distance) / initial_distance
        else:
            improvement = 0.0

        return {
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "improvement": improvement,
            "frames_alive": self.frames_alive,
        }

    def is_effective(self) -> bool:
        """Check if missile is performing well."""
        if self.frames_alive < 20:  # Need some data
            return True

        stats = self.get_performance_stats()

        # Check if missile is stuck in a circle (getting farther from target)
        if stats["improvement"] < -0.3:  # Getting significantly farther
            return False

        # Check if missile is getting closer over time
        if len(self.distance_to_target_history) > 30:
            recent_avg = sum(self.distance_to_target_history[-10:]) / 10
            earlier_avg = sum(self.distance_to_target_history[-30:-20]) / 10
            if recent_avg >= earlier_avg:  # Not improving
                return False

        return True
