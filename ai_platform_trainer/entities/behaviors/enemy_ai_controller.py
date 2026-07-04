"""
Enemy AI Controller for AI Platform Trainer.

This module handles the enemy's AI-driven movement using either
the trained neural network model or reinforcement learning policy.
"""
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple

import torch


class EnemyAIController:
    """
    Controller for enemy AI behavior.

    This class manages the enemy movement logic, providing different strategies
    including neural network inference, reinforcement learning policy, and
    fallback behaviors to prevent freezing.
    """

    def __init__(self):
        """Initialize the enemy AI controller."""
        self.last_action_time = time.time()
        # Reduced from 0.05 to 0.01 (10ms between actions)
        self.action_interval = 0.01
        # Reduced from 0.7 to 0.4 for more responsive movement
        self.smoothing_factor = 0.4
        self.prev_dx = 0
        self.prev_dy = 0
        self.stuck_counter = 0
        self.prev_positions: List[Dict[str, float]] = []
        self.max_positions = 10  # Store last 10 positions to detect being stuck

        # Missile avoidance parameters
        self.missile_detection_radius = 150.0  # How far to detect missiles
        self.missile_danger_radius = 80.0  # When to start emergency evasion
        self.evasion_strength = 1.5  # How strongly to evade (multiplier)
        self.prediction_time = 10  # How many frames to predict missile movement

        # RL model path
        self.rl_model_path = "models/enemy_rl/final_model.zip"
        self._rl_model = None  # Lazy-loaded

    @property
    def rl_model(self):
        """Lazy-load the RL model if available."""
        if self._rl_model is None and os.path.exists(self.rl_model_path):
            try:
                # Only import these when needed to avoid circular imports
                from stable_baselines3 import PPO

                self._rl_model = PPO.load(self.rl_model_path)
                logging.info(f"Successfully loaded RL model from {self.rl_model_path}")
            except Exception as e:
                logging.error(f"Failed to load RL model: {e}")
        return self._rl_model

    def _detect_missiles(self, enemy, player) -> List[Dict]:
        """
        Detect player missiles in the vicinity of the enemy.

        Args:
            enemy: Enemy instance
            player: Player instance with missiles attribute

        Returns:
            List of missile information dicts with positions and velocities
        """
        nearby_missiles: List[Dict[str, Any]] = []

        # Check if player has missiles attribute and it's not empty
        if not hasattr(player, "missiles") or not player.missiles:
            return nearby_missiles

        enemy_x, enemy_y = enemy.pos["x"], enemy.pos["y"]

        for missile in player.missiles:
            missile_x, missile_y = missile.pos["x"], missile.pos["y"]

            # Calculate distance to missile
            dx = missile_x - enemy_x
            dy = missile_y - enemy_y
            distance = math.sqrt(dx * dx + dy * dy)

            # Check if the missile is within detection range
            if distance <= self.missile_detection_radius:
                # Predict future position based on velocity
                future_x = missile_x + (self.prediction_time * missile.vx)
                future_y = missile_y + (self.prediction_time * missile.vy)

                # Calculate future distance
                future_dx = future_x - enemy_x
                future_dy = future_y - enemy_y
                # Calculate the squared distance
                # Split calculation to avoid long line
                future_dx_squared = future_dx * future_dx
                future_dy_squared = future_dy * future_dy
                future_dist_squared = future_dx_squared + future_dy_squared
                # Get actual distance
                # by taking the square root
                future_distance = math.sqrt(future_dist_squared)

                # Create the missile info dictionary
                missile_info = {
                    "missile": missile,
                    "distance": distance,
                    "future_distance": future_distance,
                    "dx": dx,
                    "dy": dy,
                    "future_dx": future_dx,
                    "future_dy": future_dy,
                    "vx": missile.vx,
                    "vy": missile.vy,
                }
                nearby_missiles.append(missile_info)

        return nearby_missiles

    def _calculate_evasion_vector(
        self, enemy_pos: Dict[str, float], missiles: List[Dict]
    ) -> Tuple[float, float]:
        """
        Calculate optimal evasion vector based on nearby missiles.

        Args:
            enemy_pos: Enemy position dictionary
            missiles: List of detected missile information

        Returns:
            (dx, dy) evasion direction tuple
        """
        if not missiles:
            return 0, 0

        evasion_x, evasion_y = 0, 0

        for missile_info in missiles:
            # Calculate danger level - higher for closer missiles
            danger = 1.0
            if missile_info["distance"] < self.missile_danger_radius:
                # Exponential increase in danger as missiles get very close
                max_dist = max(missile_info["distance"], 10)
                danger = self.evasion_strength * (self.missile_danger_radius / max_dist)

            # Strongest evasion from predicted future position
            evade_dx = -missile_info[
                "future_dx"
            ]  # Move away from missile's future position
            evade_dy = -missile_info["future_dy"]

            # Normalize evasion vector
            evade_magnitude = math.sqrt(evade_dx**2 + evade_dy**2)
            if evade_magnitude > 0:
                evade_dx /= evade_magnitude
                evade_dy /= evade_magnitude

                # Weight by danger level and add to cumulative evasion
                evasion_x += evade_dx * danger
                evasion_y += evade_dy * danger

        # Normalize the final evasion vector
        return self._normalize_vector(evasion_x, evasion_y)

    def _blend_behaviors(
        self,
        chase_vector: Tuple[float, float],
        evasion_vector: Tuple[float, float],
        evasion_weight: float,
    ) -> Tuple[float, float]:
        """
        Blend chasing behavior with evasion behavior.

        Args:
            chase_vector: Original movement vector towards player
            evasion_vector: Vector for evading missiles
            evasion_weight: How strongly to weight evasion (0-1)

        Returns:
            Blended direction vector
        """
        # No evasion means no change
        if evasion_weight == 0 or (evasion_vector[0] == 0 and evasion_vector[1] == 0):
            return chase_vector

        # Full evasion means use only evasion vector
        if evasion_weight >= 1.0:
            return evasion_vector

        # Blend the two behaviors
        blend_x = (
            chase_vector[0] * (1 - evasion_weight) + evasion_vector[0] * evasion_weight
        )
        blend_y = (
            chase_vector[1] * (1 - evasion_weight) + evasion_vector[1] * evasion_weight
        )

        # Normalize the result
        return self._normalize_vector(blend_x, blend_y)

    def update_enemy_movement(
        self,
        enemy,
        player_x: float,
        player_y: float,
        player_speed: float,
        current_time: int,
    ) -> None:
        """
        Handle the enemy's AI-driven movement.

        Args:
            enemy: EnemyPlay instance
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time
        """
        # If the enemy is not visible, skip
        if not enemy.visible:
            return

        # Throttle updates for better performance
        current_time_sec = time.time()
        if current_time_sec - self.last_action_time < self.action_interval:
            return
        self.last_action_time = current_time_sec

        # Track position history to detect if enemy is stuck
        self._update_position_history(enemy.pos)

        # Get base movement direction (not accounting for missiles)
        if self.rl_model is not None:
            action_dx, action_dy = self._get_rl_action(enemy, player_x, player_y)
        else:
            action_dx, action_dy = self._get_nn_action(enemy, player_x, player_y)

        # Check if enemy is stuck and apply special behavior if needed
        if self._is_enemy_stuck():
            action_dx, action_dy = self._handle_stuck_enemy(
                player_x, player_y, enemy.pos
            )

        # Get player object from enemy's game reference (if available)
        player = None
        if hasattr(enemy, "game") and hasattr(enemy.game, "player"):
            player = enemy.game.player

        # Detect missiles and calculate evasion if player is available
        evasion_dx, evasion_dy = 0.0, 0.0
        evasion_weight = 0.0

        if player:
            # Detect nearby missiles
            nearby_missiles = self._detect_missiles(enemy, player)

            # If missiles detected, calculate evasion vector
            if nearby_missiles:
                evasion_dx, evasion_dy = self._calculate_evasion_vector(
                    enemy.pos, nearby_missiles
                )

                # Calculate evasion weight based on closest missile
                closest_missile = min(nearby_missiles, key=lambda m: m["distance"])
                closest_distance = closest_missile["distance"]

                # Closer missiles mean stronger evasion weight
                if closest_distance < self.missile_danger_radius:
                    evasion_weight = (
                        min(1.0, self.missile_danger_radius / closest_distance) * 0.8
                    )
                else:
                    # Gradual increase in evasion weight as missiles get closer
                    ratio = closest_distance / self.missile_detection_radius
                    evasion_weight = max(0, 0.5 * (1 - ratio))

                logging.debug(
                    f"Missile detected! Distance: {closest_distance:.1f}, "
                    f"Evasion weight: {evasion_weight:.2f}"
                )

            # Blend chasing and evasion behaviors
            action_dx, action_dy = self._blend_behaviors(
                (action_dx, action_dy), (evasion_dx, evasion_dy), evasion_weight
            )

        # Apply smoothing
        action_dx = (
            self.smoothing_factor * action_dx
            + (1 - self.smoothing_factor) * self.prev_dx
        )
        action_dy = (
            self.smoothing_factor * action_dy
            + (1 - self.smoothing_factor) * self.prev_dy
        )

        # Normalize direction vector
        action_dx, action_dy = self._normalize_vector(action_dx, action_dy)

        # Store for next frame's smoothing
        self.prev_dx, self.prev_dy = action_dx, action_dy

        # Move enemy at 100% of the player's speed (increased from 70%)
        speed = player_speed * 1.0
        enemy.pos["x"] += action_dx * speed
        enemy.pos["y"] += action_dy * speed

        # Wrap-around logic
        enemy.pos["x"], enemy.pos["y"] = enemy.wrap_position(
            enemy.pos["x"], enemy.pos["y"]
        )

    def _get_nn_action(
        self, enemy, player_x: float, player_y: float
    ) -> Tuple[float, float]:
        """
        Get action from neural network model.

        Args:
            enemy: Enemy instance
            player_x: Player's x position
            player_y: Player's y position

        Returns:
            Tuple of (dx, dy) movement direction
        """
        # Construct input state for the model
        dist = math.sqrt(
            (player_x - enemy.pos["x"]) ** 2 + (player_y - enemy.pos["y"]) ** 2
        )
        state = torch.tensor(
            [[player_x, player_y, enemy.pos["x"], enemy.pos["y"], dist]],
            dtype=torch.float32,
        )

        # Inference
        try:
            with torch.no_grad():
                action = enemy.model(state)  # shape: [1, 2]

            action_dx, action_dy = action[0].tolist()

            # If action is very small, treat as zero
            if abs(action_dx) < 1e-6 and abs(action_dy) < 1e-6:
                return self._get_random_direction()

            return action_dx, action_dy

        except Exception as e:
            logging.error(f"Neural network inference error: {e}")
            return self._get_random_direction()

    def _get_rl_action(
        self, enemy, player_x: float, player_y: float
    ) -> Tuple[float, float]:
        """
        Get action from reinforcement learning model.

        Args:
            enemy: Enemy instance
            player_x: Player's x position
            player_y: Player's y position

        Returns:
            Tuple of (dx, dy) movement direction
        """
        try:
            # Normalize positions to [0,1] range
            screen_width, screen_height = 800, 600  # Default game screen size

            # Create observation
            obs = [
                enemy.pos["x"] / screen_width,
                enemy.pos["y"] / screen_height,
                player_x / screen_width,
                player_y / screen_height,
            ]

            # Get action from RL model
            action, _ = self.rl_model.predict(obs, deterministic=False)

            # Convert continuous action space to direction
            action_dx, action_dy = action[0], action[1]

            # Ensure the action isn't zero
            if abs(action_dx) < 1e-6 and abs(action_dy) < 1e-6:
                return self._get_random_direction()

            return action_dx, action_dy

        except Exception as e:
            logging.error(f"RL model inference error: {e}")
            return self._get_random_direction()

    def _normalize_vector(self, dx: float, dy: float) -> Tuple[float, float]:
        """
        Normalize a direction vector to unit length.

        Args:
            dx: X component of direction
            dy: Y component of direction

        Returns:
            Normalized (dx, dy) tuple
        """
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            return dx / length, dy / length
        else:
            # Apply a random direction if vector is zero length
            return self._get_random_direction()

    def _get_random_direction(self) -> Tuple[float, float]:
        """
        Get a random unit direction vector.

        Returns:
            Random (dx, dy) direction
        """
        angle = random.uniform(0, 2 * math.pi)
        return math.cos(angle), math.sin(angle)

    def _update_position_history(self, position: Dict[str, float]) -> None:
        """
        Update the history of enemy positions to detect if stuck.

        Args:
            position: Current enemy position
        """
        # Add current position to history
        self.prev_positions.append({"x": position["x"], "y": position["y"]})

        # Limit history size
        if len(self.prev_positions) > self.max_positions:
            self.prev_positions.pop(0)

    def _is_enemy_stuck(self) -> bool:
        """
        Check if the enemy appears to be stuck based on position history.

        Returns:
            True if enemy seems stuck, False otherwise
        """
        if len(self.prev_positions) < self.max_positions:
            return False

        # Calculate variance in positions
        x_positions = [pos["x"] for pos in self.prev_positions]
        y_positions = [pos["y"] for pos in self.prev_positions]

        x_var = max(x_positions) - min(x_positions)
        y_var = max(y_positions) - min(y_positions)

        # If the enemy hasn't moved much, it might be stuck
        if x_var < 10 and y_var < 10:
            self.stuck_counter += 1
            if self.stuck_counter > 3:  # Stuck for several frames
                return True
        else:
            self.stuck_counter = 0

        return False

    def _handle_stuck_enemy(
        self, player_x: float, player_y: float, enemy_pos: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Special behavior for when the enemy is detected as stuck.

        Args:
            player_x: Player's x position
            player_y: Player's y position
            enemy_pos: Enemy's current position

        Returns:
            Direction vector to move the enemy
        """
        logging.debug(
            f"Enemy detected as stuck at {enemy_pos}, applying escape behavior"
        )

        # Option 1: Move away from player (reversed chase)
        dx = enemy_pos["x"] - player_x
        dy = enemy_pos["y"] - player_y

        # Option 2: Sometimes use completely random movement to break patterns
        if random.random() < 0.3:
            return self._get_random_direction()

        # Normalize the escape vector
        return self._normalize_vector(dx, dy)


# Initialize the controller as a singleton
enemy_controller = EnemyAIController()


def update_enemy_movement(
    enemy, player_x: float, player_y: float, player_speed: float, current_time: int
) -> None:
    """
    Legacy function for backward compatibility.
    Delegates to the EnemyAIController instance.

    Args:
        enemy: EnemyPlay instance
        player_x: Player's x position
        player_y: Player's y position
        player_speed: Player's movement speed
        current_time: Current game time
    """
    enemy_controller.update_enemy_movement(
        enemy, player_x, player_y, player_speed, current_time
    )
