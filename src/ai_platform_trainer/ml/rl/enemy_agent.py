"""
Reinforcement Learning environment for the enemy AI in AI Platform Trainer.

This module defines a custom Gym environment that allows training an RL agent
to control the enemy character using the Proximal Policy Optimization (PPO)
algorithm from Stable Baselines3.
"""
import gym
from gym import spaces
import numpy as np
import pygame
import logging
from typing import Dict, Tuple


class EnemyGameEnv(gym.Env):
    """
    Custom Environment that follows gym interface for training the enemy AI.

    This environment wraps the game state and provides a reinforcement learning
    interface with observations, actions, rewards, and state transitions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, game_instance=None):
        """
        Initialize the environment with the game instance.

        Args:
            game_instance: A reference to the main game object
        """
        super().__init__()

        # Define action and observation space
        # Actions: continuous movement in x,y directions (-1 to 1)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        # Observation:
        # [player_x, player_y, enemy_x, enemy_y, distance,
        #  player_speed, time_since_last_hit]
        self.observation_space = spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=(7,),
            dtype=np.float32
        )

        self.game = game_instance
        self.current_state = np.zeros(7, dtype=np.float32)
        self.reset_needed = False
        self.last_hit_time = 0
        self.last_distance = 0
        self.steps_since_reset = 0
        self.max_steps = 1000  # Maximum steps per episode

        logging.info("EnemyGameEnv initialized")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Array with values between -1 and 1 for enemy movement direction

        Returns:
            observation: The new state after the action
            reward: The reward for the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated (e.g., max steps)
            info: Additional information for debugging
        """
        # Apply action to enemy in the game
        if self.game and self.game.enemy and hasattr(self.game.enemy, 'apply_rl_action'):
            self.game.enemy.apply_rl_action(action)
        else:
            logging.warning("Cannot apply RL action - game, enemy, or method not available")

        # Allow the game to update
        if self.game and hasattr(self.game, 'update_once'):
            self.game.update_once()

        # Calculate reward based on game state
        reward = self._calculate_reward()

        # Update the state
        self.current_state = self._get_observation()

        # Check if episode is done
        self.steps_since_reset += 1
        done = self.reset_needed
        truncated = self.steps_since_reset >= self.max_steps

        # Info dictionary for debugging
        info = {}

        return self.current_state, reward, done, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            observation: Initial observation after reset
            info: Additional information
        """
        super().reset(seed=seed)

        self.reset_needed = False
        self.steps_since_reset = 0

        # Reset the game state if needed
        if self.game:
            # Only reset the enemy position, not the entire game
            if hasattr(self.game, 'reset_enemy'):
                self.game.reset_enemy()

        # Get initial observation
        self.current_state = self._get_observation()
        self.last_distance = self._get_distance()

        return self.current_state, {}

    def _get_observation(self) -> np.ndarray:
        """
        Extract the current observation from the game state.

        Returns:
            numpy array with the observation features
        """
        if not self.game or not self.game.player or not self.game.enemy:
            return np.zeros(7, dtype=np.float32)

        current_time = pygame.time.get_ticks()
        time_since_hit = current_time - self.last_hit_time

        # Cap the time since last hit to avoid very large values
        time_since_hit = min(time_since_hit, 10000)

        # Normalize values to help with training stability
        screen_width = self.game.screen_width
        screen_height = self.game.screen_height

        px = self.game.player.position["x"] / screen_width
        py = self.game.player.position["y"] / screen_height
        ex = self.game.enemy.pos["x"] / screen_width
        ey = self.game.enemy.pos["y"] / screen_height

        dist = self._get_distance() / max(screen_width, screen_height)
        player_speed = self.game.player.step / 10.0  # Normalize speed
        time_factor = time_since_hit / 10000.0  # Normalize time

        obs = np.array([
            px, py, ex, ey, dist, player_speed, time_factor
        ], dtype=np.float32)

        return obs

    def _get_distance(self) -> float:
        """
        Calculate the distance between player and enemy.

        Returns:
            Euclidean distance between player and enemy
        """
        if not self.game or not self.game.player or not self.game.enemy:
            return 1000.0

        return np.sqrt(
            (self.game.player.position["x"] - self.game.enemy.pos["x"])**2 +
            (self.game.player.position["y"] - self.game.enemy.pos["y"])**2
        )

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the current game state.

        Returns:
            The calculated reward value
        """
        if not self.game or not self.game.player or not self.game.enemy:
            return 0.0

        reward = 0.0

        # Current distance to player
        current_dist = self._get_distance()

        # Reward for getting closer to the player (or penalty for moving away)
        dist_change = self.last_distance - current_dist
        reward += dist_change * 0.1
        self.last_distance = current_dist

        # Base reward for being close to player (encourages chasing)
        proximity_reward = 10.0 / (current_dist + 1.0)
        reward += proximity_reward * 0.05

        # Big reward for hitting player
        if self.game.check_collision():
            reward += 10.0
            self.last_hit_time = pygame.time.get_ticks()

        # Penalty for being hit by missile
        if self.game.enemy and not self.game.enemy.visible:
            reward -= 5.0
            self.reset_needed = True

        return reward

    def render(self, mode='human'):
        """
        Render the environment.

        Since the actual rendering is handled by the game, this is a no-op.
        """

    def close(self):
        """
        Clean up environment resources.
        """
