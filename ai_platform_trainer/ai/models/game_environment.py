"""
Gym-compatible environment for reinforcement learning training.

This module provides a Gym-compatible environment that wraps the game logic
for reinforcement learning training.
"""
import gymnasium as gym
import numpy as np
import pygame

from ai_platform_trainer.core.render_mode import RenderMode
from ai_platform_trainer.gameplay.game_core import GameCore


class GameEnvironment(gym.Env):
    """
    Gym-compatible environment wrapping the game logic.

    This environment provides a standard gym interface for RL training,
    while reusing the existing game logic from GameCore.
    """

    metadata = {"render_modes": ["human", "rgb_array", "none"]}

    def __init__(self, render_mode: str = "none"):
        """
        Initialize the game environment.

        Args:
            render_mode: 'human' for visual rendering, 'none' for headless
        """
        super().__init__()

        # Set up render mode
        self._render_mode = (
            RenderMode.FULL if render_mode == "human" else RenderMode.HEADLESS
        )

        # Create game instance with appropriate render mode
        self.game = GameCore(render_mode=self._render_mode)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation: [player_x, player_y, enemy_x, enemy_y, distance, player_speed, time_factor]
        self.observation_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(7,), dtype=np.float32
        )

        # Initialize tracking variables
        self.last_distance = 0
        self.steps = 0
        self.max_steps = 1000

        # Start game in play mode
        self.game.start_game("play")

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)

        # Reset game state
        if hasattr(self.game, "reset_game_state"):
            self.game.reset_game_state()
            self.game.start_game("play")

        # Reset tracking variables
        self.steps = 0
        self.last_distance = self._get_distance()

        return self._get_observation(), {}

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Array with values between -1 and 1 for enemy movement

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply action to enemy
        if self.game.enemy:
            self.game.enemy.apply_rl_action(action)

        # Update game state (without rendering)
        self.game.update_once()

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps

        # Check for collisions
        if self.game.check_collision():
            reward += 10.0  # Bonus for hitting player
            terminated = True

        # Get observation
        observation = self._get_observation()

        return observation, reward, terminated, truncated, {}

    def render(self):
        """
        Render the environment.

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self._render_mode == RenderMode.FULL and self.game.renderer:
            self.game.renderer.render(
                self.game.menu, self.game.player, self.game.enemy, False
            )
            pygame.display.flip()

            if self.render_mode == "rgb_array":
                # Return the rendered frame as an RGB array
                return np.array(pygame.surfarray.array3d(self.game.screen))
        return None

    def close(self):
        """Clean up resources."""
        if self.game:
            pygame.quit()

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            Numpy array with the current observation
        """
        if not (self.game and self.game.player and self.game.enemy):
            return np.zeros(7, dtype=np.float32)

        # Normalize positions
        px = self.game.player.position["x"] / self.game.screen_width
        py = self.game.player.position["y"] / self.game.screen_height
        ex = self.game.enemy.pos["x"] / self.game.screen_width
        ey = self.game.enemy.pos["y"] / self.game.screen_height

        # Calculate distance
        dist = self._get_distance() / max(
            self.game.screen_width, self.game.screen_height
        )

        # Other features
        player_speed = self.game.player.step / 10.0
        time_factor = 0.5  # Placeholder

        return np.array(
            [px, py, ex, ey, dist, player_speed, time_factor], dtype=np.float32
        )

    def _get_distance(self):
        """
        Calculate distance between player and enemy.

        Returns:
            Euclidean distance between player and enemy
        """
        if not (self.game and self.game.player and self.game.enemy):
            return 1000.0

        return np.sqrt(
            (self.game.player.position["x"] - self.game.enemy.pos["x"]) ** 2
            + (self.game.player.position["y"] - self.game.enemy.pos["y"]) ** 2
        )

    def _calculate_reward(self):
        """
        Calculate reward based on game state.

        Returns:
            Calculated reward value
        """
        # Current distance to player
        current_dist = self._get_distance()

        # Reward for getting closer to player
        dist_change = self.last_distance - current_dist
        reward = dist_change * 0.1
        self.last_distance = current_dist

        # Reward for being close to player
        proximity_reward = 10.0 / (current_dist + 1.0)
        reward += proximity_reward * 0.05

        return reward
