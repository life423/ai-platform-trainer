"""
Reinforcement Learning environment for the enemy AI in AI Platform Trainer.

This module defines a custom Gym environment that allows training an RL agent
to control the enemy character using the Proximal Policy Optimization (PPO)
algorithm from Stable Baselines3.
"""
import gym
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
import logging
from typing import Dict, Tuple, List, Optional, Any, Union

from ai_platform_trainer.core.screen_context import ScreenContext

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnemyGameEnv")

class EnemyGameEnv(gymnasium.Env):
    """
    Custom Environment that follows gym interface for training the enemy AI.

    This environment wraps the game state and provides a reinforcement learning
    interface with observations, actions, rewards, and state transitions.
    """
    metadata = {'render_modes': ['human']}

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
        
        # For standalone training without game instance
        if self.game is None:
            self.screen_width = 1280
            self.screen_height = 720
            # Initialize ScreenContext for standalone mode
            ScreenContext.initialize(self.screen_width, self.screen_height)
            self.player_pos = np.array([self.screen_width // 4, self.screen_height // 2], dtype=np.float32)
            self.enemy_pos = np.array([self.screen_width // 2, self.screen_height // 2], dtype=np.float32)
            self.player_speed = 5.0

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
        # If we have a game instance, use it
        if self.game and self.game.enemy and hasattr(self.game.enemy, 'apply_rl_action'):
            self.game.enemy.apply_rl_action(action)
            
            # Allow the game to update
            if hasattr(self.game, 'update_once'):
                self.game.update_once()
                
        # Otherwise, simulate the environment
        else:
            # Scale action to actual movement
            move_x = action[0] * 5.0  # Base speed
            move_y = action[1] * 5.0
            
            # Update enemy position
            self.enemy_pos[0] += move_x
            self.enemy_pos[1] += move_y
            
            # Wrap around screen edges
            self.enemy_pos[0] = self.enemy_pos[0] % self.screen_width
            self.enemy_pos[1] = self.enemy_pos[1] % self.screen_height
            
            # Move player randomly
            random_move = np.random.uniform(-2, 2, size=2).astype(np.float32)
            self.player_pos += random_move
            self.player_pos[0] = self.player_pos[0] % self.screen_width
            self.player_pos[1] = self.player_pos[1] % self.screen_height

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
        else:
            # Reset positions for standalone mode
            self.player_pos = np.array([
                np.random.randint(0, self.screen_width),
                np.random.randint(0, self.screen_height)
            ], dtype=np.float32)
            
            # Place enemy away from player
            while True:
                self.enemy_pos = np.array([
                    np.random.randint(0, self.screen_width),
                    np.random.randint(0, self.screen_height)
                ], dtype=np.float32)
                
                # Calculate distance to player
                dx = self.player_pos[0] - self.enemy_pos[0]
                dy = self.player_pos[1] - self.enemy_pos[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Ensure minimum distance
                if distance >= 200:  # Minimum distance
                    break

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
        if self.game and self.game.player and self.game.enemy:
            current_time = pygame.time.get_ticks()
            time_since_hit = current_time - self.last_hit_time

            # Cap the time since last hit to avoid very large values
            time_since_hit = min(time_since_hit, 10000)

            # Use ScreenContext for normalized observation
            screen_context = ScreenContext.get_instance()
            observation = screen_context.create_enemy_observation(
                self.game.player.position,
                self.game.enemy.pos,
                self.game.player.step,
                {"time_factor": time_since_hit / 10000.0}
            )
            
            obs = np.array([
                observation["player_x"], observation["player_y"],
                observation["enemy_x"], observation["enemy_y"],
                observation["distance"], observation["player_speed"],
                observation.get("time_factor", 0.5)
            ], dtype=np.float32)
        else:
            # Standalone mode - use ScreenContext
            screen_context = ScreenContext.get_instance()
            observation = screen_context.create_enemy_observation(
                {"x": self.player_pos[0], "y": self.player_pos[1]},
                {"x": self.enemy_pos[0], "y": self.enemy_pos[1]},
                self.player_speed,
                {"time_factor": 0.5}
            )
            
            obs = np.array([
                observation["player_x"], observation["player_y"],
                observation["enemy_x"], observation["enemy_y"],
                observation["distance"], observation["player_speed"],
                observation.get("time_factor", 0.5)
            ], dtype=np.float32)

        return obs

    def _get_distance(self) -> float:
        """
        Calculate the distance between player and enemy.

        Returns:
            Euclidean distance between player and enemy
        """
        if self.game and self.game.player and self.game.enemy:
            return np.sqrt(
                (self.game.player.position["x"] - self.game.enemy.pos["x"])**2 +
                (self.game.player.position["y"] - self.game.enemy.pos["y"])**2
            )
        else:
            # Standalone mode
            return np.sqrt(
                (self.player_pos[0] - self.enemy_pos[0])**2 +
                (self.player_pos[1] - self.enemy_pos[1])**2
            )

    def _calculate_reward(self) -> float:
        """
        Calculate the reward based on the current game state.

        Returns:
            The calculated reward value
        """
        # Current distance to player
        current_dist = self._get_distance()

        # Reward for getting closer to the player (or penalty for moving away)
        dist_change = self.last_distance - current_dist
        reward = dist_change * 0.1
        self.last_distance = current_dist

        # Base reward for being close to player (encourages chasing)
        proximity_reward = 10.0 / (current_dist + 1.0)
        reward += proximity_reward * 0.05

        # Big reward for hitting player
        if self.game and self.game.check_collision():
            reward += 10.0
            self.last_hit_time = pygame.time.get_ticks()
            
        # In standalone mode, check for collision
        elif not self.game:
            # Simple collision check (within 50 pixels)
            if current_dist < 50:
                reward += 10.0
                self.reset_needed = True

        # Penalty for being hit by missile
        if self.game and self.game.enemy and not self.game.enemy.visible:
            reward -= 5.0
            self.reset_needed = True

        return reward

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Since the actual rendering is handled by the game, this is a no-op.
        
        Returns:
            None as rendering is handled by the game
        """
        return None

    def close(self) -> None:
        """
        Clean up environment resources.
        """
        pass