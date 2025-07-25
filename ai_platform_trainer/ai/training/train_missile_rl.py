"""
Reinforcement Learning Training for Missile AI

This module provides RL training for missile homing behavior using PPO.
The missile learns to effectively chase and hit moving targets.
"""
import logging
import os
import numpy as np
import torch
import pygame
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL training disabled.")

from ai_platform_trainer.gameplay.game_core import GameCore
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.entities.enemy_learning import LearningEnemyAI
from ai_platform_trainer.core.screen_context import ScreenContext


class MissileRLEnvironment(gym.Env):
    """
    Gymnasium environment for training missile AI using reinforcement learning.
    
    The missile learns to navigate towards moving targets while avoiding obstacles
    and optimizing trajectory efficiency.
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize ScreenContext for resolution independence
        ScreenContext.initialize(screen_width, screen_height)
        
        # Action space: turn rate (-1 to 1, normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: [missile_x, missile_y, missile_vx, missile_vy, 
        #                     target_x, target_y, target_vx, target_vy, distance, angle_to_target]  
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
        # Initialize pygame for headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = pygame.Surface((screen_width, screen_height))
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Create missile at random position on left side
        missile_x = np.random.uniform(50, self.screen_width * 0.3)
        missile_y = np.random.uniform(50, self.screen_height - 50)
        
        # Initial velocity towards center
        initial_vx = 5.0
        initial_vy = np.random.uniform(-2.0, 2.0)
        
        self.missile = Missile(missile_x, missile_y, speed=8.0, vx=initial_vx, vy=initial_vy)
        
        # Create moving target (enemy)
        target_x = np.random.uniform(self.screen_width * 0.6, self.screen_width - 50)
        target_y = np.random.uniform(50, self.screen_height - 50)
        
        self.target = {
            "x": target_x,
            "y": target_y,
            "vx": np.random.uniform(-3.0, 3.0),
            "vy": np.random.uniform(-3.0, 3.0)
        }
        
        self.steps = 0
        self.max_steps = 600  # 10 seconds at 60 FPS - more time to learn complex maneuvers
        self.last_distance = self._calculate_distance()
        self.consecutive_no_progress = 0  # Track when missile gets stuck
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Apply missile turn action
        turn_rate = action[0] * 15.0  # Increased turn rate for more agile movement
        current_angle = np.arctan2(self.missile.vy, self.missile.vx)
        new_angle = current_angle + np.radians(turn_rate)
        
        self.missile.vx = self.missile.speed * np.cos(new_angle)
        self.missile.vy = self.missile.speed * np.sin(new_angle)
        
        # Update missile position
        self.missile.update()
        
        # Update target position (moving target)
        self.target["x"] += self.target["vx"]
        self.target["y"] += self.target["vy"]
        
        # Bounce target off walls
        if self.target["x"] <= 0 or self.target["x"] >= self.screen_width:
            self.target["vx"] *= -1
        if self.target["y"] <= 0 or self.target["y"] >= self.screen_height:
            self.target["vy"] *= -1
            
        # Keep target in bounds
        self.target["x"] = np.clip(self.target["x"], 0, self.screen_width)
        self.target["y"] = np.clip(self.target["y"], 0, self.screen_height)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        self.last_distance = self._calculate_distance()
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Use ScreenContext for resolution-independent observations
        screen_context = ScreenContext.get_instance()
        
        # Create normalized observation using ScreenContext
        observation = screen_context.create_missile_observation(
            {"x": 0, "y": 0},  # Player pos not needed
            {"x": self.target["x"], "y": self.target["y"]},
            self.missile.pos,
            {"x": self.missile.vx, "y": self.missile.vy}
        )
        
        # Calculate angle to target
        angle_to_target = np.arctan2(
            self.target["y"] - self.missile.pos["y"],
            self.target["x"] - self.missile.pos["x"]
        )
        
        # Normalize target velocities
        target_vx_norm = self.target["vx"] / 5.0
        target_vy_norm = self.target["vy"] / 5.0
        angle_norm = angle_to_target / np.pi
        
        return np.array([
            observation["missile_x"], observation["missile_y"], 
            observation["velocity_x"], observation["velocity_y"],
            observation["target_x"], observation["target_y"], 
            target_vx_norm, target_vy_norm,
            observation["distance_to_target"], angle_norm
        ], dtype=np.float32)
    
    def _calculate_distance(self) -> float:
        """Calculate distance between missile and target."""
        dx = self.missile.pos["x"] - self.target["x"]
        dy = self.missile.pos["y"] - self.target["y"]
        return np.sqrt(dx * dx + dy * dy)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        distance = self._calculate_distance()
        
        # Hit reward - massive bonus for success
        if distance < 25:  # Hit target
            return 200.0
        
        # Very close reward - encourage getting very close
        if distance < 50:
            return 50.0
        
        # Distance-based reward (closer is better) - more aggressive scaling
        max_distance = np.sqrt(self.screen_width**2 + self.screen_height**2)
        distance_reward = (max_distance - distance) / max_distance * 2.0
        
        # Progress reward (getting closer) - stronger incentive
        progress_reward = 0.0
        if distance < self.last_distance:
            improvement = self.last_distance - distance
            progress_reward = improvement * 0.1  # Scale by improvement amount
            self.consecutive_no_progress = 0
        else:
            self.consecutive_no_progress += 1
            progress_reward = -0.5  # Stronger penalty for not improving
        
        # Penalty for getting stuck in circles
        if self.consecutive_no_progress > 20:
            progress_reward -= 2.0
        
        # Penalty for going out of bounds
        boundary_penalty = 0.0
        if (self.missile.pos["x"] < 0 or self.missile.pos["x"] > self.screen_width or
            self.missile.pos["y"] < 0 or self.missile.pos["y"] > self.screen_height):
            boundary_penalty = -20.0
        
        # Velocity towards target reward - encourage pointing at target
        missile_to_target = np.array([
            self.target["x"] - self.missile.pos["x"],
            self.target["y"] - self.missile.pos["y"]
        ])
        missile_velocity = np.array([self.missile.vx, self.missile.vy])
        
        if np.linalg.norm(missile_to_target) > 0 and np.linalg.norm(missile_velocity) > 0:
            # Normalized dot product gives cosine of angle between vectors
            cos_angle = np.dot(missile_to_target, missile_velocity) / (
                np.linalg.norm(missile_to_target) * np.linalg.norm(missile_velocity)
            )
            velocity_alignment_reward = cos_angle * 1.0  # Reward pointing at target
        else:
            velocity_alignment_reward = 0.0
        
        # Predict and intercept reward - encourage interception rather than chase
        # Predict where target will be
        prediction_time = distance / 8.0  # Missile speed
        predicted_target_x = self.target["x"] + self.target["vx"] * prediction_time
        predicted_target_y = self.target["y"] + self.target["vy"] * prediction_time
        
        predicted_distance = np.sqrt(
            (self.missile.pos["x"] - predicted_target_x)**2 + 
            (self.missile.pos["y"] - predicted_target_y)**2
        )
        
        intercept_reward = 0.0
        if predicted_distance < distance:  # Moving towards intercept point
            intercept_reward = 0.5
        
        return (distance_reward + progress_reward + boundary_penalty + 
                velocity_alignment_reward + intercept_reward)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Hit target
        if self._calculate_distance() < 25:
            return True
            
        # Missile went out of bounds
        if (self.missile.pos["x"] < -50 or self.missile.pos["x"] > self.screen_width + 50 or
            self.missile.pos["y"] < -50 or self.missile.pos["y"] > self.screen_height + 50):
            return True
            
        return False


class MissileRLTrainer:
    """Trainer for missile RL using PPO."""
    
    def __init__(self, save_path: str = "models/missile_rl_model"):
        self.save_path = save_path
        self.env = None
        self.model = None
        
    def create_environment(self):
        """Create the training environment."""
        self.env = MissileRLEnvironment()
        return self.env
    
    def train(self, total_timesteps: int = 100000, save_freq: int = 10000, progress_callback=None):
        """Train the missile RL model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for RL training")
        
        # Create environment
        env = self.create_environment()
        
        # Create PPO model with optimized hyperparameters for missile control
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            # tensorboard_log="./tensorboard_logs/missile_rl/",  # Disabled - requires tensorboard
            learning_rate=1e-3,  # Higher learning rate for faster learning
            n_steps=2048,  # Reduced for faster training
            batch_size=64,  # Smaller batch size for faster training
            n_epochs=10,  # Fewer epochs for faster training
            gamma=0.995,  # Higher discount for long-term planning
            gae_lambda=0.98,  # Higher lambda for better advantage estimation
            clip_range=0.3,  # Slightly higher clip range
            ent_coef=0.005,  # Lower entropy for more focused policy
            policy_kwargs=dict(
                net_arch=[128, 128]  # Smaller network for faster training
            )
        )
        
        # Setup evaluation callback
        eval_env = MissileRLEnvironment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path="./logs/missile_rl/",
            eval_freq=save_freq,
            deterministic=True,
            render=False
        )
        
        # Create progress tracking callback
        class ProgressTracker(BaseCallback):
            def __init__(self, callback_fn=None, verbose=0):
                super().__init__(verbose)
                self.callback_fn = callback_fn
                
            def _on_step(self) -> bool:
                if self.callback_fn:
                    self.callback_fn(self.num_timesteps, total_timesteps)
                return True
        
        callbacks = [eval_callback]
        if progress_callback:
            callbacks.append(ProgressTracker(progress_callback))
        
        # Train the model
        logging.info(f"Starting missile RL training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False
        )
        
        # Save final model
        final_path = f"{self.save_path}_final"
        self.model.save(final_path)
        logging.info(f"Training completed. Model saved to {final_path}")
        
        return self.model
    
    def test_model(self, model_path: str, num_episodes: int = 10):
        """Test the trained model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for testing")
            
        # Load model
        model = PPO.load(model_path)
        env = MissileRLEnvironment()
        
        total_rewards = []
        hit_count = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if env._calculate_distance() < 25:  # Hit target
                        hit_count += 1
                    break
            
            total_rewards.append(total_reward)
            logging.info(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        avg_reward = np.mean(total_rewards)
        hit_rate = hit_count / num_episodes
        
        logging.info(f"Test Results: Avg Reward = {avg_reward:.2f}, Hit Rate = {hit_rate:.2%}")
        return avg_reward, hit_rate


def main():
    """Main training function."""
    if not STABLE_BASELINES_AVAILABLE:
        print("stable_baselines3 is not available. Please install it to run RL training.")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create trainer
    trainer = MissileRLTrainer("models/missile_rl_model")
    
    # Train model with fewer timesteps for faster testing
    model = trainer.train(total_timesteps=50000)
    
    # Test model
    trainer.test_model("models/missile_rl_model_final.zip")


if __name__ == "__main__":
    main()