"""
SAC (Soft Actor-Critic) Training for Missile AI

This module provides more sample-efficient RL training for missile homing 
behavior using SAC, which is better suited for continuous control tasks.
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
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.noise import NormalActionNoise
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL training disabled.")

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.core.screen_context import ScreenContext


class MissileSACEnvironment(gym.Env):
    """
    Optimized Gymnasium environment for SAC missile training.
    
    Designed specifically for continuous control with better reward shaping
    and observation normalization for SAC's stability requirements.
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize ScreenContext for resolution independence
        ScreenContext.initialize(screen_width, screen_height)
        
        # Action space: turn rate (-1 to 1, normalized) - perfect for SAC
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation space: normalized and bounded for SAC stability
        # [missile_x_norm, missile_y_norm, missile_vx_norm, missile_vy_norm, 
        #  target_x_norm, target_y_norm, target_vx_norm, target_vy_norm, 
        #  distance_norm, angle_to_target_norm, relative_angle_norm]  
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(11,), dtype=np.float32
        )
        
        # Initialize pygame for headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = pygame.Surface((screen_width, screen_height))
        
        # SAC-specific parameters
        self.max_speed = 10.0
        self.max_turn_rate = 20.0  # Degrees per step
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Create missile at random position 
        missile_x = np.random.uniform(50, self.screen_width - 50)
        missile_y = np.random.uniform(50, self.screen_height - 50)
        
        # Random initial velocity direction
        angle = np.random.uniform(0, 2 * np.pi)
        initial_speed = np.random.uniform(6.0, 10.0)
        initial_vx = initial_speed * np.cos(angle)
        initial_vy = initial_speed * np.sin(angle)
        
        self.missile = Missile(missile_x, missile_y, speed=initial_speed, vx=initial_vx, vy=initial_vy)
        
        # Create moving target with more complex movement patterns
        target_x = np.random.uniform(100, self.screen_width - 100)
        target_y = np.random.uniform(100, self.screen_height - 100)
        
        # Target with varied movement patterns
        target_speed = np.random.uniform(2.0, 5.0)
        target_angle = np.random.uniform(0, 2 * np.pi)
        
        self.target = {
            "x": target_x,
            "y": target_y,
            "vx": target_speed * np.cos(target_angle),
            "vy": target_speed * np.sin(target_angle),
            "pattern": np.random.choice(["linear", "circular", "zigzag"]),
            "pattern_time": 0.0
        }
        
        self.steps = 0
        self.max_steps = 500  # Slightly reduced for faster training
        self.last_distance = self._calculate_distance()
        self.distance_history = [self.last_distance]
        self.success_threshold = 20  # Smaller threshold for more precise targeting
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Apply missile turn action with bounded turn rate
        turn_rate = np.clip(action[0] * self.max_turn_rate, -self.max_turn_rate, self.max_turn_rate)
        current_angle = np.arctan2(self.missile.vy, self.missile.vx)
        new_angle = current_angle + np.radians(turn_rate)
        
        # Maintain constant speed but allow direction changes
        self.missile.vx = self.missile.speed * np.cos(new_angle)
        self.missile.vy = self.missile.speed * np.sin(new_angle)
        
        # Update missile position
        self.missile.update()
        
        # Update target with pattern-based movement
        self._update_target_movement()
        
        # Calculate reward using SAC-optimized reward function
        reward = self._calculate_sac_reward(action[0])
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        # Update history
        current_distance = self._calculate_distance()
        self.distance_history.append(current_distance)
        if len(self.distance_history) > 10:
            self.distance_history.pop(0)
        self.last_distance = current_distance
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def _update_target_movement(self):
        """Update target with complex movement patterns."""
        self.target["pattern_time"] += 0.1
        
        if self.target["pattern"] == "circular":
            # Circular movement
            center_x = self.screen_width / 2
            center_y = self.screen_height / 2
            radius = 100
            angle = self.target["pattern_time"]
            self.target["x"] = center_x + radius * np.cos(angle)
            self.target["y"] = center_y + radius * np.sin(angle)
            self.target["vx"] = -radius * 0.1 * np.sin(angle)
            self.target["vy"] = radius * 0.1 * np.cos(angle)
        
        elif self.target["pattern"] == "zigzag":
            # Zigzag movement
            self.target["x"] += self.target["vx"]
            self.target["y"] += self.target["vy"]
            
            if self.steps % 30 == 0:  # Change direction every 30 steps
                self.target["vx"] *= -0.8
                self.target["vy"] += np.random.uniform(-2, 2)
        
        else:  # linear
            # Linear movement with wall bouncing
            self.target["x"] += self.target["vx"]
            self.target["y"] += self.target["vy"]
            
            # Bounce off walls
            if self.target["x"] <= 20 or self.target["x"] >= self.screen_width - 20:
                self.target["vx"] *= -1
            if self.target["y"] <= 20 or self.target["y"] >= self.screen_height - 20:
                self.target["vy"] *= -1
        
        # Keep target in bounds
        self.target["x"] = np.clip(self.target["x"], 20, self.screen_width - 20)
        self.target["y"] = np.clip(self.target["y"], 20, self.screen_height - 20)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state - normalized for SAC stability."""
        # Normalize positions to [-1, 1]
        missile_x_norm = (self.missile.pos["x"] / self.screen_width) * 2 - 1
        missile_y_norm = (self.missile.pos["y"] / self.screen_height) * 2 - 1
        target_x_norm = (self.target["x"] / self.screen_width) * 2 - 1
        target_y_norm = (self.target["y"] / self.screen_height) * 2 - 1
        
        # Normalize velocities
        missile_vx_norm = self.missile.vx / self.max_speed
        missile_vy_norm = self.missile.vy / self.max_speed
        target_vx_norm = self.target["vx"] / 5.0
        target_vy_norm = self.target["vy"] / 5.0
        
        # Calculate relative vectors
        dx = self.target["x"] - self.missile.pos["x"]
        dy = self.target["y"] - self.missile.pos["y"]
        distance = np.sqrt(dx * dx + dy * dy)
        distance_norm = distance / (np.sqrt(self.screen_width**2 + self.screen_height**2))
        
        # Angle to target
        angle_to_target = np.arctan2(dy, dx)
        angle_to_target_norm = angle_to_target / np.pi
        
        # Relative angle (missile heading vs target direction)
        missile_angle = np.arctan2(self.missile.vy, self.missile.vx)
        relative_angle = angle_to_target - missile_angle
        # Normalize to [-1, 1]
        relative_angle = np.arctan2(np.sin(relative_angle), np.cos(relative_angle)) / np.pi
        
        return np.array([
            missile_x_norm, missile_y_norm, 
            missile_vx_norm, missile_vy_norm,
            target_x_norm, target_y_norm, 
            target_vx_norm, target_vy_norm,
            distance_norm, angle_to_target_norm, relative_angle
        ], dtype=np.float32)
    
    def _calculate_distance(self) -> float:
        """Calculate distance between missile and target."""
        dx = self.missile.pos["x"] - self.target["x"]
        dy = self.missile.pos["y"] - self.target["y"]
        return np.sqrt(dx * dx + dy * dy)
    
    def _calculate_sac_reward(self, action: float) -> float:
        """
        SAC-optimized reward function with dense, smooth rewards.
        
        SAC works better with:
        - Dense rewards (reward at every step)
        - Smooth reward functions (no sudden jumps)
        - Bounded rewards (prevents exploding Q-values)
        """
        distance = self._calculate_distance()
        
        # Primary reward: negative distance (smooth, dense)
        max_distance = np.sqrt(self.screen_width**2 + self.screen_height**2)
        distance_reward = -distance / max_distance
        
        # Success reward (but not too large to avoid instability)
        if distance < self.success_threshold:
            return 10.0  # Large but bounded success reward
        
        # Very close reward (smooth transition)
        if distance < 40:
            proximity_bonus = (40 - distance) / 40 * 2.0
            distance_reward += proximity_bonus
        
        # Progress reward (smoother than step-based)
        if len(self.distance_history) >= 2:
            recent_progress = self.distance_history[-2] - distance
            progress_reward = recent_progress / max_distance * 5.0
            progress_reward = np.clip(progress_reward, -1.0, 1.0)  # Bounded
        else:
            progress_reward = 0.0
        
        # Velocity alignment reward (smooth, continuous)
        missile_to_target = np.array([
            self.target["x"] - self.missile.pos["x"],
            self.target["y"] - self.missile.pos["y"]
        ])
        missile_velocity = np.array([self.missile.vx, self.missile.vy])
        
        if np.linalg.norm(missile_to_target) > 0 and np.linalg.norm(missile_velocity) > 0:
            cos_angle = np.dot(missile_to_target, missile_velocity) / (
                np.linalg.norm(missile_to_target) * np.linalg.norm(missile_velocity)
            )
            # Smooth alignment reward
            alignment_reward = (cos_angle + 1) / 2 * 0.5  # Maps [-1,1] to [0,1], scaled
        else:
            alignment_reward = 0.0
        
        # Interception prediction reward (advanced guidance)
        prediction_steps = min(distance / self.missile.speed, 20)
        predicted_target_x = self.target["x"] + self.target["vx"] * prediction_steps
        predicted_target_y = self.target["y"] + self.target["vy"] * prediction_steps
        
        # Vector to predicted intercept point
        intercept_vector = np.array([
            predicted_target_x - self.missile.pos["x"],
            predicted_target_y - self.missile.pos["y"]
        ])
        
        if np.linalg.norm(intercept_vector) > 0 and np.linalg.norm(missile_velocity) > 0:
            intercept_alignment = np.dot(intercept_vector, missile_velocity) / (
                np.linalg.norm(intercept_vector) * np.linalg.norm(missile_velocity)
            )
            intercept_reward = (intercept_alignment + 1) / 2 * 0.3
        else:
            intercept_reward = 0.0
        
        # Action smoothness penalty (prevent erratic behavior)
        action_penalty = -abs(action) * 0.05
        
        # Boundary penalty (smooth, not binary)
        boundary_penalty = 0.0
        margin = 50
        if self.missile.pos["x"] < margin:
            boundary_penalty -= (margin - self.missile.pos["x"]) / margin * 0.5
        elif self.missile.pos["x"] > self.screen_width - margin:
            boundary_penalty -= (self.missile.pos["x"] - (self.screen_width - margin)) / margin * 0.5
        
        if self.missile.pos["y"] < margin:
            boundary_penalty -= (margin - self.missile.pos["y"]) / margin * 0.5
        elif self.missile.pos["y"] > self.screen_height - margin:
            boundary_penalty -= (self.missile.pos["y"] - (self.screen_height - margin)) / margin * 0.5
        
        # Combine rewards with careful weighting
        total_reward = (distance_reward * 1.0 + 
                       progress_reward * 0.5 + 
                       alignment_reward * 0.3 + 
                       intercept_reward * 0.4 + 
                       action_penalty + 
                       boundary_penalty)
        
        # Clip final reward to prevent instability
        return np.clip(total_reward, -5.0, 10.0)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Hit target
        if self._calculate_distance() < self.success_threshold:
            return True
            
        # Missile went way out of bounds
        if (self.missile.pos["x"] < -100 or self.missile.pos["x"] > self.screen_width + 100 or
            self.missile.pos["y"] < -100 or self.missile.pos["y"] > self.screen_height + 100):
            return True
            
        return False


class MissileSACTrainer:
    """SAC trainer for missile guidance - optimized for sample efficiency."""
    
    def __init__(self, save_path: str = "models/missile_sac_model"):
        self.save_path = save_path
        self.env = None
        self.model = None
        
    def create_environment(self):
        """Create the training environment."""
        self.env = MissileSACEnvironment()
        return self.env
    
    def train(self, total_timesteps: int = 50000, save_freq: int = 5000, progress_callback=None):
        """Train the missile SAC model with optimized hyperparameters."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for RL training")
        
        # Create environment
        env = self.create_environment()
        
        # SAC hyperparameters optimized for continuous control
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,  # Standard SAC learning rate
            buffer_size=100000,  # Large replay buffer for off-policy learning
            learning_starts=1000,  # Start learning early
            batch_size=256,  # Larger batch for stable gradients
            tau=0.005,  # Soft update rate
            gamma=0.99,  # Standard discount factor
            train_freq=1,  # Update every step (off-policy advantage)
            gradient_steps=1,  # Number of gradient steps per update
            ent_coef='auto',  # Automatic entropy coefficient tuning
            target_update_interval=1,  # Update target networks every step
            policy_kwargs=dict(
                net_arch=[256, 256],  # Larger networks for complex control
                activation_fn=torch.nn.ReLU,
            ),
            seed=42  # For reproducible results
        )
        
        # Setup evaluation callback
        eval_env = MissileSACEnvironment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path="./logs/missile_sac/",
            eval_freq=save_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        
        # Create progress tracking callback
        class ProgressTracker(BaseCallback):
            def __init__(self, callback_fn=None, verbose=0):
                super().__init__(verbose)
                self.callback_fn = callback_fn
                self.episode_rewards = []
                self.episode_lengths = []
                
            def _on_step(self) -> bool:
                if self.callback_fn:
                    self.callback_fn(self.num_timesteps, total_timesteps)
                
                # Track episode statistics
                if len(self.locals.get('infos', [])) > 0:
                    info = self.locals['infos'][0]
                    if 'episode' in info:
                        self.episode_rewards.append(info['episode']['r'])
                        self.episode_lengths.append(info['episode']['l'])
                        
                        # Log recent performance
                        if len(self.episode_rewards) >= 10:
                            recent_reward = np.mean(self.episode_rewards[-10:])
                            recent_length = np.mean(self.episode_lengths[-10:])
                            logging.info(f"Recent 10 episodes: Avg reward = {recent_reward:.2f}, Avg length = {recent_length:.1f}")
                
                return True
        
        callbacks = [eval_callback]
        if progress_callback:
            callbacks.append(ProgressTracker(progress_callback))
        
        # Train the model
        logging.info(f"Starting missile SAC training for {total_timesteps} timesteps...")
        logging.info("SAC advantages: Off-policy learning, sample efficiency, continuous control")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=False
        )
        
        # Save final model
        final_path = f"{self.save_path}_final"
        self.model.save(final_path)
        logging.info(f"SAC training completed. Model saved to {final_path}")
        
        return self.model
    
    def test_model(self, model_path: str, num_episodes: int = 20):
        """Test the trained SAC model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for testing")
            
        # Load model
        model = SAC.load(model_path)
        env = MissileSACEnvironment()
        
        total_rewards = []
        hit_count = 0
        step_counts = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Use deterministic action for testing
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if env._calculate_distance() < env.success_threshold:
                        hit_count += 1
                    break
            
            total_rewards.append(total_reward)
            step_counts.append(steps)
            logging.info(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
        
        avg_reward = np.mean(total_rewards)
        hit_rate = hit_count / num_episodes
        avg_steps = np.mean(step_counts)
        
        logging.info(f"SAC Test Results:")
        logging.info(f"  Average Reward: {avg_reward:.2f}")
        logging.info(f"  Hit Rate: {hit_rate:.2%}")
        logging.info(f"  Average Steps to Completion: {avg_steps:.1f}")
        
        return avg_reward, hit_rate, avg_steps


def main():
    """Main SAC training function."""
    if not STABLE_BASELINES_AVAILABLE:
        print("stable_baselines3 is not available. Please install it to run SAC training.")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create SAC trainer
    trainer = MissileSACTrainer("models/missile_sac_model")
    
    # Train model - SAC typically needs fewer timesteps than PPO
    logging.info("Training with SAC (Soft Actor-Critic) - optimized for continuous control")
    model = trainer.train(total_timesteps=30000)  # Less than PPO due to sample efficiency
    
    # Test model
    trainer.test_model("models/missile_sac_model_final.zip")


if __name__ == "__main__":
    main()