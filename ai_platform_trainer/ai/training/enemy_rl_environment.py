"""
Enemy Reinforcement Learning Environment

This module implements a true RL environment from the enemy's perspective,
where the enemy learns optimal evasion and survival strategies against
a missile-equipped player.
"""
import logging
import os
import math
import numpy as np
import torch
import pygame
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.core.screen_context import ScreenContext


@dataclass
class EnemyRLConfig:
    """Configuration for enemy RL training."""
    max_episode_steps: int = 1000
    missile_frequency: float = 0.05  # Probability of missile spawn per step
    player_skill_level: float = 0.7  # How accurately player aims (0-1)
    survival_reward_per_step: float = 0.1
    missile_dodge_reward: float = 10.0
    hit_penalty: float = -50.0
    boundary_penalty_strength: float = 5.0
    efficiency_bonus_weight: float = 2.0


class EnemyRLEnvironment(gym.Env):
    """
    RL Environment from the enemy's perspective.
    
    The enemy must learn to:
    1. Evade incoming missiles
    2. Survive as long as possible
    3. Move efficiently (not waste energy on unnecessary movement)
    4. Stay within game boundaries
    5. Potentially learn advanced tactics like baiting missiles
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600,
                 config: Optional[EnemyRLConfig] = None,
                 player_ai_model_path: Optional[str] = None):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config = config or EnemyRLConfig()
        
        # Initialize ScreenContext
        ScreenContext.initialize(screen_width, screen_height)
        
        # Action space: [delta_vx, delta_vy] - velocity changes
        # Enemy can accelerate in any direction
        self.max_acceleration = 2.0
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [enemy_x, enemy_y, enemy_vx, enemy_vy,
        #                     player_x, player_y, player_vx, player_vy,
        #                     missile1_x, missile1_y, missile1_vx, missile1_vy, missile1_active,
        #                     missile2_x, missile2_y, missile2_vx, missile2_vy, missile2_active,
        #                     missile3_x, missile3_y, missile3_vx, missile3_vy, missile3_active,
        #                     time_since_last_missile, enemy_energy, boundary_distances_x, boundary_distances_y]
        # Total: 4 (enemy) + 4 (player) + 5*3 (up to 3 missiles) + 4 (meta) = 27 features
        self.max_missiles = 3  # Track up to 3 active missiles
        obs_size = 4 + 4 + (5 * self.max_missiles) + 4
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize pygame for headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = pygame.Surface((screen_width, screen_height))
        
        # Game entities
        self.enemy_pos = {"x": 0, "y": 0}
        self.enemy_velocity = {"x": 0, "y": 0}
        self.enemy_max_speed = 4.0
        self.enemy_energy = 100.0  # Energy depletes with high acceleration
        self.enemy_size = 20
        
        # Player (controlled by AI or simple heuristic)
        self.player = None
        self.player_ai_model = None
        if player_ai_model_path and os.path.exists(player_ai_model_path):
            self._load_player_ai(player_ai_model_path)
        
        # Missiles
        self.missiles = []
        self.time_since_last_missile = 0
        
        # Episode tracking
        self.steps = 0
        self.survival_time = 0
        self.missiles_dodged = 0
        self.total_distance_moved = 0.0
        self.last_position = {"x": 0, "y": 0}
        
        self.reset()
    
    def _load_player_ai(self, model_path: str):
        """Load AI model to control player behavior."""
        try:
            from stable_baselines3 import PPO, SAC
            
            # Try to load as SAC first, then PPO
            try:
                self.player_ai_model = SAC.load(model_path)
                logging.info(f"Loaded SAC player AI from {model_path}")
            except:
                self.player_ai_model = PPO.load(model_path)
                logging.info(f"Loaded PPO player AI from {model_path}")
        except Exception as e:
            logging.warning(f"Failed to load player AI model: {e}")
            self.player_ai_model = None
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        # Reset enemy position (start away from center)
        start_positions = [
            (50, 50), (self.screen_width - 50, 50),
            (50, self.screen_height - 50), (self.screen_width - 50, self.screen_height - 50)
        ]
        start_x, start_y = start_positions[np.random.randint(len(start_positions))]
        
        self.enemy_pos = {"x": start_x, "y": start_y}
        self.enemy_velocity = {"x": 0, "y": 0}
        self.enemy_energy = 100.0
        self.last_position = {"x": start_x, "y": start_y}
        
        # Reset player position (center-ish with some randomness)
        player_x = self.screen_width // 2 + np.random.randint(-100, 100)
        player_y = self.screen_height // 2 + np.random.randint(-100, 100)
        
        if self.player is None:
            self.player = PlayerPlay(self.screen_width, self.screen_height)
        
        self.player.position = {"x": player_x, "y": player_y}
        self.player.missiles = []  # Clear existing missiles
        
        # Reset tracking
        self.missiles = []
        self.time_since_last_missile = 0
        self.steps = 0
        self.survival_time = 0
        self.missiles_dodged = 0
        self.total_distance_moved = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step from enemy's perspective."""
        self.steps += 1
        self.survival_time += 1
        self.time_since_last_missile += 1
        
        # Apply enemy action (acceleration)
        self._apply_enemy_action(action)
        
        # Update player behavior
        self._update_player_behavior()
        
        # Spawn missiles based on frequency and player behavior
        self._handle_missile_spawning()
        
        # Update missiles
        self._update_missiles()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.config.max_episode_steps
        
        # Update tracking
        self._update_tracking()
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _apply_enemy_action(self, action):
        """Apply enemy's movement action."""
        # Convert action to acceleration
        accel_x = action[0] * self.max_acceleration
        accel_y = action[1] * self.max_acceleration
        
        # Update velocity
        self.enemy_velocity["x"] += accel_x
        self.enemy_velocity["y"] += accel_y
        
        # Apply speed limit
        speed = math.sqrt(self.enemy_velocity["x"]**2 + self.enemy_velocity["y"]**2)
        if speed > self.enemy_max_speed:
            self.enemy_velocity["x"] = (self.enemy_velocity["x"] / speed) * self.enemy_max_speed
            self.enemy_velocity["y"] = (self.enemy_velocity["y"] / speed) * self.enemy_max_speed
        
        # Update position
        self.enemy_pos["x"] += self.enemy_velocity["x"]
        self.enemy_pos["y"] += self.enemy_velocity["y"]
        
        # Keep enemy in bounds (soft boundary)
        margin = 30
        if self.enemy_pos["x"] < margin:
            self.enemy_pos["x"] = margin
            self.enemy_velocity["x"] = max(0, self.enemy_velocity["x"])
        elif self.enemy_pos["x"] > self.screen_width - margin:
            self.enemy_pos["x"] = self.screen_width - margin
            self.enemy_velocity["x"] = min(0, self.enemy_velocity["x"])
        
        if self.enemy_pos["y"] < margin:
            self.enemy_pos["y"] = margin
            self.enemy_velocity["y"] = max(0, self.enemy_velocity["y"])
        elif self.enemy_pos["y"] > self.screen_height - margin:
            self.enemy_pos["y"] = self.screen_height - margin
            self.enemy_velocity["y"] = min(0, self.enemy_velocity["y"])
        
        # Update energy (depletes with high acceleration)
        acceleration_magnitude = math.sqrt(accel_x**2 + accel_y**2)
        energy_cost = acceleration_magnitude * 0.5
        self.enemy_energy = max(0, self.enemy_energy - energy_cost)
        
        # Regenerate energy slowly
        self.enemy_energy = min(100.0, self.enemy_energy + 0.1)
    
    def _update_player_behavior(self):
        """Update player position and behavior."""
        if self.player_ai_model:
            # Use AI model to control player
            self._update_player_with_ai()
        else:
            # Simple heuristic player behavior
            self._update_player_heuristic()
    
    def _update_player_with_ai(self):
        """Update player using loaded AI model."""
        # This would require creating a compatible observation for the player's AI
        # For now, use heuristic
        self._update_player_heuristic()
    
    def _update_player_heuristic(self):
        """Simple heuristic player behavior."""
        # Player tries to maintain some distance from enemy but not too far
        enemy_x, enemy_y = self.enemy_pos["x"], self.enemy_pos["y"]
        player_x, player_y = self.player.position["x"], self.player.position["y"]
        
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Desired distance for good missile accuracy
        desired_distance = 150
        
        if distance > desired_distance + 50:
            # Move closer
            if dx != 0 or dy != 0:
                move_speed = 2.0
                self.player.position["x"] += (dx / distance) * move_speed
                self.player.position["y"] += (dy / distance) * move_speed
        elif distance < desired_distance - 50:
            # Move away
            if dx != 0 or dy != 0:
                move_speed = 1.5
                self.player.position["x"] -= (dx / distance) * move_speed
                self.player.position["y"] -= (dy / distance) * move_speed
        
        # Keep player in bounds
        self.player.position["x"] = np.clip(self.player.position["x"], 20, self.screen_width - 20)
        self.player.position["y"] = np.clip(self.player.position["y"], 20, self.screen_height - 20)
    
    def _handle_missile_spawning(self):
        """Handle missile spawning based on player behavior."""
        # Spawn missile based on frequency and conditions
        if (len(self.missiles) < self.max_missiles and 
            self.time_since_last_missile > 30 and  # Minimum time between missiles
            np.random.random() < self.config.missile_frequency):
            
            self._spawn_missile()
    
    def _spawn_missile(self):
        """Spawn a missile from player towards enemy."""
        player_x = self.player.position["x"]
        player_y = self.player.position["y"]
        enemy_x = self.enemy_pos["x"]
        enemy_y = self.enemy_pos["y"]
        
        # Calculate direction with some inaccuracy based on player skill
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Add noise based on player skill (lower skill = more noise)
            noise_factor = (1.0 - self.config.player_skill_level) * 0.3
            angle_noise = np.random.normal(0, noise_factor)
            
            base_angle = math.atan2(dy, dx)
            actual_angle = base_angle + angle_noise
            
            missile_speed = 8.0
            missile_vx = missile_speed * math.cos(actual_angle)
            missile_vy = missile_speed * math.sin(actual_angle)
            
            # Create missile
            missile = Missile(
                x=player_x, y=player_y,
                speed=missile_speed,
                vx=missile_vx, vy=missile_vy,
                birth_time=pygame.time.get_ticks(),
                lifespan=5000  # 5 seconds
            )
            
            self.missiles.append(missile)
            self.time_since_last_missile = 0
    
    def _update_missiles(self):
        """Update all active missiles."""
        current_time = pygame.time.get_ticks()
        missiles_to_remove = []
        
        for missile in self.missiles:
            # Update missile position
            missile.update()
            
            # Check if missile expired
            if current_time - missile.birth_time > missile.lifespan:
                missiles_to_remove.append(missile)
                continue
            
            # Check if missile went out of bounds
            if (missile.pos["x"] < -50 or missile.pos["x"] > self.screen_width + 50 or
                missile.pos["y"] < -50 or missile.pos["y"] > self.screen_height + 50):
                missiles_to_remove.append(missile)
                self.missiles_dodged += 1  # Successfully evaded
                continue
            
            # Check collision with enemy
            enemy_rect = pygame.Rect(
                int(self.enemy_pos["x"] - self.enemy_size//2),
                int(self.enemy_pos["y"] - self.enemy_size//2),
                self.enemy_size, self.enemy_size
            )
            
            # Create missile rect manually since get_rect() might not work
            missile_rect = pygame.Rect(
                int(missile.pos["x"] - 5),  # Missile size/2
                int(missile.pos["y"] - 5),
                10, 10  # Missile size
            )
            
            if enemy_rect.colliderect(missile_rect):
                missiles_to_remove.append(missile)
                # Hit will be handled in termination check
                continue
        
        # Remove expired/hit missiles
        for missile in missiles_to_remove:
            if missile in self.missiles:
                self.missiles.remove(missile)
    
    def _calculate_reward(self, action):
        """Calculate reward for enemy's current state and action."""
        reward = 0.0
        
        # 1. SURVIVAL REWARD - base reward for staying alive
        reward += self.config.survival_reward_per_step
        
        # 2. MISSILE EVASION REWARDS
        # Reward for maintaining distance from missiles
        missile_danger_reward = 0.0
        closest_missile_distance = float('inf')
        
        for missile in self.missiles:
            missile_x, missile_y = missile.pos["x"], missile.pos["y"]
            distance = math.sqrt(
                (missile_x - self.enemy_pos["x"])**2 + 
                (missile_y - self.enemy_pos["y"])**2
            )
            closest_missile_distance = min(closest_missile_distance, distance)
            
            # Reward for staying away from missiles
            if distance < 100:  # Danger zone
                # Higher reward for being farther from missile
                safety_reward = (distance / 100.0) * 2.0
                missile_danger_reward += safety_reward
                
                # Extra reward for moving away from missile
                missile_vx, missile_vy = missile.vx, missile.vy
                enemy_vx, enemy_vy = self.enemy_velocity["x"], self.enemy_velocity["y"]
                
                # Dot product of missile velocity and enemy velocity
                # Negative dot product means enemy is moving away from missile's direction
                dot_product = missile_vx * enemy_vx + missile_vy * enemy_vy
                if dot_product < 0:  # Moving away from missile
                    evasion_reward = abs(dot_product) / (8.0 * self.enemy_max_speed) * 1.5
                    missile_danger_reward += evasion_reward
        
        reward += missile_danger_reward
        
        # 3. ENERGY EFFICIENCY - penalize unnecessary movement
        action_magnitude = math.sqrt(action[0]**2 + action[1]**2)
        
        # If no immediate danger, prefer staying still
        if closest_missile_distance > 150:
            energy_penalty = -action_magnitude * 0.2
            reward += energy_penalty
        
        # 4. BOUNDARY AVOIDANCE - stay away from edges
        boundary_penalty = 0.0
        margin = 50
        
        if self.enemy_pos["x"] < margin:
            boundary_penalty -= (margin - self.enemy_pos["x"]) / margin
        elif self.enemy_pos["x"] > self.screen_width - margin:
            boundary_penalty -= (self.enemy_pos["x"] - (self.screen_width - margin)) / margin
        
        if self.enemy_pos["y"] < margin:
            boundary_penalty -= (margin - self.enemy_pos["y"]) / margin
        elif self.enemy_pos["y"] > self.screen_height - margin:
            boundary_penalty -= (self.enemy_pos["y"] - (self.screen_height - margin)) / margin
        
        reward += boundary_penalty * self.config.boundary_penalty_strength
        
        # 5. STRATEGIC POSITIONING - maintain good position relative to player
        player_x = self.player.position["x"]
        player_y = self.player.position["y"]
        distance_to_player = math.sqrt(
            (self.enemy_pos["x"] - player_x)**2 + 
            (self.enemy_pos["y"] - player_y)**2
        )
        
        # Optimal distance is neither too close nor too far
        optimal_distance = 200
        distance_penalty = -abs(distance_to_player - optimal_distance) / optimal_distance * 0.5
        reward += distance_penalty
        
        # 6. VELOCITY SMOOTHNESS - prefer smooth movements
        velocity_magnitude = math.sqrt(self.enemy_velocity["x"]**2 + self.enemy_velocity["y"]**2)
        if velocity_magnitude > self.enemy_max_speed * 0.8:
            speed_penalty = -0.1  # Small penalty for high speed
            reward += speed_penalty
        
        return reward
    
    def _is_terminated(self):
        """Check if episode should terminate (enemy got hit)."""
        # Check collision with any missile
        enemy_rect = pygame.Rect(
            int(self.enemy_pos["x"] - self.enemy_size//2),
            int(self.enemy_pos["y"] - self.enemy_size//2),
            self.enemy_size, self.enemy_size
        )
        
        for missile in self.missiles:
            # Create missile rect manually since get_rect() might not work
            missile_rect = pygame.Rect(
                int(missile.pos["x"] - 5),  # Missile size/2
                int(missile.pos["y"] - 5),
                10, 10  # Missile size
            )
            if enemy_rect.colliderect(missile_rect):
                return True
        
        return False
    
    def _update_tracking(self):
        """Update tracking variables for analysis."""
        # Calculate distance moved this step
        dx = self.enemy_pos["x"] - self.last_position["x"]
        dy = self.enemy_pos["y"] - self.last_position["y"]
        distance_moved = math.sqrt(dx*dx + dy*dy)
        self.total_distance_moved += distance_moved
        
        self.last_position = {"x": self.enemy_pos["x"], "y": self.enemy_pos["y"]}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state for the enemy."""
        obs = []
        
        # Enemy state (normalized)
        obs.extend([
            self.enemy_pos["x"] / self.screen_width * 2 - 1,
            self.enemy_pos["y"] / self.screen_height * 2 - 1,
            self.enemy_velocity["x"] / self.enemy_max_speed,
            self.enemy_velocity["y"] / self.enemy_max_speed
        ])
        
        # Player state (normalized)
        obs.extend([
            self.player.position["x"] / self.screen_width * 2 - 1,
            self.player.position["y"] / self.screen_height * 2 - 1,
            0,  # Player vx (not tracked in simple version)
            0   # Player vy (not tracked in simple version)
        ])
        
        # Missile states (up to max_missiles)
        missiles_added = 0
        for missile in self.missiles[:self.max_missiles]:
            obs.extend([
                missile.pos["x"] / self.screen_width * 2 - 1,
                missile.pos["y"] / self.screen_height * 2 - 1,
                missile.vx / 10.0,  # Normalize missile velocity
                missile.vy / 10.0,
                1.0  # Missile active
            ])
            missiles_added += 1
        
        # Pad with zeros for missing missiles
        for _ in range(self.max_missiles - missiles_added):
            obs.extend([0, 0, 0, 0, 0])  # Inactive missile
        
        # Meta information
        obs.extend([
            self.time_since_last_missile / 100.0,  # Normalized time
            self.enemy_energy / 100.0,  # Normalized energy
            min(self.enemy_pos["x"], self.screen_width - self.enemy_pos["x"]) / self.screen_width,  # Distance to closest x boundary
            min(self.enemy_pos["y"], self.screen_height - self.enemy_pos["y"]) / self.screen_height   # Distance to closest y boundary
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for analysis."""
        return {
            "survival_time": self.survival_time,
            "missiles_dodged": self.missiles_dodged,
            "enemy_energy": self.enemy_energy,
            "active_missiles": len(self.missiles),
            "total_distance_moved": self.total_distance_moved,
            "movement_efficiency": self.survival_time / max(1, self.total_distance_moved),
            "hit": self._is_terminated()
        }


class EnemyRLTrainer:
    """Trainer for enemy RL using SAC (best for continuous control)."""
    
    def __init__(self, save_path: str = "models/enemy_rl_sac",
                 config: Optional[EnemyRLConfig] = None):
        self.save_path = save_path
        self.config = config or EnemyRLConfig()
        self.env = None
        self.model = None
    
    def create_environment(self):
        """Create the enemy training environment."""
        self.env = EnemyRLEnvironment(config=self.config)
        return self.env
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 5000):
        """Train the enemy RL model."""
        try:
            from stable_baselines3 import SAC
            from stable_baselines3.common.callbacks import EvalCallback
        except ImportError:
            raise ImportError("stable_baselines3 is required for enemy RL training")
        
        # Create environment
        env = self.create_environment()
        
        # Create SAC model (best for continuous control)
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=2000,  # More exploration before learning
            batch_size=256,
            tau=0.005,
            gamma=0.995,  # Higher discount for long-term survival
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # Larger network for complex behavior
                activation_fn='torch.nn.ReLU',
            ),
            seed=42
        )
        
        # Setup evaluation
        eval_env = EnemyRLEnvironment(config=self.config)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.save_path,
            log_path="./logs/enemy_rl/",
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Train
        logging.info(f"Starting enemy RL training for {total_timesteps} timesteps...")
        logging.info("Enemy will learn: evasion, survival, energy efficiency")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=False
        )
        
        # Save final model
        final_path = f"{self.save_path}_final"
        self.model.save(final_path)
        logging.info(f"Enemy RL training completed. Model saved to {final_path}")
        
        return self.model
    
    def test_model(self, model_path: str, num_episodes: int = 20):
        """Test the trained enemy model."""
        try:
            from stable_baselines3 import SAC
        except ImportError:
            raise ImportError("stable_baselines3 is required for testing")
        
        # Load model
        model = SAC.load(model_path)
        env = EnemyRLEnvironment(config=self.config)
        
        survival_times = []
        missiles_dodged_list = []
        movement_efficiencies = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            survival_times.append(info["survival_time"])
            missiles_dodged_list.append(info["missiles_dodged"])
            movement_efficiencies.append(info["movement_efficiency"])
            
            logging.info(f"Episode {episode + 1}: Survived {info['survival_time']} steps, "
                        f"Dodged {info['missiles_dodged']} missiles")
        
        avg_survival = np.mean(survival_times)
        avg_dodged = np.mean(missiles_dodged_list)
        avg_efficiency = np.mean(movement_efficiencies)
        
        logging.info(f"Enemy RL Test Results:")
        logging.info(f"  Average Survival Time: {avg_survival:.1f} steps")
        logging.info(f"  Average Missiles Dodged: {avg_dodged:.1f}")
        logging.info(f"  Average Movement Efficiency: {avg_efficiency:.3f}")
        
        return {
            "avg_survival_time": avg_survival,
            "avg_missiles_dodged": avg_dodged,
            "avg_movement_efficiency": avg_efficiency,
            "survival_times": survival_times
        }