"""
Reinforcement Learning Training Mode.

Watch the enemy learn in real-time using PPO while you play.
"""
import pygame
import logging
import numpy as np
from typing import Optional

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

from ai_platform_trainer.game.entities.player import Player
from ai_platform_trainer.game.entities.enemy import Enemy


class RLTrainingMode:
    """Real-time reinforcement learning training mode."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create entities
        self.player = Player(screen_width, screen_height, mode="training")
        self.enemy = Enemy(screen_width, screen_height, mode="rl_training")
        
        # RL Training setup
        self.rl_model: Optional[PPO] = None
        self.training_active = False
        self.episode_count = 0
        self.total_reward = 0.0
        self.episode_reward = 0.0
        self.step_count = 0
        
        # Training metrics
        self.metrics = {
            "episodes": 0,
            "total_steps": 0,
            "avg_reward": 0.0,
            "learning_rate": 0.0003,
            "epsilon": 1.0
        }
        
        if STABLE_BASELINES_AVAILABLE:
            self._init_rl_training()
        else:
            logging.error("Stable Baselines3 not available - RL training disabled")
    
    def _init_rl_training(self):
        """Initialize RL training components."""
        try:
            # Create a simple PPO model for real-time training
            from stable_baselines3.common.env_util import make_vec_env
            from ai_platform_trainer.ai.models.game_environment import GameEnvironment
            
            # Create environment
            env = DummyVecEnv([lambda: GameEnvironment(render_mode='none')])
            
            # Create PPO model
            self.rl_model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=0.0003,
                n_steps=64,  # Smaller steps for real-time training
                batch_size=32,
                n_epochs=4,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.01
            )
            
            self.training_active = True
            logging.info("RL training initialized")
            
        except Exception as e:
            logging.error(f"Failed to initialize RL training: {e}")
            self.training_active = False
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.shoot_missile(self.enemy.pos)
            elif event.key == pygame.K_r:
                self._reset_episode()
            elif event.key == pygame.K_t:
                self.training_active = not self.training_active
                logging.info(f"Training {'enabled' if self.training_active else 'disabled'}")
    
    def update(self):
        """Update training mode."""
        current_time = pygame.time.get_ticks()
        
        # Update player
        self.player.handle_input()
        self.player.update_missiles()
        
        # RL Training step
        if self.training_active and self.rl_model:
            self._training_step()
        
        # Update enemy (using current RL policy)
        self.enemy.update_movement(
            self.player.position["x"],
            self.player.position["y"],
            self.player.step,
            current_time
        )
        
        # Check for episode end conditions
        if self._check_episode_end():
            self._end_episode()
        
        self.step_count += 1
    
    def _training_step(self):
        """Perform one RL training step."""
        try:
            # Get current observation
            obs = self._get_observation()
            
            # Get action from current policy
            action, _ = self.rl_model.predict(obs, deterministic=False)
            
            # Apply action to enemy
            self.enemy.pos["x"] += float(action[0]) * self.enemy.speed
            self.enemy.pos["y"] += float(action[1]) * self.enemy.speed
            self.enemy._wrap_position()
            
            # Calculate reward
            reward = self._calculate_reward()
            self.episode_reward += reward
            
            # Store experience (simplified for real-time training)
            # In a full implementation, you'd use a proper replay buffer
            
        except Exception as e:
            logging.error(f"Training step error: {e}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for RL."""
        # Calculate distance
        dx = self.player.position["x"] - self.enemy.pos["x"]
        dy = self.player.position["y"] - self.enemy.pos["y"]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Normalize observations
        obs = np.array([
            self.player.position["x"] / self.screen_width,
            self.player.position["y"] / self.screen_height,
            self.enemy.pos["x"] / self.screen_width,
            self.enemy.pos["y"] / self.screen_height,
            distance / max(self.screen_width, self.screen_height),
            self.player.step / 10.0,
            0.5  # Time factor
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state."""
        # Distance-based reward
        dx = self.player.position["x"] - self.enemy.pos["x"]
        dy = self.player.position["y"] - self.enemy.pos["y"]
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Reward for being close to player
        proximity_reward = 10.0 / (distance + 1.0)
        
        # Penalty for being too far
        if distance > 300:
            proximity_reward -= 1.0
        
        return proximity_reward * 0.01  # Scale down
    
    def _check_episode_end(self) -> bool:
        """Check if episode should end."""
        # End episode after certain number of steps or collision
        if self.step_count > 1000:
            return True
        
        # Check collision
        dx = self.player.position["x"] - self.enemy.pos["x"]
        dy = self.player.position["y"] - self.enemy.pos["y"]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 60:  # Collision threshold
            self.episode_reward += 50.0  # Bonus for catching player
            return True
        
        return False
    
    def _end_episode(self):
        """End current episode and start new one."""
        self.episode_count += 1
        self.total_reward += self.episode_reward
        
        # Update metrics
        self.metrics["episodes"] = self.episode_count
        self.metrics["total_steps"] = self.step_count
        self.metrics["avg_reward"] = self.total_reward / max(1, self.episode_count)
        
        logging.info(f"Episode {self.episode_count} ended. Reward: {self.episode_reward:.2f}")
        
        self._reset_episode()
    
    def _reset_episode(self):
        """Reset for new episode."""
        import random
        
        # Reset positions
        self.player.reset()
        
        # Place enemy randomly
        x = random.randint(0, self.screen_width - self.enemy.size)
        y = random.randint(0, self.screen_height - self.enemy.size)
        self.enemy.set_position(x, y)
        
        # Reset episode tracking
        self.episode_reward = 0.0
        self.step_count = 0
    
    def render(self, screen: pygame.Surface):
        """Render the RL training mode."""
        # Draw entities
        self.player.draw(screen)
        self.enemy.draw(screen)
        
        # Draw training UI
        font = pygame.font.Font(None, 24)
        y_offset = 10
        
        # Title
        title = pygame.font.Font(None, 36).render("RL TRAINING MODE", True, (255, 255, 255))
        screen.blit(title, (10, y_offset))
        y_offset += 40
        
        # Training status
        status_color = (0, 255, 0) if self.training_active else (255, 0, 0)
        status_text = "TRAINING" if self.training_active else "PAUSED"
        status = font.render(f"Status: {status_text}", True, status_color)
        screen.blit(status, (10, y_offset))
        y_offset += 25
        
        # Metrics
        metrics_text = [
            f"Episode: {self.metrics['episodes']}",
            f"Steps: {self.metrics['total_steps']}",
            f"Avg Reward: {self.metrics['avg_reward']:.2f}",
            f"Current Reward: {self.episode_reward:.2f}",
        ]
        
        for text in metrics_text:
            rendered = font.render(text, True, (255, 255, 255))
            screen.blit(rendered, (10, y_offset))
            y_offset += 20
        
        # Controls
        y_offset += 10
        controls = [
            "SPACE - Shoot missile",
            "R - Reset episode", 
            "T - Toggle training",
            "ESC - Return to menu"
        ]
        
        for control in controls:
            rendered = font.render(control, True, (200, 200, 200))
            screen.blit(rendered, (10, y_offset))
            y_offset += 18