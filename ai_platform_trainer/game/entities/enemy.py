"""
Enemy entity for AI Platform Trainer.

Unified enemy class that supports different AI modes and training.
"""
import pygame
import logging
import math
import random
import numpy as np
from typing import Dict, Optional, Any

try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel


class Enemy:
    """Enemy entity with multiple AI modes."""
    
    def __init__(self, screen_width: int, screen_height: int, mode: str = "play"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mode = mode
        
        # Visual properties
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.visible = True
        
        # Position and movement
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.speed = 5.0
        
        # AI models
        self.nn_model: Optional[EnemyMovementModel] = None
        self.rl_model: Optional[Any] = None
        self.ai_mode = "neural_network"  # "neural_network", "reinforcement_learning", "random"
        
        # Fade effects
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000
        
        # Training data
        self.training_data = [] if mode in ["supervised_training", "rl_training"] else None
        self.rl_agent = None
        
        if mode == "rl_training" and STABLE_BASELINES_AVAILABLE:
            self._init_rl_training()
    
    def _init_rl_training(self):
        """Initialize RL training components."""
        # This will be set up by the training mode
        pass
    
    def set_neural_network_model(self, model: EnemyMovementModel):
        """Set the neural network model for enemy AI."""
        self.nn_model = model
        self.ai_mode = "neural_network"
    
    def set_rl_model(self, model):
        """Set the reinforcement learning model."""
        self.rl_model = model
        self.ai_mode = "reinforcement_learning"
    
    def update_movement(self, player_x: float, player_y: float, player_speed: float, current_time: int):
        """Update enemy position based on AI."""
        if not self.visible:
            return
        
        if self.ai_mode == "neural_network" and self.nn_model:
            self._update_with_neural_network(player_x, player_y, player_speed)
        elif self.ai_mode == "reinforcement_learning" and self.rl_model:
            self._update_with_rl(player_x, player_y, player_speed)
        else:
            self._update_random(player_x, player_y)
        
        # Update fade effect
        if self.fading_in:
            self.update_fade_in(current_time)
        
        # Collect training data
        if self.training_data is not None:
            self.training_data.append({
                "enemy_pos": self.pos.copy(),
                "player_pos": {"x": player_x, "y": player_y},
                "player_speed": player_speed,
                "timestamp": current_time,
                "ai_mode": self.ai_mode
            })
    
    def _update_with_neural_network(self, player_x: float, player_y: float, player_speed: float):
        """Update using neural network model."""
        import torch
        
        # Calculate distance
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Normalize inputs
        normalized_px = player_x / self.screen_width
        normalized_py = player_y / self.screen_height
        normalized_ex = self.pos["x"] / self.screen_width
        normalized_ey = self.pos["y"] / self.screen_height
        normalized_dist = distance / max(self.screen_width, self.screen_height)
        
        # Get model prediction
        model_input = torch.tensor([
            normalized_px, normalized_py, normalized_ex, normalized_ey, normalized_dist
        ], dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            movement = self.nn_model(model_input).squeeze(0)
        
        # Apply movement
        move_x = movement[0].item() * self.speed
        move_y = movement[1].item() * self.speed
        
        self.pos["x"] += move_x
        self.pos["y"] += move_y
        self._wrap_position()
    
    def _update_with_rl(self, player_x: float, player_y: float, player_speed: float):
        """Update using reinforcement learning model."""
        # Calculate observation
        distance = math.sqrt((player_x - self.pos["x"])**2 + (player_y - self.pos["y"])**2)
        
        obs = np.array([
            player_x / self.screen_width,
            player_y / self.screen_height,
            self.pos["x"] / self.screen_width,
            self.pos["y"] / self.screen_height,
            distance / max(self.screen_width, self.screen_height),
            player_speed / 10.0,
            0.5  # Time factor placeholder
        ], dtype=np.float32)
        
        # Get action from RL model
        if hasattr(self.rl_model, 'predict'):
            action, _ = self.rl_model.predict(obs, deterministic=True)
        else:
            # Fallback to chase behavior
            dx = player_x - self.pos["x"]
            dy = player_y - self.pos["y"]
            norm = math.sqrt(dx*dx + dy*dy) + 1e-8
            action = np.array([dx/norm, dy/norm])
        
        # Apply action
        self.pos["x"] += float(action[0]) * self.speed
        self.pos["y"] += float(action[1]) * self.speed
        self._wrap_position()
    
    def _update_random(self, player_x: float, player_y: float):
        """Simple random movement toward player."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        
        # Normalize and add randomness
        length = math.sqrt(dx*dx + dy*dy) + 1e-8
        move_x = (dx / length) * self.speed * random.uniform(0.5, 1.5)
        move_y = (dy / length) * self.speed * random.uniform(0.5, 1.5)
        
        self.pos["x"] += move_x
        self.pos["y"] += move_y
        self._wrap_position()
    
    def _wrap_position(self):
        """Wrap enemy position around screen edges."""
        if self.pos["x"] < -self.size:
            self.pos["x"] = self.screen_width
        elif self.pos["x"] > self.screen_width:
            self.pos["x"] = -self.size
        
        if self.pos["y"] < -self.size:
            self.pos["y"] = self.screen_height
        elif self.pos["y"] > self.screen_height:
            self.pos["y"] = -self.size
    
    def set_position(self, x: float, y: float):
        """Set enemy position."""
        self.pos["x"] = x
        self.pos["y"] = y
    
    def hide(self):
        """Hide the enemy."""
        self.visible = False
    
    def show(self, current_time: int):
        """Show enemy with fade-in effect."""
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time
    
    def update_fade_in(self, current_time: int):
        """Update fade-in effect."""
        if not self.fading_in:
            return
        
        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)
        self.fade_alpha = int(255 * progress)
        
        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255
    
    def get_training_data(self):
        """Get collected training data."""
        return self.training_data.copy() if self.training_data else []
    
    def clear_training_data(self):
        """Clear training data."""
        if self.training_data:
            self.training_data.clear()
    
    def draw(self, screen: pygame.Surface):
        """Draw the enemy."""
        if not self.visible:
            return
        
        if self.fading_in:
            # Draw with alpha
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.fade_alpha)
            pygame.draw.rect(s, color_with_alpha, (0, 0, self.size, self.size))
            screen.blit(s, (self.pos["x"], self.pos["y"]))
        else:
            pygame.draw.rect(
                screen, self.color,
                (self.pos["x"], self.pos["y"], self.size, self.size)
            )