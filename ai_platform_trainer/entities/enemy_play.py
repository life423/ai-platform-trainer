"""
Enemy entity for play mode.

This module defines the enemy entity used in play mode, which uses AI models
for movement decisions.
"""
import logging
import math
import random
from typing import Dict, Optional, Tuple, Any

import pygame
import torch
import numpy as np

try:
    import stable_baselines3
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL features will be disabled.")

from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel


class EnemyPlay:
    """
    Enemy entity for play mode with AI-controlled movement.
    
    This class represents the enemy in play mode, which can be controlled
    by either a neural network or a reinforcement learning model.
    """
    
    def __init__(
        self, 
        screen_width: int, 
        screen_height: int, 
        model: EnemyMovementModel
    ) -> None:
        """
        Initialize the enemy entity.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            model: Neural network model for enemy movement
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.speed = 5.0
        self.model = model
        self.rl_model = None
        self.use_rl = False
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000  # 1 second fade-in

    def update_movement(
        self, 
        player_x: float, 
        player_y: float, 
        player_speed: float, 
        current_time: int
    ) -> None:
        """
        Update the enemy's position based on AI model predictions.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
            current_time: Current game time in milliseconds
        """
        if not self.visible:
            return

        # TEMPORARY: Use simple chase behavior for debugging
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        
        # Normalize and scale movement
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 1:  # Avoid division by zero
            move_x = (dx / distance) * self.speed
            move_y = (dy / distance) * self.speed
            
            # Update position
            self.pos["x"] += move_x
            self.pos["y"] += move_y
            
            logging.debug(f"Enemy chasing: moving ({move_x:.2f}, {move_y:.2f}) toward player at ({player_x:.0f}, {player_y:.0f})")
            
            # Wrap around screen edges
            self._wrap_position()
        
        # OLD CODE:
        # if self.use_rl and self.rl_model:
        #     # Use reinforcement learning model if available
        #     self._update_with_rl(player_x, player_y, player_speed)
        # else:
        #     # Use neural network model
        #     self._update_with_nn(player_x, player_y, player_speed)

        # Update fade-in effect if active
        if self.fading_in:
            self.update_fade_in(current_time)

    def _update_with_nn(
        self, 
        player_x: float, 
        player_y: float, 
        player_speed: float
    ) -> None:
        """
        Update enemy position using the neural network model.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
        """
        # Calculate distance to player
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Normalize inputs for the model
        normalized_px = player_x / self.screen_width
        normalized_py = player_y / self.screen_height
        normalized_ex = self.pos["x"] / self.screen_width
        normalized_ey = self.pos["y"] / self.screen_height
        normalized_dist = distance / max(self.screen_width, self.screen_height)
        
        # Prepare input tensor for the model
        model_input = torch.tensor([
            normalized_px, normalized_py, 
            normalized_ex, normalized_ey, 
            normalized_dist
        ], dtype=torch.float32).unsqueeze(0)
        
        # Get model prediction
        with torch.no_grad():
            movement = self.model(model_input).squeeze(0)
            logging.debug(f"Neural network output: [{movement[0].item():.3f}, {movement[1].item():.3f}]")
            
        # Apply movement (scale from [-1,1] to actual pixels)
        move_x = movement[0].item() * self.speed
        move_y = movement[1].item() * self.speed
        logging.debug(f"Scaled movement: dx={move_x:.3f}, dy={move_y:.3f}")
        
        # Debug: Log enemy movement
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            logging.debug(f"Enemy moving: dx={move_x:.2f}, dy={move_y:.2f}, toward player at ({player_x:.0f},{player_y:.0f})")
        
        # Update position
        old_x, old_y = self.pos["x"], self.pos["y"]
        self.pos["x"] += move_x
        self.pos["y"] += move_y
        
        # Debug: Log position change
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            logging.debug(f"Enemy position: ({old_x:.0f},{old_y:.0f}) -> ({self.pos['x']:.0f},{self.pos['y']:.0f})")
        
        # Wrap around screen edges
        self._wrap_position()

    def _update_with_rl(
        self, 
        player_x: float, 
        player_y: float, 
        player_speed: float
    ) -> None:
        """
        Update enemy position using the reinforcement learning model.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_speed: Player's movement speed
        """
        # Calculate distance to player
        distance = math.sqrt(
            (player_x - self.pos["x"])**2 + 
            (player_y - self.pos["y"])**2
        )
        
        # Normalize values for the observation
        px = player_x / self.screen_width
        py = player_y / self.screen_height
        ex = self.pos["x"] / self.screen_width
        ey = self.pos["y"] / self.screen_height
        dist = distance / max(self.screen_width, self.screen_height)
        player_speed_norm = player_speed / 10.0
        time_factor = 0.5  # Placeholder for time since last hit
        
        # Create observation
        obs = np.array([
            px, py, ex, ey, dist, player_speed_norm, time_factor
        ], dtype=np.float32)
        
        # Get action from model
        if hasattr(self, 'rl_model') and self.rl_model:
            # Stable Baselines model
            action, _ = self.rl_model.predict(obs, deterministic=True)
        elif hasattr(self, 'policy_net') and self.policy_net:
            # PyTorch model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = self.policy_net(obs_tensor).squeeze().numpy()
        else:
            # Fallback to simple chase behavior
            dx = player_x - self.pos["x"]
            dy = player_y - self.pos["y"]
            norm = math.sqrt(dx*dx + dy*dy) + 1e-8
            action = np.array([dx/norm, dy/norm])
        
        # Apply action
        self.apply_rl_action(action)

    def apply_rl_action(self, action: np.ndarray) -> None:
        """
        Apply an action from the RL model.
        
        Args:
            action: Action array with values between -1 and 1
        """
        try:
            # Scale action to actual movement
            move_x = float(action[0]) * self.speed
            move_y = float(action[1]) * self.speed
            
            # Update position
            self.pos["x"] += move_x
            self.pos["y"] += move_y
            
            # Wrap around screen edges
            self._wrap_position()
        except Exception as e:
            logging.error(f"Error applying RL action: {e}")
            logging.error(f"Action: {action}, Type: {type(action)}")

    def _wrap_position(self) -> None:
        """Wrap the enemy position around screen edges."""
        if self.pos["x"] < -self.size:
            self.pos["x"] = self.screen_width
        elif self.pos["x"] > self.screen_width:
            self.pos["x"] = -self.size
            
        if self.pos["y"] < -self.size:
            self.pos["y"] = self.screen_height
        elif self.pos["y"] > self.screen_height:
            self.pos["y"] = -self.size

    def set_position(self, x: float, y: float) -> None:
        """
        Set the enemy position.
        
        Args:
            x: New x position
            y: New y position
        """
        self.pos["x"] = x
        self.pos["y"] = y

    def hide(self) -> None:
        """Hide the enemy (e.g., after being hit)."""
        self.visible = False

    def show(self, current_time: int) -> None:
        """
        Show the enemy with a fade-in effect.
        
        Args:
            current_time: Current game time in milliseconds
        """
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time

    def update_fade_in(self, current_time: int) -> None:
        """
        Update the fade-in effect.
        
        Args:
            current_time: Current game time in milliseconds
        """
        if not self.fading_in:
            return
            
        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)
        
        self.fade_alpha = int(255 * progress)
        
        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the enemy on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        if not self.visible:
            return
            
        if self.fading_in:
            # Create a surface with per-pixel alpha
            s = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            color_with_alpha = (*self.color, self.fade_alpha)
            pygame.draw.rect(s, color_with_alpha, (0, 0, self.size, self.size))
            screen.blit(s, (self.pos["x"], self.pos["y"]))
        else:
            pygame.draw.rect(
                screen,
                self.color,
                (self.pos["x"], self.pos["y"], self.size, self.size)
            )

    def load_rl_model(self, model_path: str) -> bool:
        """
        Load a reinforcement learning model.
        
        Args:
            model_path: Path to the RL model file
            
        Returns:
            True if the model was loaded successfully, False otherwise
        """
        try:
            # Check if path ends with .zip (Stable Baselines) or .pth (PyTorch)
            if model_path.endswith('.zip'):
                # Stable Baselines model
                if not STABLE_BASELINES_AVAILABLE:
                    logging.warning("Cannot load RL model: stable_baselines3 not available")
                    return False
                    
                self.rl_model = PPO.load(model_path)
                self.use_rl = True
                logging.info(f"Successfully loaded Stable Baselines RL model from {model_path}")
                return True
            elif model_path.endswith('.pth'):
                # PyTorch model
                from ai_platform_trainer.ai.models.policy_network import PolicyNetwork
                
                self.policy_net = PolicyNetwork(input_size=7, hidden_size=64, output_size=2)
                success = self.policy_net.load(model_path)
                if success:
                    self.use_rl = True
                    self.rl_model = None  # Not using Stable Baselines
                    logging.info(f"Successfully loaded PyTorch RL model from {model_path}")
                    return True
                else:
                    logging.error(f"Failed to load PyTorch model from {model_path}")
                    return False
            else:
                logging.error(f"Unknown model format: {model_path}")
                return False
        except Exception as e:
            logging.error(f"Failed to load RL model: {e}")
            self.rl_model = None
            self.policy_net = None
            self.use_rl = False
            return False