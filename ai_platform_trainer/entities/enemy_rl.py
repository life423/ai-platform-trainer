"""
True RL-Trained Enemy Entity

This entity uses a real reinforcement learning model to make movement decisions,
replacing the scripted adaptive behavior with genuine learned intelligence.
"""
import logging
import math
import numpy as np
import pygame
from typing import Dict, List, Optional, Any, Tuple
import torch

try:
    from stable_baselines3 import SAC
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL enemy disabled.")

from ai_platform_trainer.entities.enemy_play import EnemyPlay


class EnemyRL:
    """
    True RL-trained enemy that learns optimal evasion and survival strategies.
    
    This enemy uses a trained SAC model to make real-time movement decisions
    based on the current game state, player position, and missile threats.
    """
    
    def __init__(self, screen_width: int, screen_height: int, 
                 rl_model_path: Optional[str] = None,
                 fallback_to_scripted: bool = True):
        """
        Initialize RL-trained enemy.
        
        Args:
            screen_width: Game screen width
            screen_height: Game screen height
            rl_model_path: Path to trained RL model
            fallback_to_scripted: Whether to fall back to scripted behavior if model loading fails
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 20
        
        # Position and movement
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.velocity = {"x": 0.0, "y": 0.0}
        self.max_speed = 4.0
        self.max_acceleration = 2.0
        
        # RL model
        self.rl_model = None
        self.rl_available = False
        self.fallback_enemy = None
        
        # Load RL model
        if rl_model_path:
            self._load_rl_model(rl_model_path)
        
        # Fallback to scripted behavior if needed
        if not self.rl_available and fallback_to_scripted:
            from ai_platform_trainer.entities.enemy_learning import AdaptiveStagedEnemyAI
            self.fallback_enemy = AdaptiveStagedEnemyAI(screen_width, screen_height)
            logging.info("RL enemy falling back to scripted behavior")
        
        # Game state tracking
        self.visible = True
        self.energy = 100.0
        self.last_missile_positions = []  # Track recent missile positions
        self.survival_time = 0
        self.missiles_dodged = 0
        
        # Visual properties
        self.color = (255, 100, 100)  # Red
        self.alpha = 255
        self.fading_in = False
        self.fade_start_time = 0
        
        logging.info("🤖 RL Enemy initialized - learning evasion and survival strategies")
    
    def _load_rl_model(self, model_path: str):
        """Load the trained RL model."""
        if not STABLE_BASELINES_AVAILABLE:
            logging.warning("stable_baselines3 not available for RL enemy")
            return
        
        try:
            self.rl_model = SAC.load(model_path)
            self.rl_available = True
            logging.info(f"✅ Loaded RL enemy model from {model_path}")
        except Exception as e:
            logging.warning(f"Failed to load RL enemy model: {e}")
            self.rl_available = False
    
    def update_movement(self, player_x: float, player_y: float, 
                       player_step: int, current_time: int,
                       missiles: Optional[List] = None):
        """
        Update enemy movement using RL model or fallback behavior.
        
        Args:
            player_x: Player x position
            player_y: Player y position
            player_step: Player step count (for compatibility)
            current_time: Current game time
            missiles: List of active missiles
        """
        if not self.visible:
            return
        
        self.survival_time += 1
        
        if self.rl_available and self.rl_model:
            self._update_with_rl(player_x, player_y, missiles or [])
        elif self.fallback_enemy:
            # Use scripted fallback
            self.fallback_enemy.update_movement(player_x, player_y, player_step, current_time, missiles)
            # Copy position from fallback
            self.pos = self.fallback_enemy.pos.copy()
        else:
            # Basic evasion if no other options
            self._basic_evasion(player_x, player_y, missiles or [])
        
        # Update energy
        self._update_energy()
        
        # Track missile evasion
        self._track_missile_evasion(missiles or [])
    
    def _update_with_rl(self, player_x: float, player_y: float, missiles: List):
        """Update movement using RL model."""
        # Create observation for RL model
        observation = self._create_rl_observation(player_x, player_y, missiles)
        
        try:
            # Get action from RL model
            action, _ = self.rl_model.predict(observation, deterministic=True)
            
            # Apply action (acceleration in x, y)
            accel_x = action[0] * self.max_acceleration
            accel_y = action[1] * self.max_acceleration
            
            # Update velocity
            self.velocity["x"] += accel_x
            self.velocity["y"] += accel_y
            
            # Apply speed limit
            speed = math.sqrt(self.velocity["x"]**2 + self.velocity["y"]**2)
            if speed > self.max_speed:
                self.velocity["x"] = (self.velocity["x"] / speed) * self.max_speed
                self.velocity["y"] = (self.velocity["y"] / speed) * self.max_speed
            
            # Update position
            self.pos["x"] += self.velocity["x"]
            self.pos["y"] += self.velocity["y"]
            
            # Keep in bounds with soft boundary
            self._enforce_boundaries()
            
            # Update energy based on action intensity
            action_intensity = math.sqrt(accel_x**2 + accel_y**2)
            energy_cost = action_intensity * 0.5
            self.energy = max(0, self.energy - energy_cost)
            
        except Exception as e:
            logging.error(f"Error in RL enemy update: {e}")
            # Fall back to basic evasion
            self._basic_evasion(player_x, player_y, missiles)
    
    def _create_rl_observation(self, player_x: float, player_y: float, 
                              missiles: List) -> np.ndarray:
        """Create observation vector for RL model."""
        obs = []
        
        # Enemy state (normalized)
        obs.extend([
            self.pos["x"] / self.screen_width * 2 - 1,
            self.pos["y"] / self.screen_height * 2 - 1,
            self.velocity["x"] / self.max_speed,
            self.velocity["y"] / self.max_speed
        ])
        
        # Player state (normalized)
        obs.extend([
            player_x / self.screen_width * 2 - 1,
            player_y / self.screen_height * 2 - 1,
            0,  # Player vx (not available)
            0   # Player vy (not available)
        ])
        
        # Missile states (up to 3 missiles)
        max_missiles = 3
        missiles_added = 0
        
        for missile in missiles[:max_missiles]:
            if hasattr(missile, 'pos') and hasattr(missile, 'vx'):
                obs.extend([
                    missile.pos["x"] / self.screen_width * 2 - 1,
                    missile.pos["y"] / self.screen_height * 2 - 1,
                    missile.vx / 10.0,
                    missile.vy / 10.0,
                    1.0  # Active
                ])
                missiles_added += 1
        
        # Pad with zeros for missing missiles
        for _ in range(max_missiles - missiles_added):
            obs.extend([0, 0, 0, 0, 0])  # Inactive missile
        
        # Meta information
        obs.extend([
            0.5,  # Time since last missile (placeholder)
            self.energy / 100.0,  # Normalized energy
            min(self.pos["x"], self.screen_width - self.pos["x"]) / self.screen_width,
            min(self.pos["y"], self.screen_height - self.pos["y"]) / self.screen_height
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _basic_evasion(self, player_x: float, player_y: float, missiles: List):
        """Basic evasion behavior as fallback."""
        target_vx, target_vy = 0, 0
        
        # Evade missiles
        for missile in missiles:
            if hasattr(missile, 'pos'):
                dx = self.pos["x"] - missile.pos["x"]
                dy = self.pos["y"] - missile.pos["y"]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance < 100 and distance > 0:
                    # Move away from missile
                    evasion_strength = (100 - distance) / 100 * 3.0
                    target_vx += (dx / distance) * evasion_strength
                    target_vy += (dy / distance) * evasion_strength
        
        # Move towards center if near edges
        center_x, center_y = self.screen_width // 2, self.screen_height // 2
        edge_margin = 80
        
        if (self.pos["x"] < edge_margin or self.pos["x"] > self.screen_width - edge_margin or
            self.pos["y"] < edge_margin or self.pos["y"] > self.screen_height - edge_margin):
            dx = center_x - self.pos["x"]
            dy = center_y - self.pos["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            if distance > 0:
                target_vx += (dx / distance) * 1.5
                target_vy += (dy / distance) * 1.5
        
        # Apply movement
        self.velocity["x"] = np.clip(target_vx, -self.max_speed, self.max_speed)
        self.velocity["y"] = np.clip(target_vy, -self.max_speed, self.max_speed)
        
        self.pos["x"] += self.velocity["x"]
        self.pos["y"] += self.velocity["y"]
        
        self._enforce_boundaries()
    
    def _enforce_boundaries(self):
        """Keep enemy within screen boundaries."""
        margin = 30
        
        if self.pos["x"] < margin:
            self.pos["x"] = margin
            self.velocity["x"] = max(0, self.velocity["x"])
        elif self.pos["x"] > self.screen_width - margin:
            self.pos["x"] = self.screen_width - margin
            self.velocity["x"] = min(0, self.velocity["x"])
        
        if self.pos["y"] < margin:
            self.pos["y"] = margin
            self.velocity["y"] = max(0, self.velocity["y"])
        elif self.pos["y"] > self.screen_height - margin:
            self.pos["y"] = self.screen_height - margin
            self.velocity["y"] = min(0, self.velocity["y"])
    
    def _update_energy(self):
        """Update energy levels."""
        # Slowly regenerate energy
        self.energy = min(100.0, self.energy + 0.1)
    
    def _track_missile_evasion(self, missiles: List):
        """Track missile evasion for statistics."""
        current_missile_positions = []
        for missile in missiles:
            if hasattr(missile, 'pos'):
                current_missile_positions.append((missile.pos["x"], missile.pos["y"]))
        
        # Check if any missiles from last frame are gone (likely evaded)
        if len(self.last_missile_positions) > len(current_missile_positions):
            missiles_lost = len(self.last_missile_positions) - len(current_missile_positions)
            self.missiles_dodged += missiles_lost
        
        self.last_missile_positions = current_missile_positions
    
    def set_position(self, x: float, y: float):
        """Set enemy position."""
        self.pos["x"] = x
        self.pos["y"] = y
    
    def hide(self):
        """Hide the enemy."""
        self.visible = False
    
    def show(self):
        """Show the enemy."""
        self.visible = True
    
    def start_fade_in(self, current_time: int):
        """Start fade-in animation."""
        self.fading_in = True
        self.fade_start_time = current_time
        self.alpha = 0
        self.visible = True
    
    def update_fade_in(self, current_time: int):
        """Update fade-in animation."""
        if not self.fading_in:
            return
        
        fade_duration = 1000  # 1 second
        elapsed = current_time - self.fade_start_time
        
        if elapsed >= fade_duration:
            self.alpha = 255
            self.fading_in = False
        else:
            self.alpha = int((elapsed / fade_duration) * 255)
    
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return pygame.Rect(
            self.pos["x"] - self.size // 2,
            self.pos["y"] - self.size // 2,
            self.size, self.size
        )
    
    def draw(self, screen: pygame.Surface):
        """Draw the enemy."""
        if not self.visible:
            return
        
        # Create surface with alpha for transparency
        enemy_surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        
        # Color intensity based on energy and RL status
        if self.rl_available:
            # RL enemy has a distinctive color
            color = (100, 255, 100, self.alpha)  # Green for RL
        else:
            # Fallback enemy
            color = (255, 100, 100, self.alpha)  # Red for scripted
        
        pygame.draw.ellipse(enemy_surface, color, 
                          (0, 0, self.size, self.size))
        
        # Add energy indicator
        if self.energy < 50:
            energy_color = (255, 255, 0, self.alpha // 2)  # Yellow for low energy
            pygame.draw.circle(enemy_surface, energy_color,
                             (self.size // 2, self.size // 2),
                             self.size // 4)
        
        # Draw RL indicator
        if self.rl_available:
            # Small indicator that this is an RL enemy
            indicator_color = (0, 255, 255, self.alpha)  # Cyan indicator
            pygame.draw.circle(enemy_surface, indicator_color,
                             (self.size - 5, 5), 3)
        
        screen.blit(enemy_surface, 
                   (self.pos["x"] - self.size // 2, 
                    self.pos["y"] - self.size // 2))
    
    def get_ai_info(self) -> str:
        """Get information about the AI being used."""
        if self.rl_available:
            return f"True RL Enemy (SAC) - Survival: {self.survival_time}, Dodged: {self.missiles_dodged}"
        elif self.fallback_enemy:
            return f"Scripted Fallback - Survival: {self.survival_time}"
        else:
            return f"Basic Evasion - Survival: {self.survival_time}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        movement_efficiency = self.survival_time / max(1, self.survival_time * 0.1)  # Rough estimate
        
        return {
            "survival_time": self.survival_time,
            "missiles_dodged": self.missiles_dodged,
            "energy": self.energy,
            "rl_enabled": self.rl_available,
            "movement_efficiency": movement_efficiency,
            "position": self.pos.copy(),
            "velocity": self.velocity.copy()
        }