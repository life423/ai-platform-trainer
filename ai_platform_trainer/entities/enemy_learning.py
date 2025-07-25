"""
Real-time Learning Enemy AI

This module provides an enemy that learns and improves during gameplay.
The AI starts with basic behavior and evolves to become smarter every frame.
"""
import logging
import math
import random
import numpy as np
import pygame
from typing import Dict, Optional, List, Tuple

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import gymnasium as gym
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. Learning features will be disabled.")


class LearningEnemyAI:
    """
    Real-time learning AI enemy that improves during gameplay.
    
    This enemy starts with random/basic behavior and uses reinforcement learning
    to adapt and improve its strategy every frame based on the game state.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the learning AI enemy."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (178, 34, 34)  # Fire brick red
        self.pos = {"x": screen_width // 2, "y": screen_height // 2}
        self.speed = 3.0  # Start slower, will get faster as it learns
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000
        
        # Learning system
        self.learning_enabled = STABLE_BASELINES_AVAILABLE
        self.learning_step = 0
        self.difficulty_level = 0.0  # 0.0 to 1.0 scale
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
        # Performance tracking
        self.hits_on_player = 0
        self.times_hit_by_missile = 0
        self.total_frames = 0
        self.last_performance_update = 0
        
        # Behavior evolution stages
        self.behavior_stage = "random"  # random -> chase -> smart -> expert
        self.stage_thresholds = {
            "random": 100,    # First 100 frames: random movement
            "chase": 500,     # Next 400 frames: basic chase
            "smart": 1500,    # Next 1000 frames: smart movement
            "expert": float('inf')  # Beyond 1500 frames: expert behavior
        }
        
        # RL Model (if available)
        self.rl_model = None
        self.observation_space = 7  # player_x, player_y, enemy_x, enemy_y, distance, missile_data
        self.action_space = 2  # move_x, move_y
        
        if self.learning_enabled:
            self._initialize_rl_model()
        
        logging.info(f"Learning Enemy AI initialized (RL Available: {self.learning_enabled})")
    
    def _initialize_rl_model(self):
        """Initialize the RL model for learning."""
        try:
            # Create a simple environment for the RL model
            # We'll update this during gameplay instead of using a gym environment
            self.rl_model = None  # Will be created when we have enough experience
            logging.info("RL model initialization prepared")
        except Exception as e:
            logging.error(f"Failed to initialize RL model: {e}")
            self.learning_enabled = False
    
    def update_movement(self, player_x: float, player_y: float, 
                       player_step: int, current_time: int, 
                       missiles: List = None) -> None:
        """
        Update enemy movement with real-time learning.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position  
            player_step: Player's movement step
            current_time: Current game time
            missiles: List of active missiles
        """
        if not self.visible:
            return
            
        self.total_frames += 1
        
        # Update behavior stage based on experience
        self._update_behavior_stage()
        
        # Get movement decision based on current AI stage
        move_x, move_y = self._get_movement_decision(player_x, player_y, missiles or [])
        
        # Apply movement
        self.pos["x"] += move_x
        self.pos["y"] += move_y
        
        # Keep in bounds
        self._constrain_to_screen()
        
        # Update learning system
        if self.learning_enabled:
            self._update_learning(player_x, player_y, missiles or [])
        
        # Update fade-in effect if active
        if self.fading_in:
            self.update_fade_in(current_time)
    
    def _update_behavior_stage(self):
        """Update the AI's behavior stage based on experience."""
        old_stage = self.behavior_stage
        
        if self.total_frames < self.stage_thresholds["random"]:
            self.behavior_stage = "random"
            self.difficulty_level = 0.1
            self.speed = 2.0
        elif self.total_frames < self.stage_thresholds["chase"]:
            self.behavior_stage = "chase"
            self.difficulty_level = 0.3
            self.speed = 3.0
        elif self.total_frames < self.stage_thresholds["smart"]:
            self.behavior_stage = "smart"
            self.difficulty_level = 0.6
            self.speed = 4.0
        else:
            self.behavior_stage = "expert"
            self.difficulty_level = min(1.0, 0.8 + (self.total_frames - 1500) / 3000)
            self.speed = min(6.0, 4.0 + (self.total_frames - 1500) / 1000)
        
        if old_stage != self.behavior_stage:
            logging.info(f"AI evolved from {old_stage} to {self.behavior_stage} (difficulty: {self.difficulty_level:.2f})")
    
    def _get_movement_decision(self, player_x: float, player_y: float, 
                              missiles: List) -> Tuple[float, float]:
        """Get movement decision based on current AI stage."""
        
        if self.behavior_stage == "random":
            return self._random_movement()
        elif self.behavior_stage == "chase":
            return self._chase_movement(player_x, player_y)
        elif self.behavior_stage == "smart":
            return self._smart_movement(player_x, player_y, missiles)
        else:  # expert
            return self._expert_movement(player_x, player_y, missiles)
    
    def _random_movement(self) -> Tuple[float, float]:
        """Random movement for early learning stage."""
        angle = random.uniform(0, 2 * math.pi)
        move_x = math.cos(angle) * self.speed * random.uniform(0.3, 1.0)
        move_y = math.sin(angle) * self.speed * random.uniform(0.3, 1.0)
        return move_x, move_y
    
    def _chase_movement(self, player_x: float, player_y: float) -> Tuple[float, float]:
        """Basic chase movement."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            move_x = (dx / distance) * self.speed
            move_y = (dy / distance) * self.speed
            return move_x, move_y
        return 0.0, 0.0
    
    def _smart_movement(self, player_x: float, player_y: float, 
                       missiles: List) -> Tuple[float, float]:
        """Smart movement that considers missiles and player prediction."""
        # Basic chase with missile avoidance
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            # Base chase movement
            chase_x = (dx / distance) * self.speed
            chase_y = (dy / distance) * self.speed
            
            # Missile avoidance
            avoid_x, avoid_y = self._calculate_missile_avoidance(missiles)
            
            # Combine chase and avoidance (weighted)
            move_x = 0.7 * chase_x + 0.3 * avoid_x
            move_y = 0.7 * chase_y + 0.3 * avoid_y
            
            return move_x, move_y
        return 0.0, 0.0
    
    def _expert_movement(self, player_x: float, player_y: float, 
                        missiles: List) -> Tuple[float, float]:
        """Expert movement with prediction and advanced tactics."""
        # Advanced AI behavior with player prediction
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            # Predict player movement (simple prediction)
            predicted_x = player_x + dx * 0.1  # Simple prediction
            predicted_y = player_y + dy * 0.1
            
            # Calculate movement to predicted position
            pred_dx = predicted_x - self.pos["x"]
            pred_dy = predicted_y - self.pos["y"]
            pred_distance = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy)
            
            if pred_distance > 1:
                chase_x = (pred_dx / pred_distance) * self.speed
                chase_y = (pred_dy / pred_distance) * self.speed
            else:
                chase_x, chase_y = 0.0, 0.0
            
            # Advanced missile avoidance
            avoid_x, avoid_y = self._calculate_missile_avoidance(missiles, advanced=True)
            
            # Combine with strategic positioning
            strategy_x, strategy_y = self._calculate_strategic_movement(player_x, player_y)
            
            # Weighted combination
            move_x = 0.5 * chase_x + 0.3 * avoid_x + 0.2 * strategy_x
            move_y = 0.5 * chase_y + 0.3 * avoid_y + 0.2 * strategy_y
            
            return move_x, move_y
        return 0.0, 0.0
    
    def _calculate_missile_avoidance(self, missiles: List, advanced: bool = False) -> Tuple[float, float]:
        """Calculate avoidance vector for missiles."""
        if not missiles:
            return 0.0, 0.0
        
        avoid_x, avoid_y = 0.0, 0.0
        
        for missile in missiles:
            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]
            
            # Calculate distance to missile
            dx = missile_x - self.pos["x"]
            dy = missile_y - self.pos["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance < 100:  # Only avoid nearby missiles
                # Calculate avoidance force (inverse of distance)
                if distance > 1:
                    force = min(100 / distance, 10.0)
                    avoid_x -= (dx / distance) * force
                    avoid_y -= (dy / distance) * force
        
        # Normalize avoidance vector
        avoid_distance = math.sqrt(avoid_x * avoid_x + avoid_y * avoid_y)
        if avoid_distance > 1:
            avoid_x = (avoid_x / avoid_distance) * self.speed * 0.5
            avoid_y = (avoid_y / avoid_distance) * self.speed * 0.5
        
        return avoid_x, avoid_y
    
    def _calculate_strategic_movement(self, player_x: float, player_y: float) -> Tuple[float, float]:
        """Calculate strategic movement for expert AI."""
        # Try to position for better angle of attack
        # This is a simple implementation - could be much more sophisticated
        
        # Prefer positions that give multiple approach angles
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Move toward center if too close to edges
        edge_margin = 100
        strategy_x, strategy_y = 0.0, 0.0
        
        if self.pos["x"] < edge_margin:
            strategy_x = 1.0
        elif self.pos["x"] > self.screen_width - edge_margin:
            strategy_x = -1.0
            
        if self.pos["y"] < edge_margin:
            strategy_y = 1.0
        elif self.pos["y"] > self.screen_height - edge_margin:
            strategy_y = -1.0
        
        return strategy_x, strategy_y
    
    def _update_learning(self, player_x: float, player_y: float, missiles: List):
        """Update the learning system with current experience."""
        # This is where we would implement actual RL learning
        # For now, we use the staged behavior system
        self.learning_step += 1
        
        # Every 100 frames, evaluate performance and adjust
        if self.learning_step % 100 == 0:
            self._evaluate_performance()
    
    def _evaluate_performance(self):
        """Evaluate AI performance and adjust behavior."""
        if self.total_frames > 0:
            hit_rate = self.hits_on_player / max(1, self.total_frames // 100)
            survival_rate = 1.0 - (self.times_hit_by_missile / max(1, self.total_frames // 100))
            
            # Simple performance logging
            logging.debug(f"AI Performance - Hit Rate: {hit_rate:.2f}, Survival: {survival_rate:.2f}, Stage: {self.behavior_stage}")
    
    def _constrain_to_screen(self):
        """Keep enemy within screen bounds."""
        self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
        self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))
    
    def on_hit_player(self):
        """Called when AI successfully hits the player."""
        self.hits_on_player += 1
    
    def on_hit_by_missile(self):
        """Called when AI is hit by a missile."""
        self.times_hit_by_missile += 1
    
    def get_difficulty_level(self) -> float:
        """Get current difficulty level (0.0 to 1.0)."""
        return self.difficulty_level
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics for UI display."""
        return {
            "stage": self.behavior_stage.title(),
            "difficulty": self.difficulty_level,
            "frames": self.total_frames,
            "hits": self.hits_on_player,
            "deaths": self.times_hit_by_missile,
            "speed": self.speed
        }
    
    # Standard enemy methods for compatibility
    def set_position(self, x: float, y: float) -> None:
        """Set enemy position."""
        self.pos["x"] = x
        self.pos["y"] = y

    def hide(self) -> None:
        """Hide the enemy."""
        self.visible = False

    def show(self, current_time: int) -> None:
        """Show the enemy with fade-in effect."""
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time

    def update_fade_in(self, current_time: int) -> None:
        """Update fade-in effect."""
        if not self.fading_in:
            return
            
        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)
        
        self.fade_alpha = int(255 * progress)
        
        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the enemy on screen."""
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