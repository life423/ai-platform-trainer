"""
Adaptive Staged Enemy AI

This module provides an enemy with escalating difficulty through scripted behavior stages.
The AI starts with basic behavior and progressively becomes more challenging using
pre-defined algorithms (not machine learning).
"""
import logging
import math
import random
import numpy as np
import pygame
from typing import Dict, Optional, List, Tuple

# Note: This module uses scripted behavior stages, not machine learning
# The stable_baselines3 imports are kept for potential future RL implementation


class AdaptiveStagedEnemyAI:
    """
    Adaptive enemy AI with escalating difficulty stages.
    
    This enemy uses scripted behavior stages that become progressively more
    challenging as time progresses. It does NOT use machine learning, but rather
    pre-defined algorithms for each difficulty stage.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize the adaptive staged AI enemy."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (178, 34, 34)  # Fire brick red
        self.pos = {"x": float(screen_width // 2), "y": float(screen_height // 2)}
        self.speed = 3.0  # Start slower, will get faster as it learns
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000
        
        # Staged difficulty system (not machine learning)
        self.difficulty_level = 0.0  # 0.0 to 1.0 scale
        self.behavior_step = 0
        
        # Performance tracking
        self.hits_on_player = 0
        self.times_hit_by_missile = 0
        self.total_frames = 0
        self.last_performance_update = 0
        
        # Behavior evolution stages - scripted difficulty progression
        self.behavior_stage = "basic"  # basic -> hunting -> predator -> nightmare
        self.stage_thresholds = {
            "basic": 60,       # First 1 second: basic chase  
            "hunting": 180,    # Next 2 seconds: active hunting
            "predator": 300,   # Next 2 seconds: smart predator
            "nightmare": float('inf')  # Beyond 5 seconds: nightmare mode
        }
        
        logging.info("Adaptive Staged Enemy AI initialized (scripted behavior system)")
    
    
    def update_movement(self, player_x: float, player_y: float, 
                       player_step: int, current_time: int, 
                       missiles: List = None) -> None:
        """
        Update enemy movement using staged difficulty progression.
        
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
        
        # Update behavior analysis (for debugging/stats)
        self._update_behavior_analysis(player_x, player_y, missiles or [])
        
        # Update fade-in effect if active
        if self.fading_in:
            self.update_fade_in(current_time)
    
    def _update_behavior_stage(self):
        """Update the AI's behavior stage based on experience."""
        old_stage = self.behavior_stage
        
        if self.total_frames < self.stage_thresholds["basic"]:
            self.behavior_stage = "basic"
            self.difficulty_level = 0.2 + (self.total_frames / 60) * 0.3  # 0.2 to 0.5 in 1 second
            self.speed = 2.5 + (self.total_frames / 60) * 1.5  # 2.5 to 4.0 in 1 second
        elif self.total_frames < self.stage_thresholds["hunting"]:
            self.behavior_stage = "hunting"
            progress = (self.total_frames - 60) / 120  # 0 to 1 over 2 seconds
            self.difficulty_level = 0.5 + progress * 0.3  # 0.5 to 0.8
            self.speed = 4.0 + progress * 2.0  # 4.0 to 6.0
        elif self.total_frames < self.stage_thresholds["predator"]:
            self.behavior_stage = "predator"
            progress = (self.total_frames - 180) / 120  # 0 to 1 over 2 seconds
            self.difficulty_level = 0.8 + progress * 0.15  # 0.8 to 0.95
            self.speed = 6.0 + progress * 1.5  # 6.0 to 7.5
        else:
            self.behavior_stage = "nightmare"
            # Nightmare mode gets progressively more dangerous
            nightmare_progress = min(1.0, (self.total_frames - 300) / 600)  # Cap progression
            self.difficulty_level = 0.95 + nightmare_progress * 0.05  # 0.95 to 1.0
            self.speed = 7.5 + nightmare_progress * 2.5  # 7.5 to 10.0 (very fast!)
        
        if old_stage != self.behavior_stage:
            logging.info(f"AI evolved from {old_stage} to {self.behavior_stage} (difficulty: {self.difficulty_level:.2f})")
    
    def _get_movement_decision(self, player_x: float, player_y: float, 
                              missiles: List) -> Tuple[float, float]:
        """Get movement decision based on current AI stage."""
        
        if self.behavior_stage == "basic":
            return self._basic_movement(player_x, player_y)
        elif self.behavior_stage == "hunting":
            return self._hunting_movement(player_x, player_y, missiles)
        elif self.behavior_stage == "predator":
            return self._predator_movement(player_x, player_y, missiles)
        else:  # nightmare
            return self._nightmare_movement(player_x, player_y, missiles)
    
    def _basic_movement(self, player_x: float, player_y: float) -> Tuple[float, float]:
        """Basic stage - mix of random and basic chase."""
        # Mix random movement with increasing chase behavior
        progress = self.total_frames / 60.0  # 0 to 1 over first 60 frames
        
        # Random component
        angle = random.uniform(0, 2 * math.pi)
        random_x = math.cos(angle) * self.speed * 0.5
        random_y = math.sin(angle) * self.speed * 0.5
        
        # Chase component  
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            chase_x = (dx / distance) * self.speed
            chase_y = (dy / distance) * self.speed
        else:
            chase_x, chase_y = 0.0, 0.0
        
        # Blend random and chase based on progress
        move_x = (1 - progress) * random_x + progress * chase_x
        move_y = (1 - progress) * random_y + progress * chase_y
        
        return move_x, move_y
    
    def _hunting_movement(self, player_x: float, player_y: float, missiles: List) -> Tuple[float, float]:
        """Hunting stage - aggressive chase with basic missile awareness."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            # Aggressive direct chase
            chase_x = (dx / distance) * self.speed * 1.2  # 20% speed boost
            chase_y = (dy / distance) * self.speed * 1.2
            
            # Basic missile avoidance
            avoid_x, avoid_y = self._calculate_missile_avoidance(missiles)
            
            # Weighted combination - mostly chase
            move_x = 0.8 * chase_x + 0.2 * avoid_x
            move_y = 0.8 * chase_y + 0.2 * avoid_y
            
            return move_x, move_y
        return 0.0, 0.0
    
    def _predator_movement(self, player_x: float, player_y: float, 
                          missiles: List) -> Tuple[float, float]:
        """Predator stage - smart hunting with prediction and advanced avoidance."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            # Player movement prediction
            predicted_x = player_x + dx * 0.15  # Better prediction
            predicted_y = player_y + dy * 0.15
            
            # Calculate movement to predicted position
            pred_dx = predicted_x - self.pos["x"]
            pred_dy = predicted_y - self.pos["y"]
            pred_distance = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy)
            
            if pred_distance > 1:
                chase_x = (pred_dx / pred_distance) * self.speed * 1.1
                chase_y = (pred_dy / pred_distance) * self.speed * 1.1
            else:
                chase_x, chase_y = 0.0, 0.0
            
            # Advanced missile avoidance
            avoid_x, avoid_y = self._calculate_missile_avoidance(missiles, advanced=True)
            
            # Strategic positioning
            strategy_x, strategy_y = self._calculate_strategic_movement(player_x, player_y)
            
            # Weighted combination - balanced
            move_x = 0.6 * chase_x + 0.3 * avoid_x + 0.1 * strategy_x
            move_y = 0.6 * chase_y + 0.3 * avoid_y + 0.1 * strategy_y
            
            return move_x, move_y
        return 0.0, 0.0
    
    def _nightmare_movement(self, player_x: float, player_y: float, 
                           missiles: List) -> Tuple[float, float]:
        """Nightmare mode - extremely intelligent and aggressive AI."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            # Advanced player prediction with velocity estimation
            prediction_strength = 0.25 + (self.total_frames - 300) / 1200  # Increases over time
            predicted_x = player_x + dx * prediction_strength
            predicted_y = player_y + dy * prediction_strength
            
            # Calculate movement to predicted position
            pred_dx = predicted_x - self.pos["x"]
            pred_dy = predicted_y - self.pos["y"]
            pred_distance = math.sqrt(pred_dx * pred_dx + pred_dy * pred_dy)
            
            if pred_distance > 1:
                chase_x = (pred_dx / pred_distance) * self.speed * 1.3  # 30% speed boost
                chase_y = (pred_dy / pred_distance) * self.speed * 1.3
            else:
                chase_x, chase_y = 0.0, 0.0
            
            # Superior missile avoidance
            avoid_x, avoid_y = self._calculate_nightmare_avoidance(missiles)
            
            # Aggressive strategic positioning
            strategy_x, strategy_y = self._calculate_aggressive_strategy(player_x, player_y)
            
            # Add unpredictable element to make it more challenging
            chaos_factor = 0.1
            chaos_x = random.uniform(-1, 1) * self.speed * chaos_factor
            chaos_y = random.uniform(-1, 1) * self.speed * chaos_factor
            
            # Weighted combination - heavily favor aggression
            move_x = 0.5 * chase_x + 0.2 * avoid_x + 0.2 * strategy_x + 0.1 * chaos_x
            move_y = 0.5 * chase_y + 0.2 * avoid_y + 0.2 * strategy_y + 0.1 * chaos_y
            
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
    
    def _calculate_nightmare_avoidance(self, missiles: List) -> Tuple[float, float]:
        """Calculate superior missile avoidance for nightmare mode."""
        if not missiles:
            return 0.0, 0.0
        
        avoid_x, avoid_y = 0.0, 0.0
        
        for missile in missiles:
            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]
            
            # Calculate distance and trajectory
            dx = missile_x - self.pos["x"]
            dy = missile_y - self.pos["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Increased avoidance range for nightmare mode
            if distance < 150:  # Larger avoidance radius
                if distance > 1:
                    # Stronger avoidance force
                    force = min(200 / distance, 15.0)  # Increased force
                    avoid_x -= (dx / distance) * force
                    avoid_y -= (dy / distance) * force
        
        # Normalize and apply stronger avoidance
        avoid_distance = math.sqrt(avoid_x * avoid_x + avoid_y * avoid_y)
        if avoid_distance > 1:
            avoid_x = (avoid_x / avoid_distance) * self.speed * 0.8  # Stronger avoidance
            avoid_y = (avoid_y / avoid_distance) * self.speed * 0.8
        
        return avoid_x, avoid_y
    
    def _calculate_aggressive_strategy(self, player_x: float, player_y: float) -> Tuple[float, float]:
        """Calculate aggressive strategic movement for nightmare mode."""
        # Try to cut off player escape routes
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # Calculate where player is relative to center
        player_from_center_x = player_x - center_x
        player_from_center_y = player_y - center_y
        
        # Try to move to intercept player's likely escape route
        strategy_x = player_from_center_x * 0.3  # Move to cut off escape
        strategy_y = player_from_center_y * 0.3
        
        # Add corner pressure - if player is near edge, pressure them
        edge_margin = 100
        if player_x < edge_margin:
            strategy_x += 2.0  # Push from left
        elif player_x > self.screen_width - edge_margin:
            strategy_x -= 2.0  # Push from right
            
        if player_y < edge_margin:
            strategy_y += 2.0  # Push from top
        elif player_y > self.screen_height - edge_margin:
            strategy_y -= 2.0  # Push from bottom
        
        return strategy_x, strategy_y
    
    def _update_behavior_analysis(self, player_x: float, player_y: float, missiles: List):
        """Update behavior analysis for performance tracking."""
        self.behavior_step += 1
        
        # Every 100 frames, evaluate performance and log stats
        if self.behavior_step % 100 == 0:
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
        self.pos["x"] = float(x)
        self.pos["y"] = float(y)

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