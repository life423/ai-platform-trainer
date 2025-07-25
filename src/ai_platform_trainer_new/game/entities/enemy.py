"""
Clean Enemy Entity

This module provides a simplified enemy entity class.
"""
import pygame
import math
import random

from config.settings import settings


class Enemy:
    """Enemy entity class."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = settings.ENEMY_SIZE
        self.speed = settings.ENEMY_SPEED
        
        # Position
        self.pos = {"x": 0, "y": 0}
        self.visible = True
        
        # Movement pattern
        self.pattern = "chase"
        self.pattern_timer = 0
        self.pattern_change_interval = 5000  # 5 seconds
        
        # Random walk state
        self.walk_direction = random.uniform(0, 2 * math.pi)
        self.walk_timer = 0
    
    def update_movement(self, player_x: float, player_y: float, player_step: int):
        """Update enemy movement."""
        if not self.visible:
            return
            
        current_time = pygame.time.get_ticks()
        
        # Change pattern occasionally
        if current_time - self.pattern_timer > self.pattern_change_interval:
            self.pattern = random.choice(["chase", "random_walk", "circle"])
            self.pattern_timer = current_time
        
        # Execute movement pattern
        if self.pattern == "chase":
            self._chase_player(player_x, player_y)
        elif self.pattern == "random_walk":
            self._random_walk()
        elif self.pattern == "circle":
            self._circle_movement(player_x, player_y)
    
    def _chase_player(self, player_x: float, player_y: float):
        """Chase the player."""
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            move_x = (dx / distance) * self.speed
            move_y = (dy / distance) * self.speed
            
            self.pos["x"] += move_x
            self.pos["y"] += move_y
            
            # Keep in bounds
            self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
            self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))
    
    def _random_walk(self):
        """Random walk movement."""
        current_time = pygame.time.get_ticks()
        
        # Change direction occasionally
        if current_time - self.walk_timer > 1000:  # 1 second
            self.walk_direction = random.uniform(0, 2 * math.pi)
            self.walk_timer = current_time
        
        # Move in current direction
        move_x = math.cos(self.walk_direction) * self.speed
        move_y = math.sin(self.walk_direction) * self.speed
        
        new_x = self.pos["x"] + move_x
        new_y = self.pos["y"] + move_y
        
        # Bounce off walls
        if new_x < 0 or new_x > self.screen_width - self.size:
            self.walk_direction = math.pi - self.walk_direction
        if new_y < 0 or new_y > self.screen_height - self.size:
            self.walk_direction = -self.walk_direction
        
        self.pos["x"] = max(0, min(self.screen_width - self.size, new_x))
        self.pos["y"] = max(0, min(self.screen_height - self.size, new_y))
    
    def _circle_movement(self, player_x: float, player_y: float):
        """Circle around the player."""
        center_x = player_x
        center_y = player_y
        radius = 150
        
        current_time = pygame.time.get_ticks()
        angle = (current_time / 1000.0) * 2  # 2 radians per second
        
        target_x = center_x + math.cos(angle) * radius
        target_y = center_y + math.sin(angle) * radius
        
        # Move towards target position
        dx = target_x - self.pos["x"]
        dy = target_y - self.pos["y"]
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 1:
            move_x = (dx / distance) * self.speed
            move_y = (dy / distance) * self.speed
            
            self.pos["x"] += move_x
            self.pos["y"] += move_y
            
            # Keep in bounds
            self.pos["x"] = max(0, min(self.screen_width - self.size, self.pos["x"]))
            self.pos["y"] = max(0, min(self.screen_height - self.size, self.pos["y"]))
    
    def set_position(self, x: float, y: float):
        """Set enemy position."""
        self.pos["x"] = x
        self.pos["y"] = y
    
    def hide(self):
        """Hide the enemy."""
        self.visible = False
    
    def show(self, current_time: int):
        """Show the enemy."""
        self.visible = True