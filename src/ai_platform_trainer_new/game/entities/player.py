"""
Clean Player Entity

This module provides a simplified player entity class.
"""
import pygame
import math
from typing import List, Dict, Optional

from config.settings import settings
from .missile import Missile


class Player:
    """Player entity class."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = settings.PLAYER_SIZE
        self.speed = settings.PLAYER_SPEED
        
        # Position
        self.position = {
            "x": screen_width // 2,
            "y": screen_height // 2
        }
        
        # Missiles
        self.missiles: List[Missile] = []
        self.step = 0
    
    def update(self, enemy_x: float = 0, enemy_y: float = 0):
        """Update player state."""
        self.handle_input()
        self.update_missiles()
        self.step += 1
    
    def handle_input(self):
        """Handle player input."""
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] = max(0, self.position["x"] - self.speed)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] = min(self.screen_width - self.size, self.position["x"] + self.speed)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] = max(0, self.position["y"] - self.speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] = min(self.screen_height - self.size, self.position["y"] + self.speed)
    
    def shoot_missile(self, target_pos: Optional[Dict[str, float]] = None):
        """Shoot a missile towards target position."""
        if len(self.missiles) >= 3:  # Limit missiles
            return
            
        # Calculate missile direction
        if target_pos:
            dx = target_pos["x"] - self.position["x"]
            dy = target_pos["y"] - self.position["y"]
        else:
            dx, dy = 0, -1  # Default upward
            
        # Normalize direction
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > 0:
            vx = (dx / distance) * settings.MISSILE_SPEED
            vy = (dy / distance) * settings.MISSILE_SPEED
        else:
            vx, vy = 0, -settings.MISSILE_SPEED
        
        # Create missile
        missile = Missile(
            self.position["x"] + self.size // 2,
            self.position["y"] + self.size // 2,
            vx, vy
        )
        self.missiles.append(missile)
    
    def update_missiles(self):
        """Update all missiles."""
        for missile in self.missiles[:]:
            missile.update()
            
            # Remove missiles that are off-screen
            if (missile.pos["x"] < 0 or missile.pos["x"] > self.screen_width or
                missile.pos["y"] < 0 or missile.pos["y"] > self.screen_height):
                self.missiles.remove(missile)
    
    def reset(self):
        """Reset player to initial state."""
        self.position = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2
        }
        self.missiles.clear()
        self.step = 0