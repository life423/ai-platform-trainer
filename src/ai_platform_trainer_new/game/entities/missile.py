"""
Clean Missile Entity

This module provides a simplified missile entity class.
"""
import pygame
import math

from config.settings import settings


class Missile:
    """Missile entity class."""
    
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.pos = {"x": x, "y": y}
        self.position = {"x": x, "y": y}  # For renderer compatibility
        self.vx = vx
        self.vy = vy
        self.size = settings.MISSILE_SIZE
        self.birth_time = pygame.time.get_ticks()
        self.lifespan = 5000  # 5 seconds
        
        # Direction for rendering (normalized)
        if vx != 0 or vy != 0:
            magnitude = math.sqrt(vx * vx + vy * vy)
            self.direction = (vx / magnitude, vy / magnitude)
        else:
            self.direction = (0, -1)
    
    def update(self):
        """Update missile position."""
        self.pos["x"] += self.vx
        self.pos["y"] += self.vy
        self.position["x"] = self.pos["x"]
        self.position["y"] = self.pos["y"]
    
    def get_rect(self) -> pygame.Rect:
        """Get missile rectangle for collision detection."""
        return pygame.Rect(self.pos["x"], self.pos["y"], self.size, self.size)