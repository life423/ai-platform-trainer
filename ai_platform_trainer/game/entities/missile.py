"""
Missile entity for AI Platform Trainer.

Simple missile with AI-guided trajectory capabilities.
"""
import pygame
import random
from typing import Dict


class Missile:
    """Missile entity with AI guidance capabilities."""
    
    def __init__(self, x: float, y: float, vx: float, vy: float):
        self.pos = {"x": x, "y": y}
        self.velocity = {"vx": vx, "vy": vy}
        self.size = 8
        self.color = (255, 255, 0)  # Yellow
        
        # Lifetime management
        self.birth_time = pygame.time.get_ticks()
        self.lifespan = random.randint(8000, 12000)  # 8-12 seconds
        
        # AI guidance (can be controlled by missile AI)
        self.ai_controlled = False
        self.target_pos = None
    
    def update(self):
        """Update missile position."""
        self.pos["x"] += self.velocity["vx"]
        self.pos["y"] += self.velocity["vy"]
    
    def set_ai_velocity(self, vx: float, vy: float):
        """Set velocity from AI controller."""
        if self.ai_controlled:
            self.velocity["vx"] = vx
            self.velocity["vy"] = vy
    
    def enable_ai_control(self, target_pos: Dict[str, float] = None):
        """Enable AI control of this missile."""
        self.ai_controlled = True
        self.target_pos = target_pos
    
    def disable_ai_control(self):
        """Disable AI control."""
        self.ai_controlled = False
        self.target_pos = None
    
    def draw(self, screen: pygame.Surface):
        """Draw the missile."""
        pygame.draw.circle(
            screen, self.color,
            (int(self.pos["x"]), int(self.pos["y"])),
            self.size // 2
        )