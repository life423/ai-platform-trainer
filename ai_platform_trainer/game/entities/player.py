"""
Player entity for AI Platform Trainer.

Unified player class that handles both play and training modes.
"""
import pygame
import logging
import random
import math
from typing import List, Dict, Optional
from ai_platform_trainer.game.entities.missile import Missile


class Player:
    """Player entity that can operate in different modes."""
    
    def __init__(self, screen_width: int, screen_height: int, mode: str = "play"):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.mode = mode
        
        # Visual properties
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        
        # Position and movement
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        
        # Missiles
        self.missiles: List[Missile] = []
        self.missile_cooldown = 500
        self.last_missile_time = 0
        
        # Training data collection
        self.training_data = [] if mode == "training" else None
    
    def reset(self):
        """Reset player to initial state."""
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()
        self.last_missile_time = 0
        if self.training_data:
            self.training_data.clear()
    
    def handle_input(self) -> bool:
        """Handle player input. Returns False if quit requested."""
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step
        
        # Screen wrapping
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size
        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size
        
        return True
    
    def shoot_missile(self, target_pos: Optional[Dict[str, float]] = None):
        """Shoot a missile toward target."""
        current_time = pygame.time.get_ticks()
        
        if current_time - self.last_missile_time < self.missile_cooldown:
            return
        
        if len(self.missiles) >= 3:
            return
        
        # Calculate missile properties
        start_x = self.position["x"] + self.size // 2
        start_y = self.position["y"] + self.size // 2
        speed = 5.0
        
        if target_pos:
            # Aim toward target
            dx = target_pos["x"] - start_x
            dy = target_pos["y"] - start_y
            angle = math.atan2(dy, dx)
            angle += random.uniform(-0.1, 0.1)  # Add inaccuracy
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
        else:
            # Shoot forward
            vx = speed
            vy = 0.0
        
        missile = Missile(start_x, start_y, vx, vy)
        self.missiles.append(missile)
        self.last_missile_time = current_time
        
        # Collect training data
        if self.training_data is not None and target_pos:
            self.training_data.append({
                "player_pos": self.position.copy(),
                "target_pos": target_pos.copy(),
                "missile_velocity": {"vx": vx, "vy": vy},
                "timestamp": current_time
            })
    
    def update_missiles(self):
        """Update all missiles."""
        current_time = pygame.time.get_ticks()
        
        for missile in self.missiles[:]:
            missile.update()
            
            # Remove expired missiles
            if current_time - missile.birth_time >= missile.lifespan:
                self.missiles.remove(missile)
                continue
            
            # Screen wrapping
            if missile.pos["x"] < -missile.size:
                missile.pos["x"] = self.screen_width
            elif missile.pos["x"] > self.screen_width:
                missile.pos["x"] = -missile.size
            if missile.pos["y"] < -missile.size:
                missile.pos["y"] = self.screen_height
            elif missile.pos["y"] > self.screen_height:
                missile.pos["y"] = -missile.size
    
    def get_training_data(self) -> List[Dict]:
        """Get collected training data."""
        return self.training_data.copy() if self.training_data else []
    
    def clear_training_data(self):
        """Clear collected training data."""
        if self.training_data:
            self.training_data.clear()
    
    def draw(self, screen: pygame.Surface):
        """Draw the player and missiles."""
        pygame.draw.rect(
            screen, self.color,
            (self.position["x"], self.position["y"], self.size, self.size)
        )
        
        for missile in self.missiles:
            missile.draw(screen)