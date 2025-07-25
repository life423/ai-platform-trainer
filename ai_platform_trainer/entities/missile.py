# ai_platform_trainer/entities/missile.py

import pygame
import math


class Missile:
    def __init__(
        self,
        x: int,
        y: int,
        speed: float = 8.0,  # Increased from 5.0 to allow faster missile travel
        vx: float = 8.0,  # Increased from 5.0 to allow faster missile travel
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 20000,  # default 20s (doubled again from 10s to allow
                                # even longer travel distance)
    ):
        self.size = 10
        self.color = (255, 255, 0)  # Yellow
        self.pos = {"x": x, "y": y}
        self.position = {"x": x, "y": y}  # Add position attribute for renderer
        self.speed = speed
        # Velocity components for straight line movement
        self.vx = vx
        self.vy = vy
        
        # Direction for rendering (normalized)
        if vx != 0 or vy != 0:
            magnitude = math.sqrt(vx * vx + vy * vy)
            self.direction = (vx / magnitude, vy / magnitude)
        else:
            self.direction = (1.0, 0.0)  # Default rightward direction

        # New fields for matching training logic:
        self.birth_time = birth_time
        self.lifespan = lifespan
        self.last_action = 0.0  # Store last AI action for training

    def update(self) -> None:
        """
        Update missile position based on its velocity.
        """
        self.pos["x"] += self.vx
        self.pos["y"] += self.vy
        # Keep position in sync with pos for renderer
        self.position["x"] = self.pos["x"]
        self.position["y"] = self.pos["y"]

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the missile on the screen."""
        # Draw missile as a small triangle pointing in the direction of movement
        angle = math.atan2(self.vy, self.vx)
        
        # Calculate triangle points
        center_x, center_y = int(self.pos["x"]), int(self.pos["y"])
        
        # Front point (nose of missile)
        front_x = center_x + int(self.size * math.cos(angle))
        front_y = center_y + int(self.size * math.sin(angle))
        
        # Back points (tail of missile)
        back_angle1 = angle + math.pi * 0.8  # 144 degrees from front
        back_angle2 = angle - math.pi * 0.8  # -144 degrees from front
        
        back_x1 = center_x + int(self.size * 0.6 * math.cos(back_angle1))
        back_y1 = center_y + int(self.size * 0.6 * math.sin(back_angle1))
        
        back_x2 = center_x + int(self.size * 0.6 * math.cos(back_angle2))
        back_y2 = center_y + int(self.size * 0.6 * math.sin(back_angle2))
        
        # Draw the triangle
        pygame.draw.polygon(
            screen,
            self.color,
            [(front_x, front_y), (back_x1, back_y1), (back_x2, back_y2)]
        )

    def get_rect(self) -> pygame.Rect:
        """Get the missile's rectangle for collision detection."""
        return pygame.Rect(
            self.pos["x"] - self.size,
            self.pos["y"] - self.size,
            self.size * 2,
            self.size * 2,
        )