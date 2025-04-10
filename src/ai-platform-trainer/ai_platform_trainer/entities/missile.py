# ai_platform_trainer/entities/missile.py

import pygame



class Missile:
    def __init__(
        self,
        x: int,
        y: int,
        speed: float = 5.0,
        vx: float = 5.0,
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 1000,  # default 1s if not overridden
    ):
        self.size = 10
        self.color = (255, 255, 0)  # Yellow
        self.pos = {"x": x, "y": y}
        self.speed = speed
        # Velocity components for straight line movement
        self.vx = vx
        self.vy = vy

        # New fields for matching training logic:
        self.birth_time = birth_time
        self.lifespan = lifespan

    def update(self) -> None:
        """
        Update missile position based on its velocity.
        """
        self.pos["x"] += self.vx
        self.pos["y"] += self.vy

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the missile on the screen."""
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos["x"]), int(self.pos["y"])),
            self.size,
        )

    def get_rect(self) -> pygame.Rect:
        """Get the missile's rectangle for collision detection."""
        return pygame.Rect(
            self.pos["x"] - self.size,
            self.pos["y"] - self.size,
            self.size * 2,
            self.size * 2,
        )
