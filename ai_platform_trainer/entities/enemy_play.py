"""
Enemy entity for play mode: drawing, fade-in, respawn and stats. Movement comes
from the AI Agent subclass (entities/enemy_agent.py).
"""
from typing import Tuple

import pygame


class EnemyPlay:
    """The enemy in Play mode. Subclasses decide how it moves."""

    def __init__(self, screen_width: int, screen_height: int) -> None:
        """
        Initialize the enemy entity.

        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (139, 0, 0)  # Dark red
        self.pos = {"x": float(screen_width // 2), "y": float(screen_height // 2)}
        self.speed = 5.0
        self.visible = True
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_start_time = 0
        self.fade_duration = 1000  # 1 second fade-in

        # Simple counters for the stats panel.
        self.hits_on_player = 0
        self.times_hit_by_missile = 0

    def on_hit_player(self) -> None:
        """Called when this enemy successfully hits the player."""
        self.hits_on_player += 1

    def on_hit_by_missile(self) -> None:
        """Called when this enemy is hit by a missile."""
        self.times_hit_by_missile += 1

    def panel_stats(self) -> dict:
        """What the Play mode enemy panel shows (see PlayLearningMode)."""
        return {
            "title": "Enemy",
            "summary": [],
            "hits": self.hits_on_player,
            "deaths": self.times_hit_by_missile,
        }

    def _wrap_position(self) -> None:
        """Wrap the enemy position around screen edges."""
        self.pos["x"], self.pos["y"] = self.wrap_position(self.pos["x"], self.pos["y"])

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Wrap the given coordinates around screen edges.

        Args:
            x: X position to wrap
            y: Y position to wrap

        Returns:
            Tuple of wrapped (x, y) coordinates
        """
        if x < -self.size:
            x = self.screen_width
        elif x > self.screen_width:
            x = -self.size

        if y < -self.size:
            y = self.screen_height
        elif y > self.screen_height:
            y = -self.size

        return x, y

    def set_position(self, x: float, y: float) -> None:
        """
        Set the enemy position.

        Args:
            x: New x position
            y: New y position
        """
        self.pos["x"] = x
        self.pos["y"] = y

    def hide(self) -> None:
        """Hide the enemy (e.g., after being hit)."""
        self.visible = False

    def show(self, current_time: int) -> None:
        """
        Show the enemy with a fade-in effect.

        Args:
            current_time: Current game time in milliseconds
        """
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time

    def update_fade_in(self, current_time: int) -> None:
        """
        Update the fade-in effect.

        Args:
            current_time: Current game time in milliseconds
        """
        if not self.fading_in:
            return

        elapsed = current_time - self.fade_start_time
        progress = min(1.0, elapsed / self.fade_duration)

        self.fade_alpha = int(255 * progress)

        if progress >= 1.0:
            self.fading_in = False
            self.fade_alpha = 255

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the enemy on the screen.

        Args:
            screen: Pygame surface to draw on
        """
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
                screen, self.color, (self.pos["x"], self.pos["y"], self.size, self.size)
            )
