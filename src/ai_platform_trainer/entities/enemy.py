import torch
import math
import pygame
import logging
import random
from typing import Optional


class Enemy:
    def __init__(
        self, screen_width: int, screen_height: int, model: Optional[torch.nn.Module]
    ):
        """
        Base enemy class.
        If a model is provided, it uses the model's output to guide movement.
        Otherwise, a subclass might define different behavior.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (173, 153, 228)
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True
        self.fading_in = False
        self.fade_start_time = 0
        self.alpha = 255
        self.image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.image.fill((*self.color, self.alpha))

    def wrap_position(self, x: float, y: float) -> (float, float):
        """Wrap the enemy's position around screen edges."""
        if x < -self.size:
            x = self.screen_width
        elif x > self.screen_width:
            x = -self.size
        if y < -self.size:
            y = self.screen_height
        elif y > self.screen_height:
            y = -self.size
        return x, y

    def update_movement(self, player_x: float, player_y: float, player_speed: int):
        """
        Update enemy movement based on player position and an optional model output.
        If no model is provided, subclass should override this method.
        """
        if not self.visible:
            return

        if self.model is None:
            # If no model, just don't move or let subclasses handle it
            return

        # Compute direction from model
        dist = math.sqrt(
            (player_x - self.pos["x"]) ** 2 + (player_y - self.pos["y"]) ** 2
        )
        state = torch.tensor(
            [[player_x, player_y, self.pos["x"], self.pos["y"], dist]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            action = self.model(state)

        action_dx, action_dy = action[0].tolist()

        # Normalize action vector
        action_len = math.sqrt(action_dx**2 + action_dy**2)
        if action_len > 0:
            action_dx /= action_len
            action_dy /= action_len
        else:
            # Apply a small random movement instead of freezing
            angle = random.uniform(0, 2 * math.pi)
            action_dx = math.cos(angle)
            action_dy = math.sin(angle)
            logging.debug(f"Applied fallback random movement for enemy at position {self.pos}")

        speed = player_speed * 0.7
        self.pos["x"] += action_dx * speed
        self.pos["y"] += action_dy * speed

        self.pos["x"], self.pos["y"] = self.wrap_position(self.pos["x"], self.pos["y"])

    def draw(self, screen: pygame.Surface):
        """Draw the enemy if visible."""
        if self.visible:
            self.image.set_alpha(self.alpha)
            screen.blit(self.image, (self.pos["x"], self.pos["y"]))

    def hide(self):
        """Hide the enemy."""
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self, current_time: int):
        """
        Show the enemy with a fade-in effect starting at current_time.
        """
        self.visible = True
        self.fading_in = True
        self.fade_start_time = current_time
        self.alpha = 0
        self.image.set_alpha(self.alpha)
        logging.info("Enemy set to fade in.")

    def set_position(self, x: int, y: int):
        """Set the enemy's position."""
        self.pos["x"], self.pos["y"] = x, y

    def start_fade_in(self, current_time: int):
        """Initiate fade-in effect."""
        self.fading_in = True
        self.fade_start_time = current_time
        self.alpha = 0
        self.image.set_alpha(self.alpha)
        logging.info("Enemy fade-in started.")

    def update_fade_in(self, current_time: int, fade_duration: int = 300):
        """Update fade-in effect based on elapsed time."""
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= fade_duration:
                self.alpha = 255
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                self.alpha = int((elapsed / fade_duration) * 255)
                logging.debug(f"Enemy fade-in alpha updated to {self.alpha}")
            self.image.set_alpha(self.alpha)
