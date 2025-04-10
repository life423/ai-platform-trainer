# ai_platform_trainer/entities/enemy_play.py

import math
import pygame
import logging
from typing import Optional, Tuple

import torch  # Possibly no longer needed if we fully delegate AI logic
from ai_platform_trainer.gameplay.ai.enemy_ai_controller import update_enemy_movement

class EnemyPlay:
    def __init__(self, screen_width: int, screen_height: int, model):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (255, 215, 0)  # Gold
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.model = model
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        # Fade-in attributes
        self.fading_in = False
        self.fade_alpha = 255
        self.fade_duration = 300  # milliseconds
        self.fade_start_time = 0

        # Create a Surface for the enemy with per-pixel alpha
        self.surface = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        self.surface.fill((*self.color, 255))

    def wrap_position(self, x: float, y: float) -> Tuple[float, float]:
        """
        Example wrap-around logic. 
        You might unify this in a separate utils file if used by multiple entities.
        """
        def wrap(val: float, lower: float, upper: float) -> float:
            if val < lower:
                return upper
            elif val > upper:
                return lower
            return val

        new_x = wrap(x, -self.size, self.screen_width)
        new_y = wrap(y, -self.size, self.screen_height)
        return new_x, new_y

    def update_movement(
        self,
        player_x: float,
        player_y: float,
        player_speed: int,
        current_time: int
    ):
        """
        Delegates AI movement logic to the 'enemy_ai_controller.update_enemy_movement'
        function, keeping EnemyPlay simpler.
        """
        update_enemy_movement(
            self,
            player_x=player_x,
            player_y=player_y,
            player_speed=player_speed,
            current_time=current_time,
        )

    def draw(self, screen: pygame.Surface) -> None:
        if self.visible:
            self.surface.set_alpha(self.fade_alpha)
            screen.blit(self.surface, (self.pos["x"], self.pos["y"]))

    def hide(self) -> None:
        self.visible = False
        logging.info("Enemy hidden due to collision.")

    def show(self, current_time: int) -> None:
        self.visible = True
        self.fading_in = True
        self.fade_alpha = 0
        self.fade_start_time = current_time
        logging.info("Enemy set to fade in.")

    def update_fade_in(self, current_time: int) -> None:
        if self.fading_in:
            elapsed = current_time - self.fade_start_time
            if elapsed >= self.fade_duration:
                self.fade_alpha = 255
                self.fading_in = False
                logging.info("Enemy fade-in completed.")
            else:
                self.fade_alpha = int((elapsed / self.fade_duration) * 255)
                logging.debug(f"Enemy fade-in alpha: {self.fade_alpha}")

    def set_position(self, x: int, y: int) -> None:
        self.pos["x"], self.pos["y"] = x, y