# file: ai_platform_trainer/entities/player_play.py

import pygame
import logging
import random
from typing import List
from ai_platform_trainer.entities.missile import Missile

class PlayerPlay:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)  # Dark Blue
        self.position = {"x": screen_width // 4, "y": screen_height // 2}
        self.step = 5
        self.missiles: List[Missile] = []

    @property
    def x(self) -> int:
        """
        Allows environment or other code to get/set player's x-position.
        """
        return self.position["x"]

    @x.setter
    def x(self, value: int):
        self.position["x"] = value

    @property
    def y(self) -> int:
        """
        Allows environment or other code to get/set player's y-position.
        """
        return self.position["y"]

    @y.setter
    def y(self, value: int):
        self.position["y"] = value

    def reset(self) -> None:
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        keys = pygame.key.get_pressed()

        # WASD / Arrow key movement
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.position["x"] -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.position["x"] += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.position["y"] -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.position["y"] += self.step

        # Wrap-around logic
        if self.position["x"] < -self.size:
            self.position["x"] = self.screen_width
        elif self.position["x"] > self.screen_width:
            self.position["x"] = -self.size

        if self.position["y"] < -self.size:
            self.position["y"] = self.screen_height
        elif self.position["y"] > self.screen_height:
            self.position["y"] = -self.size

        return True

    def shoot_missile(self) -> None:
        # Shoot only if no missile is active (or tweak logic if you want multiple missiles)
        if len(self.missiles) == 0:
            missile_start_x = self.position["x"] + self.size // 2
            missile_start_y = self.position["y"] + self.size // 2

            birth_time = pygame.time.get_ticks()
            # Random lifespan from 0.5–1.5s
            random_lifespan = random.randint(500, 1500)

            missile = Missile(
                x=missile_start_x,
                y=missile_start_y,
                vx=5.0,
                vy=0.0,
                birth_time=birth_time,
                lifespan=random_lifespan,
            )
            self.missiles.append(missile)
            logging.info("Play mode: Shot a missile with random lifespan.")
        else:
            logging.debug("Attempted to shoot missile, but one is already active.")

    def update_missiles(self) -> None:
        current_time = pygame.time.get_ticks()
        for missile in self.missiles[:]:
            missile.update()
            # Remove if it expires or goes off-screen
            if current_time - missile.birth_time >= missile.lifespan:
                self.missiles.remove(missile)
                logging.debug("Missile removed for exceeding lifespan.")
                continue

            if (
                missile.pos["x"] < 0
                or missile.pos["x"] > self.screen_width
                or missile.pos["y"] < 0
                or missile.pos["y"] > self.screen_height
            ):
                self.missiles.remove(missile)
                logging.debug("Missile removed for going off-screen.")

    def draw_missiles(self, screen: pygame.Surface) -> None:
        for missile in self.missiles:
            missile.draw(screen)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)