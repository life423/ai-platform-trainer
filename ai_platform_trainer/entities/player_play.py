import pygame
import logging
import random
import math
from typing import List, Optional, Dict
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

    def shoot_missile(self, enemy_pos: Optional[Dict[str, float]] = None) -> None:
        # Shoot only if no missile is active, or allow multiple—up to you
        if len(self.missiles) == 0:
            missile_start_x = self.position["x"] + self.size // 2
            missile_start_y = self.position["y"] + self.size // 2

            birth_time = pygame.time.get_ticks()
            # Random lifespan from 8-12s (significantly increased to allow longer travel distance)
            random_lifespan = random.randint(8000, 12000)
            missile_speed = 5.0

            # Determine initial velocity based on enemy position if available
            if enemy_pos is not None:
                # Calculate the angle toward the enemy's position
                dy = enemy_pos["y"] - missile_start_y
                dx = enemy_pos["x"] - missile_start_x
                angle = math.atan2(dy, dx)
                # Add a small random deviation to simulate inaccuracy
                angle += random.uniform(-0.1, 0.1)  # deviation in radians
                vx = missile_speed * math.cos(angle)
                vy = missile_speed * math.sin(angle)
            else:
                vx = missile_speed
                vy = 0.0

            # Create a new missile object with calculated initial velocity and random lifespan
            missile = Missile(
                x=missile_start_x,
                y=missile_start_y,
                speed=missile_speed,
                vx=vx,
                vy=vy,
                birth_time=birth_time,
                lifespan=random_lifespan,
            )
            self.missiles.append(missile)
            logging.info("Play mode: Shot a missile with increased travel distance.")
        else:
            logging.debug("Attempted to shoot missile, but one is already active.")

    def update_missiles(self) -> None:
        current_time = pygame.time.get_ticks()
        for missile in self.missiles[:]:
            missile.update()

            # Remove if it expires
            if current_time - missile.birth_time >= missile.lifespan:
                self.missiles.remove(missile)
                logging.debug("Missile removed for exceeding lifespan.")
                continue

            # Screen wrapping for missiles, similar to player wrapping
            if missile.pos["x"] < -missile.size:
                missile.pos["x"] = self.screen_width
            elif missile.pos["x"] > self.screen_width:
                missile.pos["x"] = -missile.size
            if missile.pos["y"] < -missile.size:
                missile.pos["y"] = self.screen_height
            elif missile.pos["y"] > self.screen_height:
                missile.pos["y"] = -missile.size

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
