import logging
import math
import random
from typing import Dict, List, Optional

import pygame

from ai_platform_trainer.ai.missile_ai_loader import (
    create_smart_missile,
    get_missile_ai_status,
)
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
        self.missile_cooldown = 500  # Cooldown in milliseconds
        self.last_missile_time = 0
        # Degrees for pygame.transform.rotate; 0 matches the sprite's
        # native "facing up" artwork. Holds its last value while idle.
        self.facing_angle = 0.0

    def reset(self) -> None:
        self.position = {"x": self.screen_width // 4, "y": self.screen_height // 2}
        self.missiles.clear()
        self.last_missile_time = 0
        logging.info("Player has been reset to the initial position.")

    def handle_input(self) -> bool:
        keys = pygame.key.get_pressed()

        # WASD / Arrow key movement
        dx = 0
        dy = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= self.step
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += self.step
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= self.step
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += self.step

        self.position["x"] += dx
        self.position["y"] += dy

        # Face the direction of travel. WASD/arrow combinations only ever
        # produce 8 possible (dx, dy) vectors, so this already lands
        # exactly on one of the 8 compass angles with no extra snapping
        # needed. Holds the last facing angle while standing still.
        if dx != 0 or dy != 0:
            self.facing_angle = -math.degrees(math.atan2(dx, -dy))

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
        current_time = pygame.time.get_ticks()

        # Check if cooldown has elapsed
        if current_time - self.last_missile_time < self.missile_cooldown:
            logging.debug("Missile on cooldown")
            return

        # Allow multiple missiles (up to 3)
        if len(self.missiles) >= 3:
            logging.debug("Maximum number of missiles already active")
            return

        missile_start_x = self.position["x"] + self.size // 2
        missile_start_y = self.position["y"] + self.size // 2

        birth_time = current_time
        # Random lifespan from 2.5-3s. Short enough that a fast, fleeing
        # enemy has a real shot at outrunning the missile before it expires,
        # instead of it homing in for 8-12s (long enough to always connect).
        random_lifespan = random.randint(2500, 3000)
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
            # If no enemy position, shoot forward
            vx = missile_speed
            vy = 0.0

        # Create a smart missile with AI homing capabilities
        target_x = enemy_pos["x"] if enemy_pos else missile_start_x + 200
        target_y = enemy_pos["y"] if enemy_pos else missile_start_y

        missile = create_smart_missile(
            x=missile_start_x,
            y=missile_start_y,
            target_x=target_x,
            target_y=target_y,
            speed=missile_speed,
            vx=vx,
            vy=vy,
            birth_time=birth_time,
            lifespan=random_lifespan,
        )
        self.missiles.append(missile)
        self.last_missile_time = current_time
        logging.info("Play mode: Shot a missile with increased travel distance.")

    def update_missiles(self, enemy_pos: Optional[Dict[str, float]] = None) -> None:
        current_time = pygame.time.get_ticks()
        for missile in self.missiles[:]:
            # A missile that has already expired just plays out its
            # explosion animation in place, then gets removed once it's done.
            if missile.exploded:
                missile.update_explosion(current_time)
                if missile.explosion_finished(current_time):
                    self.missiles.remove(missile)
                    logging.debug("Missile explosion finished; removed.")
                continue

            # Check if it's a SmartMissile with AI capabilities
            if hasattr(missile, "update_with_ai") and enemy_pos:
                # Use the AI update method with target position
                missile.update_with_ai(self.position, enemy_pos)
            else:
                # Fallback to basic update
                missile.update()

            # Explode in place once it expires, instead of vanishing -
            # gives a clear signal the enemy actually escaped this one.
            if current_time - missile.birth_time >= missile.lifespan:
                missile.explode(current_time)
                logging.debug("Missile lifespan exceeded; exploding.")
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
