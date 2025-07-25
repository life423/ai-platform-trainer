import random
import math
import logging
import pygame
from typing import Optional, Dict
from ai_platform_trainer.entities.components.missile import Missile
from ai_platform_trainer.utils.helpers import wrap_position


class PlayerTraining:
    PATTERNS = ["random_walk", "circle_move", "diagonal_move"]

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.size = 50
        self.color = (0, 0, 139)
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.step = 5
        self.missiles = []
        logging.info("PlayerTraining initialized.")

        # Desired distance logic for movement patterns
        self.desired_distance = 200
        self.margin = 20

        self.current_pattern = None
        self.state_timer = 0
        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        self.random_walk_speed = self.step
        self.circle_angle = 0.0
        self.circle_radius = 100
        self.diagonal_direction = (1, 1)

        # Remove wrap-around cooldown attributes altogether
        # self.wrap_cooldown = 0
        # self.wrap_cooldown_frames = 120

        # Velocity for smoother movement
        self.velocity = {"x": 0.0, "y": 0.0}
        self.velocity_blend_factor = (
            0.2  # Lower = smoother movement, Higher = more responsive
        )

        # Initialize a random pattern
        self.switch_pattern()

    def switch_pattern(self):
        """
        Chooses a new random pattern that differs from the current one.
        """
        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.PATTERNS)

        self.current_pattern = new_pattern
        self.state_timer = random.randint(180, 300)

        if self.current_pattern == "circle_move":
            cx = max(self.size, min(self.screen_width - self.size, self.position["x"]))
            cy = max(self.size, min(self.screen_height - self.size, self.position["y"]))
            self.circle_center = (cx, cy)
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)
        elif self.current_pattern == "diagonal_move":
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

        logging.debug(f"Switched pattern to {self.current_pattern} at {self.position}")

    def reset(self) -> None:
        """
        Reset player position and missile list, then pick a new movement pattern.
        """
        self.position = {
            "x": random.randint(0, self.screen_width - self.size),
            "y": random.randint(0, self.screen_height - self.size),
        }
        self.missiles.clear()
        self.switch_pattern()
        logging.info("PlayerTraining has been reset.")

    def bias_angle_away_from_enemy(self, enemy_x, enemy_y, base_angle):
        """
        Adjusts the given angle away from the enemy if the player is too close,
        or somewhat towards the enemy if the player is too far, adding realism.
        """
        dx = enemy_x - self.position["x"]
        dy = enemy_y - self.position["y"]
        dist = math.hypot(dx, dy)
        if dist == 0:
            return (base_angle + math.pi) % (2 * math.pi)

        enemy_angle = math.atan2(dy, dx)
        # Decide how strongly to bias the angle
        if dist < self.desired_distance - self.margin:
            bias_strength = math.radians(30)
        elif dist > self.desired_distance + self.margin:
            bias_strength = math.radians(15)
        else:
            bias_strength = math.radians(45)

        angle_diff = (base_angle - enemy_angle) % (2 * math.pi)
        if angle_diff < math.pi:
            new_angle = base_angle + bias_strength
        else:
            new_angle = base_angle - bias_strength

        return new_angle % (2 * math.pi)

    def move_with_velocity(self, ndx, ndy):
        """
        Smooth velocity-based movement. 'ndx' and 'ndy' are normalized direction components.
        We blend current velocity toward (ndx * step, ndy * step).
        """
        target_vx = ndx * self.step
        target_vy = ndy * self.step

        # Blend velocities to reduce jitter
        self.velocity["x"] = (1 - self.velocity_blend_factor) * self.velocity[
            "x"
        ] + self.velocity_blend_factor * target_vx
        self.velocity["y"] = (1 - self.velocity_blend_factor) * self.velocity[
            "y"
        ] + self.velocity_blend_factor * target_vy

        self.position["x"] += self.velocity["x"]
        self.position["y"] += self.velocity["y"]

    def random_walk_pattern(self, enemy_x, enemy_y):
        """
        Random angle movement that adjusts slightly away from the enemy if too close,
        or more random if not near the enemy.
        """
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            self.random_walk_speed = self.step
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        angle = self.bias_angle_away_from_enemy(
            enemy_x, enemy_y, self.random_walk_angle
        )
        ndx = math.cos(angle)
        ndy = math.sin(angle)
        self.move_with_velocity(ndx, ndy)

        logging.debug(f"Random walk: pos={self.position}")

    def circle_pattern(self, enemy_x, enemy_y):
        """
        Moves in a rough circle around a center point, while also adjusting
        away from the enemy if too close.
        """
        angle_increment = 0.02
        self.circle_angle += angle_increment

        desired_x = (
            self.circle_center[0] + math.cos(self.circle_angle) * self.circle_radius
        )
        desired_y = (
            self.circle_center[1] + math.sin(self.circle_angle) * self.circle_radius
        )

        dx = desired_x - self.position["x"]
        dy = desired_y - self.position["y"]

        base_angle = math.atan2(dy, dx)
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)
        ndx = math.cos(final_angle)
        ndy = math.sin(final_angle)
        self.move_with_velocity(ndx, ndy)

        # Occasionally adjust circle radius
        if random.random() < 0.01:
            self.circle_radius += random.randint(-2, 2)
            self.circle_radius = max(20, min(200, self.circle_radius))

        logging.debug(
            f"Circle move: pos={self.position}, center={self.circle_center}, "
            f"radius={self.circle_radius}"
        )

    def diagonal_pattern(self, enemy_x, enemy_y):
        """
        Moves diagonally, occasionally adjusting angle slightly.
        Also biases away from or towards the enemy as needed.
        """
        if random.random() < 0.02:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        base_angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
        final_angle = self.bias_angle_away_from_enemy(enemy_x, enemy_y, base_angle)

        self.diagonal_direction = (math.cos(final_angle), math.sin(final_angle))
        ndx, ndy = self.diagonal_direction
        self.move_with_velocity(ndx, ndy)

        logging.debug(
            f"Diagonal move: pos={self.position}, direction={self.diagonal_direction}"
        )

    def update(self, enemy_x: float, enemy_y: float) -> None:
        """
        Main update method:
        - Picks patterns based on distance to enemy
        - Applies direct toroidal wrap every frame (Pac-Man style)
        """
        dist = math.hypot(self.position["x"] - enemy_x, self.position["y"] - enemy_y)
        close_threshold = self.desired_distance - self.margin
        far_threshold = self.desired_distance + self.margin

        # Decrement state_timer and switch pattern if needed
        self.state_timer -= 1
        if self.state_timer <= 0:
            self.switch_pattern()

        # Choose movement pattern based on distance
        if dist < close_threshold:
            self.random_walk_pattern(enemy_x, enemy_y)
        elif dist > far_threshold:
            if self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)
            else:
                self.random_walk_pattern(enemy_x, enemy_y)
        else:
            if self.current_pattern == "random_walk":
                self.random_walk_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "circle_move":
                self.circle_pattern(enemy_x, enemy_y)
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern(enemy_x, enemy_y)

        # Perform direct wrap each frame
        new_x, new_y = wrap_position(
            self.position["x"],
            self.position["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )
        self.position["x"], self.position["y"] = new_x, new_y

    def shoot_missile(self, enemy_pos: Optional[Dict[str, float]] = None) -> None:
        """
        Fires a missile toward the enemy with a slight random angle offset
        and a random lifespan. Only one missile at a time.
        """
        if len(self.missiles) == 0:
            missile_start_x = self.position["x"] + self.size // 2
            missile_start_y = self.position["y"] + self.size // 2

            # Extract enemy coordinates from position dict
            if enemy_pos:
                enemy_x = enemy_pos["x"]
                enemy_y = enemy_pos["y"]
            else:
                # Fallback to center of screen if no target provided
                enemy_x = 640
                enemy_y = 360

            # Compute base angle
            dx = enemy_x - missile_start_x
            dy = enemy_y - missile_start_y
            angle = math.atan2(dy, dx)

            # NEW: Add a random offset to the angle for variety
            offset_degrees = random.uniform(-10, 10)  # e.g., ±10 degrees
            angle += math.radians(offset_degrees)

            speed = 5.0
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # Random missile lifespan
            lifespan = random.randint(750, 4500)  # 0.75s - 4.5s
            birth_time = pygame.time.get_ticks()

            missile = Missile(
                missile_start_x,
                missile_start_y,
                speed=speed,
                vx=vx,
                vy=vy,
                lifespan=lifespan,
                birth_time=birth_time,
            )
            self.missiles.append(missile)
            logging.info(
                f"Training Mode: Missile shot with offset {offset_degrees:.1f}°, "
                f"final angle: {math.degrees(angle):.1f}°"
            )

    def update_missiles(self) -> None:
        """
        Let each missile move and remove it if it goes off-screen.
        """
        for missile in self.missiles[:]:
            missile.update()
            # Remove if off-screen
            if (
                missile.pos["x"] < 0
                or missile.pos["x"] > self.screen_width
                or missile.pos["y"] < 0
                or missile.pos["y"] > self.screen_height
            ):
                self.missiles.remove(missile)
                logging.debug("Missile removed for going off-screen.")

    def draw_missiles(self, screen: pygame.Surface) -> None:
        """
        Draw each missile on the given screen surface.
        """
        for missile in self.missiles:
            missile.draw(screen)

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the player (a rectangle) and any active missiles.
        """
        pygame.draw.rect(
            screen,
            self.color,
            (self.position["x"], self.position["y"], self.size, self.size),
        )
        self.draw_missiles(screen)
