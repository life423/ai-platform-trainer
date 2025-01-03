import pygame
import random
import math
import logging
from ai_platform_trainer.entities.enemy import Enemy
from ai_platform_trainer.utils.helpers import wrap_position


class EnemyTrain(Enemy):
    DEFAULT_SIZE = 50
    DEFAULT_COLOR = (173, 153, 228)
    PATTERNS = ["random_walk", "circle_move", "diagonal_move"]
    WALL_MARGIN = 20

    def __init__(self, screen_width: int, screen_height: int):
        # No model needed for EnemyTrain in current logic, pass model=None
        super().__init__(screen_width, screen_height, model=None)
        self.size = self.DEFAULT_SIZE
        self.color = self.DEFAULT_COLOR
        self.pos = {"x": self.screen_width // 2, "y": self.screen_height // 2}
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        self.state_timer = 0
        self.current_pattern = None

        self.wall_stall_counter = 0
        self.wall_stall_threshold = 10
        self.forced_escape_timer = 0
        self.forced_angle = None
        self.forced_speed = None

        self.circle_center = (self.pos["x"], self.pos["y"])
        self.circle_angle = 0.0
        self.circle_radius = 100
        self.diagonal_direction = (1, 1)

        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        self.random_walk_speed = self.base_speed

        self.switch_pattern()

    def switch_pattern(self):
        if self.forced_escape_timer > 0:
            return

        new_pattern = self.current_pattern
        while new_pattern == self.current_pattern:
            new_pattern = random.choice(self.PATTERNS)

        self.current_pattern = new_pattern
        self.state_timer = random.randint(120, 300)

        if self.current_pattern == "circle_move":
            self.circle_center = (self.pos["x"], self.pos["y"])
            self.circle_angle = random.uniform(0, 2 * math.pi)
            self.circle_radius = random.randint(50, 150)
        elif self.current_pattern == "diagonal_move":
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            self.diagonal_direction = (dx, dy)

    def update_movement(self, player_x, player_y, player_speed):
        """
        EnemyTrain does not use a model. Movement is pattern-based.
        """
        if self.forced_escape_timer > 0:
            self.forced_escape_timer -= 1
            self.apply_forced_escape_movement()
        else:
            self.state_timer -= 1
            if self.state_timer <= 0:
                self.switch_pattern()

            if self.current_pattern == "random_walk":
                self.random_walk_pattern()
            elif self.current_pattern == "circle_move":
                self.circle_pattern()
            elif self.current_pattern == "diagonal_move":
                self.diagonal_pattern()

        self.pos["x"], self.pos["y"] = wrap_position(
            self.pos["x"],
            self.pos["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )

    def initiate_forced_escape(self):
        dist_left = self.pos["x"]
        dist_right = (self.screen_width - self.size) - self.pos["x"]
        dist_top = self.pos["y"]
        dist_bottom = (self.screen_height - self.size) - self.pos["y"]

        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

        if min_dist == dist_left:
            base_angle = 0.0
        elif min_dist == dist_right:
            base_angle = math.pi
        elif min_dist == dist_top:
            base_angle = math.pi / 2
        else:
            base_angle = math.pi * 3 / 2

        angle_variation = math.radians(30)
        self.forced_angle = base_angle + random.uniform(
            -angle_variation, angle_variation
        )
        self.forced_speed = self.base_speed * 1.0
        self.forced_escape_timer = random.randint(1, 30)
        self.wall_stall_counter = 0
        self.state_timer = self.forced_escape_timer * 2

    def apply_forced_escape_movement(self):
        dx = math.cos(self.forced_angle) * self.forced_speed
        dy = math.sin(self.forced_angle) * self.forced_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def is_hugging_wall(self) -> bool:
        return (
            self.pos["x"] < self.WALL_MARGIN
            or self.pos["x"] > self.screen_width - self.size - self.WALL_MARGIN
            or self.pos["y"] < self.WALL_MARGIN
            or self.pos["y"] > self.screen_height - self.size - self.WALL_MARGIN
        )

    def random_walk_pattern(self):
        if self.random_walk_timer <= 0:
            self.random_walk_angle = random.uniform(0, 2 * math.pi)
            self.random_walk_speed = self.base_speed * random.uniform(0.5, 2.0)
            self.random_walk_timer = random.randint(30, 90)
        else:
            self.random_walk_timer -= 1

        dx = math.cos(self.random_walk_angle) * self.random_walk_speed
        dy = math.sin(self.random_walk_angle) * self.random_walk_speed
        self.pos["x"] += dx
        self.pos["y"] += dy

    def circle_pattern(self):
        speed = self.base_speed
        angle_increment = 0.02 * (speed / self.base_speed)
        self.circle_angle += angle_increment

        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.pos["x"] = self.circle_center[0] + dx
        self.pos["y"] = self.circle_center[1] + dy

        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        if random.random() < 0.05:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        speed = self.base_speed
        self.pos["x"] += self.diagonal_direction[0] * speed
        self.pos["y"] += self.diagonal_direction[1] * speed

    def hide(self):
        """
        Override to hide EnemyTrain. No fade here, just hide immediately.
        """
        self.visible = False
        logging.info("EnemyTrain hidden due to collision.")

    def show(self, current_time: int = None):
        """
        For EnemyTrain, we won't do fade-in. Just make it visible again.
        The current_time argument is optional and unused here.
        """
        self.visible = True
        logging.info("EnemyTrain made visible again.")
