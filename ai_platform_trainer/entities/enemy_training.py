import logging
import math
import random

from ai_platform_trainer.entities.enemy import Enemy
from ai_platform_trainer.utils.helpers import wrap_position


class EnemyTrain(Enemy):
    """
    EnemyTrain is a subclass of Enemy that does not rely on a model.
    It uses pattern-based movement to generate training data or mimic
    certain AI behaviors.

    The available patterns are:
    - random_walk
    - circle_move
    - diagonal_move
    - pursue (optional direct pursuit of the player)
    """

    DEFAULT_SIZE = 50
    DEFAULT_COLOR = (173, 153, 228)
    PATTERNS = [
        "random_walk",
        "circle_move",
        "diagonal_move",
        "pursue",
    ]  # Added "pursue"
    PURSUIT_SPEED_FACTOR = 0.8  # Relative speed factor when pursuing the player

    def __init__(self, screen_width: int, screen_height: int):
        """
        Initialize the EnemyTrain with no model. Movement is purely pattern-based.
        """
        super().__init__(screen_width, screen_height, model=None)
        self.size = self.DEFAULT_SIZE
        self.color = self.DEFAULT_COLOR
        self.pos = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2,
        }
        self.base_speed = max(2, screen_width // 400)
        self.visible = True

        # Timers and states controlling pattern selection
        self.state_timer = 0
        self.current_pattern = None

        # Circle-move parameters
        self.circle_center = (self.pos["x"], self.pos["y"])
        self.circle_angle = 0.0
        self.circle_radius = 100

        # Diagonal-move parameters
        self.diagonal_direction = (1, 1)

        # Random-walk parameters
        self.random_walk_timer = 0
        self.random_walk_angle = 0.0
        self.random_walk_speed = self.base_speed

        # Pick an initial pattern
        self.switch_pattern()

    def switch_pattern(self):
        """
        Switch to a new movement pattern from PATTERNS, ensuring it's different
        from the current pattern. Also resets state_timer for the new pattern.
        """
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
        Primary movement logic, choosing from the current pattern:
        random_walk, circle_move, diagonal_move, or pursue.
        """
        # Decrement timer and possibly switch patterns
        self.state_timer -= 1
        if self.state_timer <= 0:
            self.switch_pattern()

        if self.current_pattern == "random_walk":
            self.random_walk_pattern()
        elif self.current_pattern == "circle_move":
            self.circle_pattern()
        elif self.current_pattern == "diagonal_move":
            self.diagonal_pattern()
        elif self.current_pattern == "pursue":
            self.pursue_pattern(player_x, player_y, player_speed)

        # Wrap around screen edges
        self.pos["x"], self.pos["y"] = wrap_position(
            self.pos["x"],
            self.pos["y"],
            self.screen_width,
            self.screen_height,
            self.size,
        )

    def pursue_pattern(self, player_x: float, player_y: float, player_speed: float):
        """
        Pursue the player by moving directly toward the player's position.
        The movement speed is based on the player's speed and a pursuit factor.
        """
        dx = player_x - self.pos["x"]
        dy = player_y - self.pos["y"]
        dist = math.hypot(dx, dy)

        if dist > 0:
            # Normalize dx, dy
            dx /= dist
            dy /= dist

            # Move at a fraction of the player's speed to avoid immediate collisions
            speed = player_speed * self.PURSUIT_SPEED_FACTOR
            self.pos["x"] += dx * speed
            self.pos["y"] += dy * speed

    def random_walk_pattern(self):
        """
        Move in a random direction for a random duration, then pick another angle.
        """
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
        """
        Move in a circular trajectory around a center point.
        """
        speed = self.base_speed
        angle_increment = 0.02 * (speed / self.base_speed)
        self.circle_angle += angle_increment

        dx = math.cos(self.circle_angle) * self.circle_radius
        dy = math.sin(self.circle_angle) * self.circle_radius
        self.pos["x"] = self.circle_center[0] + dx
        self.pos["y"] = self.circle_center[1] + dy

        # Occasionally alter circle radius
        if random.random() < 0.01:
            self.circle_radius += random.randint(-5, 5)
            self.circle_radius = max(20, min(200, self.circle_radius))

    def diagonal_pattern(self):
        """
        Move diagonally; occasionally randomize the diagonal angle.
        """
        if random.random() < 0.05:
            angle = math.atan2(self.diagonal_direction[1], self.diagonal_direction[0])
            angle += random.uniform(-0.3, 0.3)
            self.diagonal_direction = (math.cos(angle), math.sin(angle))

        speed = self.base_speed
        self.pos["x"] += self.diagonal_direction[0] * speed
        self.pos["y"] += self.diagonal_direction[1] * speed

    def hide(self):
        """
        Immediately hide EnemyTrain (no fade). Typically called upon collision.
        """
        self.visible = False
        logging.info("EnemyTrain hidden due to collision.")

    def show(self, current_time: int = None):
        """
        Make EnemyTrain visible again (no fade) after a collision.
        The current_time argument is optional and unused here.
        """
        self.visible = True
        logging.info("EnemyTrain made visible again.")
