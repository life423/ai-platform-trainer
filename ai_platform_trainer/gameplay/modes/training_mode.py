"""
Training Data Collection Mode

Runs the player and enemy on scripted (non-human, non-learning) movement
patterns so their interactions can be logged as supervised training data
for the missile guidance model (see ai/training/train_missile_model.py),
with the whole session visible in the GUI rather than running headless.
"""
import logging
import math

import pygame

from ai_platform_trainer.ai.inference.missile_controller import update_missile_ai
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.spawner import spawn_entities


class TrainingMode:
    """
    Generates missile-guidance training data.

    The player and enemy both move on scripted patterns (no human input,
    no learned behavior), while every active missile's state is logged
    each frame - including whether it just hit the enemy - to
    data/raw/training_data.json for train_missile_model.py to learn from.
    """

    # How often (ms) the scripted player fires a missile at the enemy.
    FIRE_INTERVAL_MS = 1500

    def __init__(self, game):
        """Set up scripted entities and start a fresh data-logging session."""
        self.game = game

        self.player = PlayerTraining(game.screen_width, game.screen_height)
        self.enemy = EnemyTrain(game.screen_width, game.screen_height)
        game.player = self.player
        game.enemy = self.enemy

        spawn_entities(game)

        # NOTE: DataLogger deletes any existing file at this path and starts
        # fresh each session - it captures only this session's new samples,
        # matching how ai/utils/data_validator_and_trainer.py expects to
        # receive a batch of *new* data to validate/merge/retrain from,
        # rather than being handed the whole accumulated dataset.
        self.data_logger = DataLogger(config.DATA_PATH)
        game.data_logger = self.data_logger

        self.last_fire_time = 0

        logging.info("Training data collection mode initialized")

    def update(self, current_time: int) -> None:
        """Advance one frame of scripted play and log a data point per missile."""
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])
        self.enemy.update_movement(
            self.player.position["x"], self.player.position["y"], self.player.step
        )

        if (
            self.enemy.visible
            and current_time - self.last_fire_time >= self.FIRE_INTERVAL_MS
        ):
            self.player.shoot_missile(self.enemy.pos)
            self.last_fire_time = current_time

        self.player.update_missiles()

        # Guide missiles with the current missile model - this also fills in
        # missile.last_action, which becomes the supervised training label.
        if self.game.missile_model and self.player.missiles:
            update_missile_ai(
                self.player.missiles,
                self.player.position,
                self.enemy.pos,
                self.game._missile_input,
                self.game.missile_model,
            )

        # Snapshot before collision resolution: handle_missile_collisions
        # removes any missile that hits, so this is the only way to still
        # log a final data point (with missile_collision=True) for it.
        missiles_before = list(self.player.missiles)

        def respawn_callback() -> None:
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay

        handle_missile_collisions(self.player, self.enemy, respawn_callback)
        hit_missiles = [m for m in missiles_before if m not in self.player.missiles]

        for missile in missiles_before:
            self._log_frame(missile, current_time, missile in hit_missiles)

        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self.game.handle_respawn(current_time)

        if self.enemy.fading_in:
            self.enemy.update_fade_in(current_time)

    def _log_frame(self, missile, current_time: int, collided: bool) -> None:
        """Log one training data point for the given missile's current state."""
        self.data_logger.log(
            {
                "player_x": self.player.position["x"],
                "player_y": self.player.position["y"],
                "enemy_x": self.enemy.pos["x"],
                "enemy_y": self.enemy.pos["y"],
                "missile_x": missile.pos["x"],
                "missile_y": missile.pos["y"],
                "missile_angle": math.atan2(missile.vy, missile.vx),
                "missile_collision": collided,
                "missile_action": missile.last_action,
                "timestamp": current_time,
            }
        )

    def draw_mode_info(self, screen: pygame.Surface) -> None:
        """Draw a small overlay confirming data collection is running."""
        font = pygame.font.Font(None, 32)
        text = font.render(
            f"TRAINING MODE - {len(self.data_logger.data)} samples logged",
            True,
            (255, 255, 255),
        )
        bg = pygame.Surface(
            (text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA
        )
        bg.fill((0, 0, 0, 140))
        screen.blit(bg, (10, 10))
        screen.blit(text, (20, 15))
