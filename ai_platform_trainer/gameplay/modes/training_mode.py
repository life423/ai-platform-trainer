"""
Training Data Collection Mode

Runs the player and enemy on scripted (non-human, non-learning) movement
patterns so their interactions can be logged as supervised training data
for the missile guidance model (see ai/training/train_missile_model.py),
with the whole session visible in the GUI rather than running headless.
"""
import logging
import math
import os

import pygame
import torch

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

    # DataLogger deletes any existing file at its target path, so this
    # session's samples are captured in a scratch file - never directly in
    # data/raw/training_data.json - and only merged into the real dataset
    # by finalize(), which is safe to call repeatedly and backs up the
    # existing file first.
    SESSION_DATA_PATH = os.path.join(
        os.path.dirname(config.DATA_PATH), "_training_session.json"
    )

    def __init__(self, game):
        """Set up scripted entities and start a fresh data-logging session."""
        self.game = game

        self.player = PlayerTraining(game.screen_width, game.screen_height)
        self.enemy = EnemyTrain(game.screen_width, game.screen_height)
        game.player = self.player
        game.enemy = self.enemy

        spawn_entities(game)

        self.data_logger = DataLogger(self.SESSION_DATA_PATH)
        game.data_logger = self.data_logger
        self.finalized = False

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

        # Guide missiles with pure deterministic targeting (model_blend_factor=0)
        # rather than the current (still-learning) model's own guesses - this
        # is what fills in missile.last_action, the supervised training label,
        # so it needs to be a clean "correct" signal to imitate, not the model
        # grading its own homework. It also keeps missile flight paths
        # realistic successful intercepts, giving good state coverage.
        if self.game.missile_model and self.player.missiles:
            update_missile_ai(
                self.player.missiles,
                self.player.position,
                self.enemy.pos,
                self.game._missile_input,
                self.game.missile_model,
                model_blend_factor=0.0,
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

    def finalize(self) -> bool:
        """
        Merge this session's collected samples into the master dataset and
        retrain the missile model from it. Called when leaving Training
        mode (menu return or app exit) - safe to call more than once, and
        a no-op if nothing was collected.

        Deliberately does not touch enemy RL training:
        DataValidatorAndTrainer.process_new_data() would also kick off a
        100k-timestep RL retrain, which is far too slow to run as a side
        effect of leaving a short GUI session.
        """
        if self.finalized:
            return False
        self.finalized = True

        new_data = self.data_logger.data
        if not new_data:
            logging.info("Training mode: no samples collected this session.")
            return False

        # Imported lazily: data_validator_and_trainer imports train_enemy_rl,
        # which imports GameCore, which imports this module - a circular
        # import at module load time if done at the top of this file.
        from ai_platform_trainer.utils.data_validator_and_trainer import (
            DataValidatorAndTrainer,
        )

        validator = DataValidatorAndTrainer(
            training_data_path=config.DATA_PATH,
            missile_model_path=config.MISSILE_MODEL_PATH,
        )
        valid, error_msg = validator.validate_data_format(new_data)
        if not valid:
            logging.error(
                f"Training mode: collected data failed validation: {error_msg}"
            )
            return False

        validator.backup_existing_data()
        existing_data = validator.load_existing_data()
        if not validator.merge_and_save_data(existing_data, new_data):
            logging.error("Training mode: failed to merge session data into dataset.")
            return False

        total = len(existing_data) + len(new_data)
        logging.info(
            f"Training mode: merged {len(new_data)} new samples "
            f"({total} total). Retraining missile model..."
        )
        if validator.train_missile_model():
            logging.info("Training mode: missile model retrained successfully.")
            self._reload_missile_model()
        else:
            logging.error("Training mode: missile model retraining failed.")

        return True

    def _reload_missile_model(self) -> None:
        """
        Hot-reload the just-retrained weights into the model object the
        running game already holds a reference to, so the improvement is
        visible immediately without restarting the app.
        """
        if self.game.missile_model is None:
            return
        try:
            state_dict = torch.load(config.MISSILE_MODEL_PATH, map_location="cpu")
            self.game.missile_model.load_state_dict(state_dict)
            self.game.missile_model.eval()
            logging.info("Training mode: reloaded updated missile model weights.")
        except Exception as e:
            logging.error(f"Training mode: failed to reload retrained model: {e}")

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
