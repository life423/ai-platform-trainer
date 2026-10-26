"""
Training Data Collection Mode

Runs the player and enemy on scripted (non-human, non-learning) movement
patterns so their interactions can be logged as supervised training data
for the missile guidance model (see ai/training/train_missile_model.py),
with the whole session visible in the GUI rather than running headless.
"""

import logging
import os
from typing import Dict, List, Tuple

import pygame
import torch

from ai_platform_trainer.ai.inference.missile_controller import steer_toward
from ai_platform_trainer.ai.missile_features import missile_features
from ai_platform_trainer.ai.training.missile_dataset import SCHEMA_VERSION
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.spawner import spawn_entities
from ai_platform_trainer.utils.data_validator_and_trainer import DataValidatorAndTrainer


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

    # Largest turn (degrees per frame) the training labels can contain. The
    # data collector steers at this rate. It is separate from the in-game cap.
    LABEL_MAX_TURN_DEG = 5.0

    # How often (ms) the session is written to its scratch file, so a crash
    # loses at most this much.
    SAVE_INTERVAL_MS = 5000

    # The in-memory session is written here as it goes. Samples only reach
    # data/raw/training_data.json through the validator. Leftovers from a
    # session that did not finish are merged on the next start.
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

        self.validator = DataValidatorAndTrainer(
            training_data_path=config.DATA_PATH,
            missile_model_path=config.MISSILE_MODEL_PATH,
            backup_dir=os.path.join(os.path.dirname(config.DATA_PATH), "backups"),
        )

        # DataLogger replaces the scratch file, so recover any leftover first.
        recovered = self.validator.recover_session_file(self.SESSION_DATA_PATH)
        if recovered:
            logging.info(
                f"Training mode: recovered {recovered} samples from an unfinished session."
            )

        self.data_logger = DataLogger(self.SESSION_DATA_PATH)
        game.data_logger = self.data_logger
        self.finalized = False

        self.last_fire_time = 0
        self.last_save_time = 0

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

        # Record each missile's state before it turns, and the turn that
        # labels it. Logged before collision resolution so each missile keeps
        # its decision for the frame it hits.
        decisions = self._steer_missiles()

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
            features, action = decisions[id(missile)]
            self._log_frame(features, action, missile in hit_missiles, current_time)

        if current_time - self.last_save_time >= self.SAVE_INTERVAL_MS:
            self.data_logger.save()
            self.last_save_time = current_time

        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self.game.handle_respawn(current_time)

        if self.enemy.fading_in:
            self.enemy.update_fade_in(current_time)

    def _steer_missiles(self) -> Dict[int, Tuple[List[float], float]]:
        """
        Steer every missile straight at the enemy.

        Returns, keyed by id(missile), the feature vector the supervised model
        would have seen before this turn, and the turn that was applied. The
        turn is the label. Steering is deterministic and needs no model, so
        data can be collected before a supervised model exists.
        """
        decisions = {}
        for missile in self.player.missiles:
            features = missile_features(
                self.player.position,
                self.enemy.pos,
                missile.pos,
                {"x": missile.vx, "y": missile.vy},
            )
            action = steer_toward(
                missile, self.enemy.pos, max_turn_rate=self.LABEL_MAX_TURN_DEG
            )
            decisions[id(missile)] = (features, action)
        return decisions

    def _log_frame(
        self, features: List[float], action: float, collided: bool, current_time: int
    ) -> None:
        """Log one training sample, in the schema the dataset reads."""
        self.data_logger.log(
            {
                "schema": SCHEMA_VERSION,
                "features": features,
                "missile_action": action,
                "missile_collision": collided,
                "timestamp": current_time,
            }
        )

    def finalize(self, retrain: bool = True) -> bool:
        """
        Merge this session's samples into the master dataset and, if asked,
        retrain the missile model from it. Called when leaving Training mode.

        Safe to call more than once. If the merge fails, the samples stay in
        the session file so nothing is lost.

        Args:
            retrain: Whether to retrain the missile model after merging.
        """
        if self.finalized:
            return False
        self.finalized = True

        new_data = self.data_logger.data
        if not new_data:
            logging.info("Training mode: no samples collected this session.")
            return False

        # Keep the session file current, so it holds everything if the merge fails.
        self.data_logger.save()
        if not self.validator.commit_samples(new_data):
            logging.error(
                "Training mode: samples were not merged; they remain in "
                f"{self.SESSION_DATA_PATH}."
            )
            return False

        self._discard_session_file()
        logging.info(f"Training mode: merged {len(new_data)} new samples.")

        if not retrain:
            return True

        if self.validator.train_missile_model():
            logging.info("Training mode: missile model retrained successfully.")
            self._reload_missile_model()
        else:
            logging.error("Training mode: missile model retraining failed (see above).")

        return True

    def _discard_session_file(self) -> None:
        """Remove the scratch file once its samples are in the dataset."""
        try:
            if os.path.exists(self.SESSION_DATA_PATH):
                os.remove(self.SESSION_DATA_PATH)
        except OSError as e:
            logging.error(
                f"Training mode: could not remove {self.SESSION_DATA_PATH}: {e}"
            )

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
