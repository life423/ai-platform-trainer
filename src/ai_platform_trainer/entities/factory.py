# file: ai_platform_trainer/entities/entity_factory.py
"""
Entity factory for creating game entities.
"""
import logging
import torch
from ai_platform_trainer.core.interfaces import IEntityFactory
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.ml.model_definition.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.gameplay.config import config


class PlayEntityFactory(IEntityFactory):
    """
    Factory for creating entities in play mode.
    """

    def __init__(self, model_path=None):
        """
        Initialize the factory.

        Args:
            model_path: Path to the enemy AI model
        """
        self.model_path = model_path or config.MODEL_PATH
        self.enemy_model = None
        self._load_enemy_model()

    def _load_enemy_model(self):
        """Load the enemy AI model."""
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
            model.eval()
            self.enemy_model = model
            logging.info("Enemy AI model loaded for play mode.")
        except Exception as e:
            logging.error(f"Failed to load enemy model: {e}")
            raise e

    def create_player(self, screen_width, screen_height):
        """
        Create a player entity for play mode.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            PlayerPlay: A player entity for play mode
        """
        player = PlayerPlay(screen_width, screen_height)
        player.reset()
        return player

    def create_enemy(self, screen_width, screen_height):
        """
        Create an enemy entity for play mode.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            EnemyPlay: An enemy entity for play mode
        """
        if not self.enemy_model:
            self._load_enemy_model()

        enemy = EnemyPlay(screen_width, screen_height, self.enemy_model)
        return enemy


class TrainingEntityFactory(IEntityFactory):
    """
    Factory for creating entities in training mode.
    """

    def create_player(self, screen_width, screen_height):
        """
        Create a player entity for training mode.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            PlayerTraining: A player entity for training mode
        """
        player = PlayerTraining(screen_width, screen_height)
        player.reset()
        return player

    def create_enemy(self, screen_width, screen_height):
        """
        Create an enemy entity for training mode.

        Args:
            screen_width: Width of the screen
            screen_height: Height of the screen

        Returns:
            EnemyTrain: An enemy entity for training mode
        """
        enemy = EnemyTrain(screen_width, screen_height)
        return enemy
