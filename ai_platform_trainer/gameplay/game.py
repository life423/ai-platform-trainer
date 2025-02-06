# file: ai_platform_trainer/gameplay/game.py
import logging
import os
from typing import Optional, Tuple

import pygame
import torch

from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.ai_model.train_missile_model import SimpleMissileModel

from ai_platform_trainer.gameplay.env.pixel_pursuit_env import PixelPursuitEnv
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import spawn_entities
from config_manager import load_settings, save_settings
from ai_platform_trainer.core.data_logger import DataLogger

# Our new managers
from ai_platform_trainer.gameplay.modes.play_mode import PlayModeManager
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode


class Game:
    """
    Main class to run the Pixel Pursuit game. This references a single environment
    (PixelPursuitEnv) for both 'play' and 'train' modes. The difference is how we
    feed actions (keyboard vs. AI).
    """

    def __init__(self) -> None:
        setup_logging()
        pygame.init()

        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.settings = load_settings("settings.json")

        # Initialize display
        if self.settings.get("fullscreen", False):
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.screen_width, self.screen_height = self.screen.get_size()
        self.clock = pygame.time.Clock()

        # Menu & Renderer
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # Entities (player, enemy)
        self.player = None
        self.enemy = None

        # For data logging or advanced usage
        self.data_logger: Optional[DataLogger] = None

        # The environment
        self.env: Optional[PixelPursuitEnv] = None

        # Manager references
        self.play_mode_manager: Optional[PlayModeManager] = None
        self.training_mode_manager: Optional[TrainingMode] = None

        # Missile model (loaded once at startup)
        self.missile_model = None
        self._load_missile_model_once()

        logging.info("Game initialized.")

    def _load_missile_model_once(self) -> None:
        """
        Looks for 'models/missile_model.pth' and, if found, loads the
        model weights into a SimpleMissileModel instance.
        """
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(f"Found missile model at '{missile_model_path}'. Loading once...")
            try:
                model = SimpleMissileModel()
                # Best practice: load only state_dict, with weights_only=True
                state_dict = torch.load(missile_model_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                self.missile_model = model
            except Exception as e:
                logging.error(f"Failed to load missile model: {e}")
                self.missile_model = None
        else:
            logging.warning(f"No missile model found at '{missile_model_path}'.")

    def run(self) -> None:
        """
        Main game loop. Depending on mode, update either training or play manager, then render.
        """
        while self.running:
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                # 'train' or 'play' update
                if self.mode == "train" and self.training_mode_manager:
                    self.training_mode_manager.update()
                elif self.mode == "play" and self.play_mode_manager:
                    self.play_mode_manager.update()

                # RENDER (the environment or your old approach)
                self.renderer.render(
                    self.menu, self.player, self.enemy, self.menu_active
                )

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        # If in training mode, save data if needed
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited.")

    def start_game(self, mode: str) -> None:
        """
        Initialize game in 'train' or 'play' mode, set up environment, managers, etc.
        """
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        # Common data logger usage:
        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
        else:
            self.data_logger = None

        # Create player, enemy
        if mode == "train":
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)
        else:
            self.player, self.enemy = self._init_play_mode()

        spawn_entities(self)
        self.player.reset()

        # Create the environment once
        self.env = PixelPursuitEnv(self.screen_width, self.screen_height)
        self.env.reset(self.player, self.enemy, data_logger=self.data_logger)

        # Create the mode manager
        if mode == "train":
            self.training_mode_manager = TrainingMode(self)
            self.play_mode_manager = None
        else:
            self.play_mode_manager = PlayModeManager(self)
            self.training_mode_manager = None

        # Close menu
        self.menu_active = False

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Dynamically imports EnemyMovementModel to handle enemy AI during 'play' mode.
        """
        from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import (
            EnemyMovementModel,
        )

        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        model_path = config.MODEL_PATH
        if os.path.isfile(model_path):
            try:
                # Best practice: load only state_dict
                state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.eval()
                logging.info("Enemy AI model loaded for play mode.")
            except Exception as e:
                logging.error(f"Failed to load enemy model: {e}")
        else:
            logging.warning(f"No enemy model found at '{model_path}'. Using default.")

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        return player, enemy

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected.")
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_f:
                    self.toggle_fullscreen()

                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    # Possibly handle in-game key logic or pass to manager
                    if event.key == pygame.K_m:
                        # Return to menu
                        self.menu_active = True
                        self.reset_game_state()

    def check_menu_selection(self, selected_action: str) -> None:
        if selected_action == "exit":
            logging.info("Exit selected.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"Selected: {selected_action}")
            self.start_game(selected_action)

    def toggle_fullscreen(self) -> None:
        self.settings["fullscreen"] = not self.settings.get("fullscreen", False)
        save_settings(self.settings, "settings.json")
        flags = pygame.FULLSCREEN if self.settings["fullscreen"] else 0
        if self.settings["fullscreen"]:
            self.screen = pygame.display.set_mode((0, 0), flags)
        else:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE, flags)
        self.screen_width, self.screen_height = self.screen.get_size()
        logging.info(
            f"Fullscreen toggled. Now {self.screen_width}x{self.screen_height}."
        )
        self.menu = Menu(self.screen_width, self.screen_height)

    def reset_game_state(self) -> None:
        """
        Clear entities and data logger if returning to menu or toggling resolution.
        """
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.play_mode_manager = None
        self.training_mode_manager = None
        logging.info("Game state reset, back to menu.")