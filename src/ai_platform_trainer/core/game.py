"""
Main game controller for AI Platform Trainer.

This module provides the Game class which manages game state, handles
user events, and coordinates between various subsystems.
"""
import logging
import os
import math
import pygame
import torch
from typing import Optional, Tuple

# Logging setup
from ai_platform_trainer.core.logging_config import setup_logging
from config_manager import load_settings, save_settings

# Gameplay imports
from ai_platform_trainer.physics.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.rendering.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import (
    spawn_entities,
    respawn_enemy_with_fade_in,
)
from ai_platform_trainer.rendering.display_manager import (
    init_pygame_display,
    toggle_fullscreen_display,
)
# New import for missile AI updates
from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai

# AI and model imports
from ai_platform_trainer.ml.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ml.models.missile_model import SimpleMissileModel

# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode


class Game:
    """
    Main class to run the Pixel Pursuit game.
    Manages both training ('train') and play ('play') modes,
    as well as the main loop, event handling, and initialization.
    """

    def __init__(self) -> None:
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None

        # 1) Load user settings
        self.settings = load_settings("settings.json")

        # 2) Initialize Pygame and the display
        (self.screen, self.screen_width, self.screen_height) = init_pygame_display(
            fullscreen=self.settings.get("fullscreen", False)
        )

        # 3) Create clock, menu, and renderer
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # 4) Entities and managers
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None
        self.training_mode_manager: Optional[TrainingMode] = None  # For train mode

        # 5) Load missile model once
        self.missile_model: Optional[SimpleMissileModel] = None
        self._load_missile_model_once()

        # 6) Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        logging.info("Game initialized.")

    def _load_missile_model_once(self) -> None:
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(f"Found missile model at '{missile_model_path}'.")
            logging.info("Loading missile model once...")
            try:
                model = SimpleMissileModel()
                model.load_state_dict(torch.load(missile_model_path, map_location="cpu"))
                model.eval()
                self.missile_model = model
            except Exception as e:
                logging.error(f"Failed to load missile model: {e}")
                self.missile_model = None
        else:
            logging.warning(f"No missile model found at '{missile_model_path}'.")
            logging.warning("Skipping missile AI.")

    def run(self) -> None:
        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                self.menu.draw(self.screen)
            else:
                self.update(current_time)
                self.renderer.render(self.menu, self.player, self.enemy, self.menu_active)

            pygame.display.flip()
            self.clock.tick(config.FRAME_RATE)

        # Save data if we were training
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(self, mode: str) -> None:
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            spawn_entities(self)
            self.player.reset()
            self.training_mode_manager = TrainingMode(self)

        else:  # "play"
            self.player, self.enemy = self._init_play_mode()
            self.player.reset()
            spawn_entities(self)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        # Load the traditional neural network model
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
            model.eval()
            logging.info("Enemy AI model loaded for play mode.")
        except Exception as e:
            logging.error(f"Failed to load enemy model: {e}")
            raise e

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)

        # Check for RL model and try to load if available
        rl_model_path = "models/enemy_rl/final_model.zip"
        if os.path.exists(rl_model_path):
            try:
                success = enemy.load_rl_model(rl_model_path)
                if success:
                    logging.info("Using reinforcement learning model for enemy behavior")
                else:
                    logging.warning("RL model exists but couldn't be loaded.")
                    logging.warning("Falling back to neural network.")
            except Exception as e:
                logging.error(f"Error loading RL model: {e}.")
                logging.error("Using neural network instead.")
        else:
            logging.info("No RL model found, using traditional neural network")

        logging.info("Initialized PlayerPlay and EnemyPlay for play mode.")
        return player, enemy

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Quit event detected. Exiting game.")
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Fullscreen toggling
                if event.key == pygame.K_f:
                    logging.debug("F pressed - toggling fullscreen.")
                    self._toggle_fullscreen()

                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)
                else:
                    if event.key == pygame.K_ESCAPE:
                        logging.info("Escape key pressed. Exiting game.")
                        self.running = False
                    elif event.key == pygame.K_SPACE and self.player:
                        self.player.shoot_missile(self.enemy.pos)
                    elif event.key == pygame.K_m:
                        logging.info("M key pressed. Returning to menu.")
                        self.menu_active = True
                        self.reset_game_state()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.menu_active:
                    selected_action = self.menu.handle_menu_events(event)
                    if selected_action:
                        self.check_menu_selection(selected_action)

    def check_menu_selection(self, selected_action: str) -> None:
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play"]:
            logging.info(f"'{selected_action}' selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def _toggle_fullscreen(self) -> None:
        """
        Helper that toggles between windowed and fullscreen,
        updating self.screen, self.screen_width, self.screen_height.
        """
        was_fullscreen = self.settings["fullscreen"]
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen,
            config.SCREEN_SIZE
        )
        self.settings["fullscreen"] = not was_fullscreen
        save_settings(self.settings, "settings.json")

        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.menu = Menu(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(current_mode)

    def update(self, current_time: int) -> None:
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.update()
        elif self.mode == "play":
            # If we haven't created a play_mode_manager yet, do so now
            if not hasattr(self, 'play_mode_manager') or self.play_mode_manager is None:
                from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
                self.play_mode_manager = PlayMode(self)

            self.play_mode_manager.update(current_time)

    def play_update(self, current_time: int) -> None:
        """
        Main update logic for 'play' mode.
        """
        if self.player and not self.player.handle_input():
            logging.info("Player requested to quit.")
            self.running = False
            return

        if self.enemy:
            try:
                self.enemy.update_movement(
                    self.player.position["x"],
                    self.player.position["y"],
                    self.player.step,
                    current_time,
                )
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.running = False
                return

        # Check if player & enemy overlap
        if self.check_collision():
            logging.info("Collision detected between player and enemy.")
            if self.enemy:
                self.enemy.hide()
            self.is_respawning = True
            self.respawn_timer = current_time + self.respawn_delay
            logging.info("Player-Enemy collision in play mode.")

        # Update missile AI
        if self.missile_model and self.player and self.player.missiles:
            update_missile_ai(
                self.player.missiles,
                self.player.position,
                self.enemy.pos if self.enemy else None,
                self._missile_input,
                self.missile_model
            )

    def check_collision(self) -> bool:
        if not (self.player and self.enemy):
            return False
        player_rect = pygame.Rect(
            self.player.position["x"],
            self.player.position["y"],
            self.player.size,
            self.player.size,
        )
        enemy_rect = pygame.Rect(
            self.enemy.pos["x"],
            self.enemy.pos["y"],
            self.enemy.size,
            self.enemy.size
        )
        return player_rect.colliderect(enemy_rect)

    def check_missile_collisions(self) -> None:
        if not self.enemy or not self.player:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        handle_missile_collisions(self.player, self.enemy, respawn_callback)

    def handle_respawn(self, current_time: int) -> None:
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        logging.info("Game state reset, returning to menu.")

    def reset_enemy(self) -> None:
        """Reset the enemy's position but keep it in the game.

        This is primarily used during RL training to reset the
        environment without disturbing other game elements.
        """
        if self.enemy:
            # Place the enemy at a random location away from the player
            import random
            if self.player:
                # Keep enemy away from player during resets
                while True:
                    x = random.randint(0, self.screen_width - self.enemy.size)
                    y = random.randint(0, self.screen_height - self.enemy.size)

                    # Calculate distance to player
                    distance = math.sqrt(
                        (x - self.player.position["x"])**2 +
                        (y - self.player.position["y"])**2
                    )

                    # Ensure minimum distance
                    min_distance = max(self.screen_width, self.screen_height) * 0.3
                    if distance >= min_distance:
                        break
            else:
                # No player present, just pick a random position
                x = random.randint(0, self.screen_width - self.enemy.size)
                y = random.randint(0, self.screen_height - self.enemy.size)

            self.enemy.set_position(x, y)
            self.enemy.visible = True
            logging.debug(f"Enemy reset to position ({x}, {y})")

    def update_once(self) -> None:
        """Process a single update frame for the game.

        This is used during RL training to advance the game state
        without relying on the main game loop.
        """
        current_time = pygame.time.get_ticks()

        # Process pending events to avoid queue overflow
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Update based on current mode
        if self.mode == "play" and not self.menu_active:
            if hasattr(self, 'play_mode_manager') and self.play_mode_manager:
                self.play_mode_manager.update(current_time)
            else:
                self.play_update(current_time)
