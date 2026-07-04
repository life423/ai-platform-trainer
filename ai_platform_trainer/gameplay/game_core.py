"""
Core Game class for AI Platform Trainer.
"""
import logging
import math
from typing import Any, Optional, Union

import pygame
import torch

# AI imports
from ai_platform_trainer.ai.missile_ai_loader import missile_ai_manager
from ai_platform_trainer.core.config_manager import get_config_manager

# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger

# Logging setup
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.enemy_learning import AdaptiveStagedEnemyAI
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.player_play import PlayerPlay

# Gameplay imports
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.display_manager import DisplayManager
from ai_platform_trainer.gameplay.input_handler import InputHandler
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode
from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import (
    respawn_enemy_with_fade_in,
    spawn_entities,
)


class GameCore:
    """Owns the main game loop, menu, and mode managers (Play / Train)."""

    def __init__(self) -> None:
        """Initialize the game."""
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.paused: bool = False

        # Get configuration manager
        self.config_manager = get_config_manager()

        # Use fullscreen mode for the game
        self.config_manager.set("display.fullscreen", True)
        self.config_manager.save()

        # Initialize display - DisplayManager calls pygame.init()
        self.display_manager = DisplayManager(
            fullscreen=self.config_manager.get("display.fullscreen", True)
        )
        self.screen = self.display_manager.get_screen()
        (
            self.screen_width,
            self.screen_height,
        ) = self.display_manager.get_dimensions()

        # Initialize ScreenContext with actual screen dimensions
        ScreenContext.initialize(self.screen_width, self.screen_height)

        # Create clock, menu, and renderer
        self.clock = pygame.time.Clock()
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)

        # Entities and managers
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[Union[EnemyPlay, AdaptiveStagedEnemyAI]] = None
        self.data_logger: Optional[DataLogger] = None

        self.play_mode_manager: Optional[PlayMode] = None
        self.play_learning_mode_manager: Optional[PlayLearningMode] = None
        self.training_mode_manager: Optional[TrainingMode] = None

        # Use shared missile AI manager for missile models
        self.missile_model = missile_ai_manager.neural_network_model

        # Initialize input handler
        self.input_handler = InputHandler()
        self._setup_input_callbacks()

        # Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        logging.info("Game initialized.")

    def _setup_input_callbacks(self) -> None:
        """Set up input handler callbacks for key events."""

        def handle_keydown(event):
            if event.key == pygame.K_f:
                logging.debug("F pressed - toggling fullscreen.")
                self._toggle_fullscreen()
            elif not self.menu_active:
                if event.key == pygame.K_ESCAPE:
                    logging.info("Escape key pressed. Exiting game.")
                    self.running = False
                elif event.key == pygame.K_SPACE and self.player and self.enemy:
                    logging.debug("Space key pressed in event handler")
                    self.player.shoot_missile(self.enemy.pos)
                elif event.key == pygame.K_m:
                    logging.info("M key pressed. Returning to menu.")
                    if self.mode == "train" and self.training_mode_manager:
                        self.training_mode_manager.finalize()
                    self.menu_active = True
                    self.reset_game_state()

        self.input_handler.register_callback(pygame.KEYDOWN, handle_keydown)

    def run(self) -> None:
        """Main game loop."""
        self._run_standard()

    def _run_standard(self) -> None:
        """Standard game loop without state machine."""
        while self.running:
            current_time = pygame.time.get_ticks()

            # Handle input events
            should_continue, events = self.input_handler.handle_input()
            if not should_continue:
                self.running = False

            # Handle menu-specific events
            if self.menu_active:
                for event in events:
                    if event.type == pygame.KEYDOWN or (
                        event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    ):
                        selected_action = self.menu.handle_menu_events(event)
                        if selected_action:
                            action, payload = selected_action
                            self.check_menu_selection(action, payload)

            if self.menu_active:
                self.menu.draw(self.screen)
                if self.display_manager:
                    self.display_manager.flip()
            else:
                self.update(current_time)
                # Only pass a mode manager if it has UI to draw
                learning_manager: Optional[Any] = None
                if self.mode == "play_learning":
                    learning_manager = self.play_learning_mode_manager
                elif self.mode == "train":
                    learning_manager = self.training_mode_manager
                self.renderer.render(
                    self.menu,
                    self.player,
                    self.enemy,
                    self.menu_active,
                    self.mode,
                    learning_manager,
                )

            # Display flip is handled by renderer in game mode, menu handles its own flip

            self.clock.tick(config.FRAME_RATE)

        # Merge and retrain from collected data if we were training
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.finalize()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(
        self, mode: str, model_choice: str = "sac", enemy_choice: str = "adaptive"
    ) -> None:
        """
        Start the game in the specified mode.

        Args:
            mode: The game mode ("train" or "play_learning")
            model_choice: Which missile guidance model to play against
                ("sac", "ppo", or "supervised") - only relevant for
                "play_learning".
            enemy_choice: Which enemy behavior to face ("adaptive" for the
                scripted staged-difficulty AI, or "trained" for the
                supervised/RL EnemyPlay model) - only relevant for
                "play_learning".
        """
        self.mode = mode
        self.model_choice = model_choice
        self.enemy_choice = enemy_choice
        logging.info(
            f"Starting game in '{mode}' mode "
            f"(missile: {model_choice}, enemy: {enemy_choice})."
        )

        if mode == "play_learning":
            # Play against real-time learning AI
            self.player = PlayerPlay(
                self.screen_width, self.screen_height, model_choice=model_choice
            )
            self.player.reset()

            # Create learning mode manager which will handle enemy creation
            self.play_learning_mode_manager = PlayLearningMode(
                self, enemy_choice=enemy_choice
            )

            # Set the enemy reference for compatibility with other systems
            self.enemy = self.play_learning_mode_manager.learning_enemy

            # Now spawn entities with both player and enemy available
            spawn_entities(self)
        elif mode == "train":
            # Scripted data-collection mode: TrainingMode creates its own
            # player/enemy entities and spawns them.
            self.training_mode_manager = TrainingMode(self)

    def check_menu_selection(
        self, selected_action: str, payload: Optional[dict] = None
    ) -> None:
        """
        Handle menu selection.

        Args:
            selected_action: The selected menu action
            payload: For "play_learning", a dict with "model_choice"
                ("sac"/"ppo"/"supervised") and "enemy_choice"
                ("adaptive"/"trained")
        """
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action == "train":
            logging.info("'train' selected from menu.")
            self.menu_active = False
            self.start_game("train")
        elif selected_action == "play_learning":
            payload = payload or {}
            model_choice = payload.get("model_choice", "sac")
            enemy_choice = payload.get("enemy_choice", "adaptive")
            logging.info(
                "'play_learning' selected from menu "
                f"(missile: {model_choice}, enemy: {enemy_choice})."
            )
            self.menu_active = False
            self.start_game("play_learning", model_choice, enemy_choice)

    def _toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen modes."""
        if not self.display_manager:
            return  # Skip in headless mode

        was_fullscreen = self.config_manager.get("display.fullscreen", False)
        self.display_manager.toggle_fullscreen()
        self.config_manager.set("display.fullscreen", not was_fullscreen)
        self.config_manager.save()

        self.screen = self.display_manager.get_screen()
        self.screen_width, self.screen_height = self.display_manager.get_dimensions()
        self.menu = Menu(self.screen_width, self.screen_height)

        # Update ScreenContext with new dimensions
        ScreenContext.update_dimensions(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(
                current_mode,
                getattr(self, "model_choice", "sac"),
                getattr(self, "enemy_choice", "adaptive"),
            )

    def update(self, current_time: int) -> None:
        """
        Update game state.

        Args:
            current_time: Current game time in milliseconds
        """
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.update(current_time)

        elif self.mode == "play_learning":
            if self.play_learning_mode_manager:
                self.play_learning_mode_manager.update(current_time)
            else:
                self.play_learning_mode_manager = PlayLearningMode(self)
                self.play_learning_mode_manager.update(current_time)

    def check_collision(self) -> bool:
        """
        Check for collision between player and enemy.

        Returns:
            True if collision detected, False otherwise
        """
        if not (self.player and self.enemy):
            return False

        # Make sure enemy is visible
        if not self.enemy.visible:
            return False

        # Ensure pos is a dictionary with x and y keys
        if (
            not isinstance(self.enemy.pos, dict)
            or "x" not in self.enemy.pos
            or "y" not in self.enemy.pos
        ):
            logging.error(f"Invalid enemy position format: {self.enemy.pos}")
            return False

        try:
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
                self.enemy.size,
            )
            return player_rect.colliderect(enemy_rect)
        except TypeError as e:
            logging.error(f"Error in collision detection: {e}")
            return False

    def check_missile_collisions(self) -> None:
        """Check for collisions between missiles and enemy."""
        if not self.enemy or not self.player:
            return

        def respawn_callback() -> None:
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
            logging.info("Missile-Enemy collision in play mode, enemy will respawn.")

        handle_missile_collisions(self.player, self.enemy, respawn_callback)

    def handle_respawn(self, current_time: int) -> None:
        """
        Handle respawning the enemy after a delay.

        Args:
            current_time: Current game time in milliseconds
        """
        if (
            self.is_respawning
            and current_time >= self.respawn_timer
            and self.enemy
            and self.player
        ):
            respawn_enemy_with_fade_in(self, current_time)

    def reset_game_state(self) -> None:
        """Reset game state, typically when returning to menu."""
        self.player = None
        self.enemy = None
        self.data_logger = None
        self.is_respawning = False
        self.respawn_timer = 0
        self.play_mode_manager = None
        self.play_learning_mode_manager = None
        self.training_mode_manager = None
        logging.info("Game state reset, returning to menu.")

    def reset_enemy(self) -> None:
        """
        Reset the enemy's position but keep it in the game.

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
                        (x - self.player.position["x"]) ** 2
                        + (y - self.player.position["y"]) ** 2
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
        """
        Process a single update frame for the game.

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
            if self.play_mode_manager:
                self.play_mode_manager.update(current_time)
            else:
                self.play_mode_manager = PlayMode(self)
                self.play_mode_manager.update(current_time)
