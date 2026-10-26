"""The game: the window, the menu, and Play mode."""
import logging
from typing import Optional

import pygame

from ai_platform_trainer.ai.missile_ai_loader import missile_ai_manager
from ai_platform_trainer.ai.model_paths import MissingModelError
from ai_platform_trainer.arena.config import WORLD_HEIGHT, WORLD_WIDTH
from ai_platform_trainer.core.config_manager import get_config_manager
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.display_manager import DisplayManager
from ai_platform_trainer.gameplay.input_handler import InputHandler
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import spawn_entities


class GameCore:
    """Owns the window, the menu, and the Play game."""

    def __init__(self) -> None:
        self.running: bool = True
        self.menu_active: bool = True
        self.model_choice: str = "sac"

        self.config_manager = get_config_manager()

        # The window is whatever the display gives us. The game world is always
        # WORLD_WIDTH x WORLD_HEIGHT and is scaled onto the window, so every
        # model sees the coordinates it was trained on.
        self.display_manager = DisplayManager(
            fullscreen=self.config_manager.get("display.fullscreen", False)
        )
        self.screen = self.display_manager.get_screen()
        self.window_width, self.window_height = self.display_manager.get_dimensions()
        self.screen_width, self.screen_height = WORLD_WIDTH, WORLD_HEIGHT
        self.world = pygame.Surface((self.screen_width, self.screen_height))
        ScreenContext.initialize(self.screen_width, self.screen_height)

        self.clock = pygame.time.Clock()
        self.menu = Menu(self.window_width, self.window_height)
        self.renderer = Renderer(self.world, self.display_manager)

        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.play_learning_mode_manager: Optional[PlayLearningMode] = None

        # Set by Play mode when the enemy is caught or hit
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        self.input_handler = InputHandler()
        self._setup_input_callbacks()
        logging.info("Game initialized.")

    def _setup_input_callbacks(self) -> None:
        def handle_keydown(event):
            if event.key == pygame.K_f:
                self._toggle_fullscreen()
            elif not self.menu_active:
                if event.key == pygame.K_ESCAPE:
                    logging.info("Escape pressed. Quitting.")
                    self.running = False
                elif event.key == pygame.K_SPACE and self.player and self.enemy:
                    self.player.shoot_missile(self.enemy.pos)
                elif event.key == pygame.K_m:
                    logging.info("M pressed. Returning to the menu.")
                    self.menu_active = True
                    self.reset_game_state()

        self.input_handler.register_callback(pygame.KEYDOWN, handle_keydown)

    def run(self) -> None:
        """The main loop: menu or game, then display."""
        while self.running:
            current_time = pygame.time.get_ticks()

            should_continue, events = self.input_handler.handle_input()
            if not should_continue:
                self.running = False

            if self.menu_active:
                for event in events:
                    if event.type == pygame.KEYDOWN or (
                        event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                    ):
                        selected = self.menu.handle_menu_events(event)
                        if selected:
                            self.check_menu_selection(*selected)
                self.menu.draw(self.screen)
                self.display_manager.flip()
            else:
                self.update(current_time)
                self.renderer.render(
                    self.player,
                    self.enemy,
                    self.play_learning_mode_manager,
                )

            self.clock.tick(config.FRAME_RATE)

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def start_game(self, model_choice: str = "sac") -> None:
        """Start a Play game against the AI Agent with the chosen missile AI.

        Raises:
            MissingModelError: if the chosen missile model or the AI Agent
                has not been trained.
        """
        self.model_choice = model_choice
        logging.info(f"Starting Play (missile: {model_choice}).")

        # Refuse up front rather than failing when the first missile is fired
        problem = missile_ai_manager.problem_with(model_choice)
        if problem is not None:
            raise MissingModelError(problem)

        self.player = PlayerPlay(
            self.screen_width, self.screen_height, model_choice=model_choice
        )
        self.player.reset()
        self.play_learning_mode_manager = PlayLearningMode(self)
        self.enemy = self.play_learning_mode_manager.learning_enemy
        spawn_entities(self)

    def check_menu_selection(self, action: str, payload: Optional[dict] = None) -> None:
        """Handle a menu choice. For "play", payload has the "model_choice"."""
        if action == "exit":
            logging.info("Exit selected from the menu.")
            self.running = False
        elif action == "play":
            self.menu_active = False
            self.start_game((payload or {}).get("model_choice", "sac"))

    def _toggle_fullscreen(self) -> None:
        """Switch between window and fullscreen, keeping the current game going."""
        was_fullscreen = self.config_manager.get("display.fullscreen", False)
        self.display_manager.toggle_fullscreen()
        self.config_manager.set("display.fullscreen", not was_fullscreen)
        self.config_manager.save()

        self.screen = self.display_manager.get_screen()
        self.window_width, self.window_height = self.display_manager.get_dimensions()
        self.menu = Menu(self.window_width, self.window_height)

    def update(self, current_time: int) -> None:
        if self.play_learning_mode_manager is None:
            self.play_learning_mode_manager = PlayLearningMode(self)
        self.play_learning_mode_manager.update(current_time)

    def reset_game_state(self) -> None:
        """Clear the game, for example when returning to the menu."""
        self.player = None
        self.enemy = None
        self.is_respawning = False
        self.respawn_timer = 0
        self.play_learning_mode_manager = None
        logging.info("Game state reset, returning to the menu.")
