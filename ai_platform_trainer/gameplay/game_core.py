"""
Core Game class for AI Platform Trainer.

This module provides a consolidated implementation of the game that combines
the best aspects of the standard, DI, and state machine approaches.
"""
import logging
import os
import math
import pygame
import torch
from typing import Optional, Tuple, Dict, Any, List

# Logging setup
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.core.config_manager import get_config_manager

# Gameplay imports
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.menu import Menu
from ai_platform_trainer.gameplay.renderer import Renderer
from ai_platform_trainer.gameplay.spawner import (
    spawn_entities,
    respawn_enemy_with_fade_in,
)
from ai_platform_trainer.gameplay.display_manager import (
    init_pygame_display,
    toggle_fullscreen_display,
)

# AI imports
from ai_platform_trainer.ai.inference.missile_controller import update_missile_ai
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.ai.models.missile_model import MissileModel

# Data logger and entity imports
from ai_platform_trainer.core.data_logger import DataLogger
from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.enemy_training import EnemyTrain
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode
from ai_platform_trainer.gameplay.modes.play_mode import PlayMode
from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode


class GameCore:
    """
    Core implementation of the game that combines the best aspects of all approaches.
    
    This class provides a unified implementation that can be used directly or
    extended by other game classes.
    """

    def __init__(self, use_state_machine: bool = False, render_mode=None) -> None:
        """
        Initialize the game.
        
        Args:
            use_state_machine: Whether to use the state machine for game flow control
            render_mode: Rendering mode (FULL or HEADLESS)
        """
        setup_logging()
        self.running: bool = True
        self.menu_active: bool = True
        self.mode: Optional[str] = None
        self.paused: bool = False
        self.render_mode = render_mode

        # Get configuration manager
        self.config_manager = get_config_manager()
        
        # Use fullscreen mode for the game
        self.settings["fullscreen"] = True
        save_settings(self.settings, "settings.json")

        # Initialize Pygame
        pygame.init()
        
        # Initialize display based on render mode
        if self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS:
            # Headless mode - use dummy video driver
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.Surface((1280, 720))
            self.screen_width = 1280
            self.screen_height = 720
        else:
            # Normal mode with display
            (self.screen, self.screen_width, self.screen_height) = init_pygame_display(
                fullscreen=self.settings["fullscreen"]
            )

        # Create clock, menu, and renderer
        if self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS:
            # Headless mode - minimal initialization
            self.clock = pygame.time.Clock()
            self.menu = None
            self.renderer = None
        else:
            # Normal mode with full rendering
            pygame.display.set_caption(config.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            self.menu = Menu(self.screen_width, self.screen_height)
            self.renderer = Renderer(self.screen)

        # Entities and managers
        self.player: Optional[PlayerPlay] = None
        self.enemy: Optional[EnemyPlay] = None
        self.data_logger: Optional[DataLogger] = None
        self.training_mode_manager: Optional[TrainingMode] = None
        self.play_mode_manager: Optional[PlayMode] = None
        self.play_learning_mode_manager: Optional[PlayLearningMode] = None

        # Load missile model once
        self.missile_model: Optional[MissileModel] = None
        self._load_missile_model_once()

        # Additional logic
        self.respawn_delay = 1000
        self.respawn_timer = 0
        self.is_respawning = False

        # Reusable tensor for missile AI input
        self._missile_input = torch.zeros((1, 9), dtype=torch.float32)

        # State machine setup if requested
        self.use_state_machine = use_state_machine
        self.states = {}
        self.current_state = None
        
        if self.use_state_machine:
            self._setup_state_machine()

        logging.info("Game initialized.")

    def _setup_state_machine(self) -> None:
        """Set up the state machine for game flow control."""
        from ai_platform_trainer.gameplay.state_machine import (
            MenuState,
            PlayState,
            TrainingState,
            PausedState,
            GameOverState,
        )
        
        self.states = {
            "menu": MenuState(self),
            "play": PlayState(self),
            "train": TrainingState(self),
            "paused": PausedState(self),
            "game_over": GameOverState(self),
        }
        self.current_state = self.states["menu"]
        self.current_state.enter()

    def _load_missile_model_once(self) -> None:
        """Load the missile AI model once during initialization."""
        missile_model_path = "models/missile_model.pth"
        if os.path.isfile(missile_model_path):
            logging.info(f"Found missile model at '{missile_model_path}'.")
            logging.info("Loading missile model once...")
            try:
                model = MissileModel()
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
        """Main game loop."""
        if self.use_state_machine:
            self._run_state_machine()
        else:
            self._run_standard()

    def _run_standard(self) -> None:
        """Standard game loop without state machine."""
        while self.running:
            current_time = pygame.time.get_ticks()
            self.handle_events()

            if self.menu_active:
                if self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS:
                    # Skip menu rendering in headless mode
                    pass
                else:
                    self.menu.draw(self.screen)
                    pygame.display.flip()
            else:
                self.update(current_time)
                if self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS:
                    # Skip rendering in headless mode
                    pass
                else:
                    # Only pass learning mode manager if we're actually in learning mode
                    learning_manager = self.play_learning_mode_manager if self.mode == "play_learning" else None
                    self.renderer.render(
                        self.menu, 
                        self.player, 
                        self.enemy, 
                        self.menu_active, 
                        self.mode, 
                        learning_manager
                    )

            # Display flip is handled by renderer in game mode, menu handles its own flip
                
            self.clock.tick(config.FRAME_RATE)

        # Save data if we were training
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def _run_state_machine(self) -> None:
        """State machine-based game loop."""
        while self.running:
            delta_time = self.clock.tick(config.FRAME_RATE) / 1000.0
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                
                # Let the current state handle the event
                if self.current_state:
                    next_state = self.current_state.handle_event(event)
                    if next_state:
                        self.transition_to(next_state)
            
            # Update and render the current state
            if self.current_state:
                next_state = self.current_state.update(delta_time)
                if next_state:
                    self.transition_to(next_state)
                
                self.current_state.render(self.renderer)
            
            pygame.display.flip()

        # Save data if we were training
        if self.mode == "train" and self.data_logger:
            self.data_logger.save()

        pygame.quit()
        logging.info("Game loop exited and Pygame quit.")

    def transition_to(self, state_name: str) -> None:
        """
        Transition from the current state to a new state.
        
        Args:
            state_name: The name of the state to transition to
        """
        if not self.use_state_machine:
            logging.warning("Attempted to use state machine when not enabled")
            return
            
        if state_name in self.states:
            logging.info(f"Transitioning from {type(self.current_state).__name__} to {state_name}")
            self.current_state.exit()
            self.current_state = self.states[state_name]
            self.current_state.enter()
        else:
            logging.error(f"Attempted to transition to unknown state: {state_name}")

    def start_game(self, mode: str) -> None:
        """
        Start the game in the specified mode.
        
        Args:
            mode: The game mode ("train", "play_supervised", or "play_learning")
        """
        self.mode = mode
        logging.info(f"Starting game in '{mode}' mode.")

        if mode == "train":
            self.data_logger = DataLogger(config.DATA_PATH)
            self.player = PlayerTraining(self.screen_width, self.screen_height)
            self.enemy = EnemyTrain(self.screen_width, self.screen_height)

            spawn_entities(self)
            self.player.reset()
            self.training_mode_manager = TrainingMode(self)

        elif mode == "play_supervised":
            # Play against pre-trained supervised learning model
            self.player, self.enemy = self._init_supervised_play_mode()
            self.player.reset()
            spawn_entities(self)
            self.play_mode_manager = PlayMode(self)

        elif mode == "play_learning":
            # Play against real-time learning AI
            self.player = PlayerPlay(self.screen_width, self.screen_height)
            self.player.reset()
            
            # Create learning mode manager which will handle enemy creation
            self.play_learning_mode_manager = PlayLearningMode(self)
            
            # Set the enemy reference for compatibility with other systems
            self.enemy = self.play_learning_mode_manager.learning_enemy
            
            # Now spawn entities with both player and enemy available
            spawn_entities(self)

    def _init_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize entities for play mode.
        
        Returns:
            Tuple of (player, enemy) entities
        """
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

    def _init_supervised_play_mode(self) -> Tuple[PlayerPlay, EnemyPlay]:
        """
        Initialize entities for supervised learning play mode.
        
        Returns:
            Tuple of (player, enemy) entities
        """
        # Load the traditional neural network model (no RL)
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        try:
            model.load_state_dict(torch.load(config.MODEL_PATH, map_location="cpu"))
            model.eval()
            logging.info("Supervised AI model loaded for play mode.")
        except Exception as e:
            logging.error(f"Failed to load supervised model: {e}")
            raise e

        player = PlayerPlay(self.screen_width, self.screen_height)
        enemy = EnemyPlay(self.screen_width, self.screen_height, model)
        
        # Force use of neural network (disable RL)
        enemy.use_rl = False
        
        logging.info("Initialized supervised learning play mode.")
        return player, enemy


    def handle_events(self) -> None:
        """Handle pygame events."""
        # Skip event handling in headless mode
        if self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS:
            return
            
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
                    elif event.key == pygame.K_SPACE and self.player and self.enemy:
                        logging.debug("Space key pressed in event handler")
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
        """
        Handle menu selection.
        
        Args:
            selected_action: The selected menu action
        """
        if selected_action == "exit":
            logging.info("Exit action selected from menu.")
            self.running = False
        elif selected_action in ["train", "play_supervised", "play_learning"]:
            logging.info(f"'{selected_action}' selected from menu.")
            self.menu_active = False
            self.start_game(selected_action)

    def _toggle_fullscreen(self) -> None:
        """Toggle between windowed and fullscreen modes."""
        was_fullscreen = self.config_manager.get("display.fullscreen", False)
        new_display, w, h = toggle_fullscreen_display(
            not was_fullscreen,
            config.SCREEN_SIZE
        )
        self.config_manager.set("display.fullscreen", not was_fullscreen)
        self.config_manager.save()

        self.screen = new_display
        self.screen_width, self.screen_height = w, h
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.menu = Menu(self.screen_width, self.screen_height)

        if not self.menu_active:
            current_mode = self.mode
            self.reset_game_state()
            self.start_game(current_mode)

    def update(self, current_time: int) -> None:
        """
        Update game state.
        
        Args:
            current_time: Current game time in milliseconds
        """
        if self.mode == "train" and self.training_mode_manager:
            self.training_mode_manager.update()
        elif self.mode == "play_supervised":
            if self.play_mode_manager:
                self.play_mode_manager.update(current_time)
            else:
                self.play_mode_manager = PlayMode(self)
                self.play_mode_manager.update(current_time)
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
        if not isinstance(self.enemy.pos, dict) or "x" not in self.enemy.pos or "y" not in self.enemy.pos:
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
                self.enemy.size
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
        """
        Process a single update frame for the game.
        
        This is used during RL training to advance the game state
        without relying on the main game loop.
        """
        current_time = pygame.time.get_ticks()

        # Process pending events to avoid queue overflow
        if not (self.render_mode and hasattr(self.render_mode, "HEADLESS") and self.render_mode == self.render_mode.HEADLESS):
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
