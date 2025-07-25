"""
Game Core - Clean Architecture

This module provides the main game engine with a simplified, clean architecture.
"""
import logging
import pygame
from typing import Optional

from config.settings import settings
from config.paths import paths
from .ui.menu import Menu
from .ui.renderer import Renderer
from .modes.play_mode import PlayMode
from .modes.training_mode import TrainingMode


class Game:
    """
    Main game engine with clean architecture.
    
    This class provides the core game loop and manages different game modes.
    """

    def __init__(self):
        """Initialize the game."""
        pygame.init()
        logging.info("Initializing AI Platform Trainer")
        
        # Ensure directories exist
        paths.ensure_directories()
        
        # Display setup
        self.screen_width, self.screen_height = settings.SCREEN_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption(settings.WINDOW_TITLE)
        
        # Core components
        self.clock = pygame.time.Clock()
        self.running = True
        self.menu_active = True
        
        # UI components
        self.menu = Menu(self.screen_width, self.screen_height)
        self.renderer = Renderer(self.screen)
        
        # Game modes
        self.current_mode: Optional[object] = None
        self.mode_name = ""
        
        logging.info("Game initialized successfully")

    def run(self):
        """Main game loop."""
        logging.info("Starting main game loop")
        
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(settings.FRAME_RATE)
            
        pygame.quit()
        logging.info("Game loop exited")

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            if self.menu_active:
                action = self.menu.handle_event(event)
                if action:
                    self.handle_menu_action(action)
            else:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.return_to_menu()
                elif self.current_mode:
                    self.current_mode.handle_event(event)

    def handle_menu_action(self, action: str):
        """Handle menu selections."""
        if action == "play":
            self.start_mode("play")
        elif action == "training":
            self.start_mode("training")
        elif action == "exit":
            self.running = False

    def start_mode(self, mode_name: str):
        """Start a specific game mode."""
        logging.info(f"Starting {mode_name} mode")
        
        self.menu_active = False
        self.mode_name = mode_name
        
        if mode_name == "play":
            self.current_mode = PlayMode(self.screen_width, self.screen_height)
        elif mode_name == "training":
            self.current_mode = TrainingMode(self.screen_width, self.screen_height)

    def return_to_menu(self):
        """Return to main menu."""
        logging.info("Returning to menu")
        self.menu_active = True
        self.current_mode = None
        self.mode_name = ""

    def update(self):
        """Update game state."""
        if not self.menu_active and self.current_mode:
            self.current_mode.update()

    def render(self):
        """Render the game."""
        self.screen.fill(settings.COLOR_BACKGROUND)
        
        if self.menu_active:
            self.menu.render(self.screen)
        elif self.current_mode:
            self.current_mode.render(self.screen)
            
        pygame.display.flip()