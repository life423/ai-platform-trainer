"""
Core Game class for AI Platform Trainer.

Simplified game engine focused on interactive AI training.
"""
import logging
import pygame
from typing import Optional

from ai_platform_trainer.core.config_manager import get_config_manager
from ai_platform_trainer.game.ui.menu import Menu
from ai_platform_trainer.game.ui.renderer import Renderer
from ai_platform_trainer.game.modes.play_mode import PlayMode
from ai_platform_trainer.game.modes.rl_training_mode import RLTrainingMode
from ai_platform_trainer.game.modes.supervised_training_mode import SupervisedTrainingMode


class Game:
    """Simplified game core focused on AI training modes."""
    
    def __init__(self):
        pygame.init()
        self.config = get_config_manager()
        
        # Display setup
        self.screen_width = self.config.get("display.width", 1280)
        self.screen_height = self.config.get("display.height", 720)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("AI Platform Trainer")
        
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
        
    def run(self):
        """Main game loop."""
        while self.running:
            self.handle_events()
            self.update()
            self.render()
            self.clock.tick(60)
            
        pygame.quit()
    
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
        elif action == "rl_training":
            self.start_mode("rl_training")
        elif action == "supervised_training":
            self.start_mode("supervised_training")
        elif action == "exit":
            self.running = False
    
    def start_mode(self, mode_name: str):
        """Start a specific game mode."""
        self.menu_active = False
        self.mode_name = mode_name
        
        if mode_name == "play":
            self.current_mode = PlayMode(self.screen_width, self.screen_height)
        elif mode_name == "rl_training":
            self.current_mode = RLTrainingMode(self.screen_width, self.screen_height)
        elif mode_name == "supervised_training":
            self.current_mode = SupervisedTrainingMode(self.screen_width, self.screen_height)
    
    def return_to_menu(self):
        """Return to main menu."""
        self.menu_active = True
        self.current_mode = None
        self.mode_name = ""
    
    def update(self):
        """Update game state."""
        if not self.menu_active and self.current_mode:
            self.current_mode.update()
    
    def render(self):
        """Render the game."""
        self.screen.fill((0, 0, 0))
        
        if self.menu_active:
            self.menu.render(self.screen)
        elif self.current_mode:
            self.current_mode.render(self.screen)
            
        pygame.display.flip()