"""
Main menu for AI Platform Trainer.

Simple menu system for selecting game modes.
"""
import pygame
from typing import Optional, List


class Menu:
    """Main menu for game mode selection."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Menu options
        self.options = [
            ("Play Mode", "play"),
            ("RL Training", "rl_training"),
            ("Supervised Training", "supervised_training"),
            ("Exit", "exit")
        ]
        
        self.selected_index = 0
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Colors
        self.bg_color = (20, 20, 40)
        self.title_color = (255, 255, 255)
        self.selected_color = (255, 255, 0)
        self.normal_color = (200, 200, 200)
        self.description_color = (150, 150, 150)
        
        # Menu descriptions
        self.descriptions = {
            "play": "Human player vs trained AI enemy",
            "rl_training": "Watch enemy learn in real-time using reinforcement learning",
            "supervised_training": "Collect gameplay data and train AI models",
            "exit": "Quit the application"
        }
    
    def handle_event(self, event) -> Optional[str]:
        """Handle menu events. Returns selected action or None."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_index = (self.selected_index - 1) % len(self.options)
            elif event.key == pygame.K_DOWN:
                self.selected_index = (self.selected_index + 1) % len(self.options)
            elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                return self.options[self.selected_index][1]
        
        elif event.type == pygame.MOUSEMOTION:
            # Update selection based on mouse position
            mouse_y = event.pos[1]
            menu_start_y = self.screen_height // 2 - 50
            
            for i, _ in enumerate(self.options):
                option_y = menu_start_y + i * 60
                if option_y <= mouse_y <= option_y + 50:
                    self.selected_index = i
                    break
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                return self.options[self.selected_index][1]
        
        return None
    
    def render(self, screen: pygame.Surface):
        """Render the menu."""
        # Background
        screen.fill(self.bg_color)
        
        # Title
        title_text = self.font_large.render("AI Platform Trainer", True, self.title_color)
        title_rect = title_text.get_rect(center=(self.screen_width // 2, 150))
        screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_text = self.font_small.render("Interactive Machine Learning Training Platform", True, self.description_color)
        subtitle_rect = subtitle_text.get_rect(center=(self.screen_width // 2, 200))
        screen.blit(subtitle_text, subtitle_rect)
        
        # Menu options
        menu_start_y = self.screen_height // 2 - 50
        
        for i, (option_text, option_key) in enumerate(self.options):
            y_pos = menu_start_y + i * 60
            
            # Highlight selected option
            color = self.selected_color if i == self.selected_index else self.normal_color
            
            # Render option text
            option_surface = self.font_medium.render(option_text, True, color)
            option_rect = option_surface.get_rect(center=(self.screen_width // 2, y_pos))
            screen.blit(option_surface, option_rect)
            
            # Show description for selected option
            if i == self.selected_index:
                desc_text = self.descriptions.get(option_key, "")
                if desc_text:
                    desc_surface = self.font_small.render(desc_text, True, self.description_color)
                    desc_rect = desc_surface.get_rect(center=(self.screen_width // 2, y_pos + 25))
                    screen.blit(desc_surface, desc_rect)
        
        # Instructions
        instructions = [
            "Use UP/DOWN arrows or mouse to navigate",
            "Press ENTER/SPACE or click to select"
        ]
        
        instruction_y = self.screen_height - 100
        for instruction in instructions:
            inst_surface = self.font_small.render(instruction, True, self.description_color)
            inst_rect = inst_surface.get_rect(center=(self.screen_width // 2, instruction_y))
            screen.blit(inst_surface, inst_rect)
            instruction_y += 25