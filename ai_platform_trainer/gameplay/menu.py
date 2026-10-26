import logging
from typing import Optional

import pygame

from ai_platform_trainer.ai.missile_ai_loader import missile_ai_manager
from ai_platform_trainer.entities.enemy_agent import enemy_agent_problem


class Menu:
    def __init__(self, screen_width, screen_height):
        # Main menu options
        self.main_menu_options = ["Play", "Help", "Exit"]
        self.selected_option = 0

        # Shown on the main menu when Play cannot start because a model is missing
        self.error_message: Optional[str] = None

        # Flag to show help screen
        self.show_help = False

        # Store screen dimensions to position menu items correctly
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors for text rendering and background
        self.font_title = pygame.font.Font(None, 80)
        self.font_option = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 28)
        self.color_background = (135, 206, 235)
        self.color_title = (0, 51, 102)
        self.color_option = (245, 245, 245)
        self.color_selected = (255, 223, 0)
        self.color_error = (180, 0, 0)
        self.color_highlight = (20, 30, 70, 215)  # the pill behind the selection
        self.option_rects = {}  # Store clickable rects for each menu option

    def handle_menu_events(self, event):
        """
        Handle user input for the menu.

        Returns None, or an (action, payload) tuple where action is "play" or
        "exit". The payload is always None.
        """
        # If the help screen is currently displayed
        if self.show_help:
            if event.type == pygame.KEYDOWN and event.key in [
                pygame.K_ESCAPE,
                pygame.K_RETURN,
            ]:
                self.show_help = False
            return None

        # If user presses ENTER on the menu
        if event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            return self._handle_selection()

        # Handle arrow keys for navigating menu options
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_option = (self.selected_option - 1) % len(
                    self.main_menu_options
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_option = (self.selected_option + 1) % len(
                    self.main_menu_options
                )
            elif event.key == pygame.K_ESCAPE:
                return ("exit", None)

        # Mouse click detection
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_option = index
                    return self._handle_selection()

        return None

    def _handle_selection(self):
        """Handle the current main menu selection."""
        chosen = self.main_menu_options[self.selected_option]
        if chosen == "Play":
            return self._handle_play()
        elif chosen == "Help":
            self.show_help = True
            return None
        elif chosen == "Exit":
            return ("exit", None)
        return None

    def _handle_play(self):
        """Start the game, or say on this menu what is missing."""
        problem = enemy_agent_problem()
        if problem is None:
            problem = missile_ai_manager.problem_with()
        if problem is not None:
            self.error_message = problem
            logging.error(problem)
            return None

        self.error_message = None
        return ("play", None)

    def draw(self, screen):
        """
        Draw the menu on the given Pygame screen.
        If show_help is True, the help screen is drawn instead.
        """
        if self.show_help:
            self.draw_help(screen)
            return

        screen.fill(self.color_background)

        # Render the main title
        title_surface = self.font_title.render("Pixel Pursuit", True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 4)
        )
        screen.blit(title_surface, title_rect)

        # Render menu options
        start_y = self.screen_height // 2
        spacing = 70

        for index, option in enumerate(self.main_menu_options):
            selected = index == self.selected_option
            color = self.color_selected if selected else self.color_option
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, start_y + index * spacing)
            )
            self.option_rects[index] = option_rect
            if selected:
                self._draw_highlight(screen, option_rect)
            screen.blit(option_surface, option_rect)

        if self.error_message:
            error_surface = self.font_small.render(
                self.error_message, True, self.color_error
            )
            error_rect = error_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height - 100)
            )
            screen.blit(error_surface, error_rect)

        self._draw_hint(screen, "Up/Down to choose    Enter to select    Esc to quit")

    def _draw_highlight(self, screen, option_rect) -> None:
        """A dark pill behind the selected option."""
        pill = option_rect.inflate(48, 16)
        surface = pygame.Surface(pill.size, pygame.SRCALPHA)
        pygame.draw.rect(
            surface,
            self.color_highlight,
            surface.get_rect(),
            border_radius=pill.height // 2,
        )
        screen.blit(surface, pill.topleft)

    def _draw_hint(self, screen, text: str) -> None:
        """One line of key hints along the bottom of the screen."""
        hint = self.font_small.render(text, True, self.color_title)
        hint_rect = hint.get_rect(
            center=(self.screen_width // 2, self.screen_height - 40)
        )
        screen.blit(hint, hint_rect)

    def draw_help(self, screen):
        """
        Draw a dedicated Help / Controls screen.
        This screen instructs the user on game controls and how to return to the main menu.
        It also provides information about game modes and AI training.
        """

        # Clear the screen with the background color
        screen.fill(self.color_background)

        # Title for the Help screen
        help_title_surface = self.font_title.render(
            "Help / Controls", True, self.color_title
        )
        help_title_rect = help_title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 10)
        )
        screen.blit(help_title_surface, help_title_rect)

        # Create a smaller font for more detailed explanations
        font_info = pygame.font.Font(None, 34)

        # Controls section
        controls_title = self.font_option.render("Controls:", True, self.color_selected)
        controls_rect = controls_title.get_rect(
            topleft=(self.screen_width // 10, self.screen_height // 5)
        )
        screen.blit(controls_title, controls_rect)

        # Text lines describing the controls
        controls_text = [
            "• Arrow Keys or W/S to navigate menu",
            "• Control Player with Arrow Keys or WASD",
            "• Press Space to shoot missiles",
            "• Press Enter to select menu items",
            "• Press F to toggle fullscreen",
            "• Press M to return to the menu",
            "• Press Escape to quit help or game",
        ]

        # Draw controls text
        for i, line in enumerate(controls_text):
            help_surface = font_info.render(line, True, self.color_option)
            help_rect = help_surface.get_rect(
                topleft=(self.screen_width // 10, self.screen_height // 5 + 40 + i * 30)
            )
            screen.blit(help_surface, help_rect)

        # Game Modes section - starts halfway down screen
        modes_title = self.font_option.render("Game Modes:", True, self.color_selected)
        modes_rect = modes_title.get_rect(
            topleft=(self.screen_width // 10, self.screen_height // 2)
        )
        screen.blit(modes_title, modes_rect)

        # Text explaining the game modes
        modes_text = [
            "• Play: the enemy is the AI Agent; the SAC AI guides your missiles",
            "• AI Agent: a PPO reinforcement-learning policy",
            "• SAC: the reinforcement-learning missile AI",
        ]

        # Draw modes text
        y_offset = self.screen_height // 2 + 40
        for i, line in enumerate(modes_text):
            mode_surface = font_info.render(line, True, self.color_option)
            mode_rect = mode_surface.get_rect(
                topleft=(self.screen_width // 10, y_offset + i * 30)
            )
            screen.blit(mode_surface, mode_rect)

        # Missing models section
        ai_title = self.font_option.render("Missing models:", True, self.color_selected)
        ai_rect = ai_title.get_rect(
            topleft=(self.screen_width // 10, y_offset + len(modes_text) * 30 + 20)
        )
        screen.blit(ai_title, ai_rect)

        # Text explaining what happens when a model has not been trained
        ai_text = [
            "• Play refuses to start until every AI it needs is trained",
            "• The menu names the command that trains the missing model",
        ]

        # Draw AI text
        ai_y_offset = y_offset + len(modes_text) * 30 + 60
        for i, line in enumerate(ai_text):
            ai_surface = font_info.render(line, True, self.color_option)
            ai_rect = ai_surface.get_rect(
                topleft=(self.screen_width // 10, ai_y_offset + i * 30)
            )
            screen.blit(ai_surface, ai_rect)

        # Return instructions at bottom
        exit_text = self.font_option.render(
            "Press ESC or ENTER to return to menu", True, self.color_title
        )
        exit_rect = exit_text.get_rect(
            center=(self.screen_width // 2, self.screen_height - 50)
        )
        screen.blit(exit_text, exit_rect)
