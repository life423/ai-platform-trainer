import pygame

from ai_platform_trainer.ai.missile_ai_loader import MODEL_CHOICES, missile_ai_manager
from ai_platform_trainer.entities.enemy_play import (
    ENEMY_CHOICES,
    is_trained_enemy_available,
    is_trained_enemy_rl_available,
)


class Menu:
    def __init__(self, screen_width, screen_height):
        # Main menu options
        self.main_menu_options = ["Play", "Train", "Help", "Exit"]
        self.selected_option = 0

        # Submenu shown after choosing "Play", letting the player pick which
        # enemy behavior to face: the scripted staged-difficulty AI, or the
        # trained (supervised NN, optionally RL) EnemyPlay model.
        self.show_enemy_menu = False
        self.enemy_keys = list(ENEMY_CHOICES.keys())  # ["adaptive", "trained"]
        self.selected_enemy_index = 0
        self.selected_enemy_choice = "adaptive"
        self.enemy_option_rects = {}

        # Submenu shown after choosing the enemy, letting the player pick
        # which missile guidance model to play against.
        self.show_model_menu = False
        self.model_keys = list(MODEL_CHOICES.keys())  # ["sac", "ppo", "supervised"]
        self.selected_model_index = 0
        self.model_option_rects = {}

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
        self.color_unavailable = (120, 120, 120)
        self.option_rects = {}  # Store clickable rects for each menu option

    def handle_menu_events(self, event):
        """
        Handle user input for the menu.

        Returns None, or an (action, payload) tuple where action is one of
        "play_learning" / "train" / "exit". For "play_learning", payload is
        a dict with "enemy_choice" ("adaptive"/"trained") and "model_choice"
        ("sac"/"ppo"/"supervised"); otherwise payload is None.
        """
        # If the help screen is currently displayed
        if self.show_help:
            if event.type == pygame.KEYDOWN and event.key in [
                pygame.K_ESCAPE,
                pygame.K_RETURN,
            ]:
                self.show_help = False
            return None

        # If the "choose your opponent" submenu is currently displayed
        if self.show_model_menu:
            return self._handle_model_menu_event(event)

        # If the "choose your enemy" submenu is currently displayed
        if self.show_enemy_menu:
            return self._handle_enemy_menu_event(event)

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
            # Don't start the game yet - let the player pick an enemy, then
            # a missile model, first.
            self.show_enemy_menu = True
            self.selected_enemy_index = 0
            return None
        elif chosen == "Train":
            return ("train", None)
        elif chosen == "Help":
            self.show_help = True
            return None
        elif chosen == "Exit":
            return ("exit", None)
        return None

    def _handle_enemy_menu_event(self, event):
        """Handle input while the "choose your enemy" submenu is shown."""
        if event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            return self._handle_enemy_selection()

        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_enemy_index = (self.selected_enemy_index - 1) % len(
                    self.enemy_keys
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_enemy_index = (self.selected_enemy_index + 1) % len(
                    self.enemy_keys
                )
            elif event.key == pygame.K_ESCAPE:
                # Back out to the main menu instead of exiting the game.
                self.show_enemy_menu = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.enemy_option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_enemy_index = index
                    return self._handle_enemy_selection()

        return None

    def _handle_enemy_selection(self):
        """Confirm the highlighted enemy, then move on to the missile submenu."""
        self.selected_enemy_choice = self.enemy_keys[self.selected_enemy_index]
        self.show_enemy_menu = False
        self.show_model_menu = True
        self.selected_model_index = 0
        return None

    def _handle_model_menu_event(self, event):
        """Handle input while the "choose your opponent" submenu is shown."""
        if event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            return self._handle_model_selection()

        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_model_index = (self.selected_model_index - 1) % len(
                    self.model_keys
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_model_index = (self.selected_model_index + 1) % len(
                    self.model_keys
                )
            elif event.key == pygame.K_ESCAPE:
                # Back one level, to the enemy submenu, not all the way out.
                self.show_model_menu = False
                self.show_enemy_menu = True

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for index, rect in self.model_option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_model_index = index
                    return self._handle_model_selection()

        return None

    def _handle_model_selection(self):
        """Confirm the highlighted opponent model and start the game."""
        model_choice = self.model_keys[self.selected_model_index]
        self.show_model_menu = False
        return (
            "play_learning",
            {
                "model_choice": model_choice,
                "enemy_choice": self.selected_enemy_choice,
            },
        )

    def draw(self, screen):
        """
        Draw the menu on the given Pygame screen.
        If show_help is True, the help screen is drawn instead.
        """
        if self.show_help:
            self.draw_help(screen)
            return

        if self.show_model_menu:
            self.draw_model_menu(screen)
            return

        if self.show_enemy_menu:
            self.draw_enemy_menu(screen)
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
            color = (
                self.color_selected
                if index == self.selected_option
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, start_y + index * spacing)
            )
            self.option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)

    def draw_enemy_menu(self, screen):
        """Draw the "choose your enemy" submenu shown after selecting Play."""
        screen.fill(self.color_background)

        title_surface = self.font_title.render(
            "Choose Your Enemy", True, self.color_title
        )
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 4)
        )
        screen.blit(title_surface, title_rect)

        start_y = self.screen_height // 2
        spacing = 70

        for index, key in enumerate(self.enemy_keys):
            label = ENEMY_CHOICES[key]
            available = key != "trained" or is_trained_enemy_available()

            if key == "trained" and available:
                label += (
                    " (NN + RL)"
                    if is_trained_enemy_rl_available()
                    else " (Neural Network)"
                )
            if not available:
                label += " (not trained yet - falls back to Adaptive)"

            if index == self.selected_enemy_index:
                color = self.color_selected
            elif not available:
                color = self.color_unavailable
            else:
                color = self.color_option

            font = self.font_option if available else self.font_small
            option_surface = font.render(label, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, start_y + index * spacing)
            )
            self.enemy_option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)

        hint_surface = self.font_small.render(
            "Press ESC to go back", True, self.color_title
        )
        hint_rect = hint_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height - 50)
        )
        screen.blit(hint_surface, hint_rect)

    def draw_model_menu(self, screen):
        """Draw the "choose your opponent" submenu shown after selecting Play."""
        screen.fill(self.color_background)

        title_surface = self.font_title.render(
            "Choose Your Opponent", True, self.color_title
        )
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 4)
        )
        screen.blit(title_surface, title_rect)

        start_y = self.screen_height // 2
        spacing = 70

        for index, key in enumerate(self.model_keys):
            label = MODEL_CHOICES[key]
            available = missile_ai_manager.is_model_available(key)
            if not available:
                label += " (not trained yet - falls back to SAC)"

            if index == self.selected_model_index:
                color = self.color_selected
            elif not available:
                color = self.color_unavailable
            else:
                color = self.color_option

            font = self.font_option if available else self.font_small
            option_surface = font.render(label, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, start_y + index * spacing)
            )
            self.model_option_rects[index] = option_rect
            screen.blit(option_surface, option_rect)

        hint_surface = self.font_small.render(
            "Press ESC to go back", True, self.color_title
        )
        hint_rect = hint_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height - 50)
        )
        screen.blit(hint_surface, hint_rect)

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
            "• Train: Collect gameplay data for AI training",
            "• Play: pick your enemy, then which AI guides your missiles",
            "• Adaptive Staged: hand-tuned enemy that ramps up difficulty",
            "• Trained AI: enemy driven by a trained network (+ RL if available)",
            "• SAC / PPO: reinforcement learning (self-taught by trial and error)",
            "• Supervised NN: neural net trained by imitating logged missile data",
        ]

        # Draw modes text
        y_offset = self.screen_height // 2 + 40
        for i, line in enumerate(modes_text):
            mode_surface = font_info.render(line, True, self.color_option)
            mode_rect = mode_surface.get_rect(
                topleft=(self.screen_width // 10, y_offset + i * 30)
            )
            screen.blit(mode_surface, mode_rect)

        # AI Training section
        ai_title = self.font_option.render("AI Training:", True, self.color_selected)
        ai_rect = ai_title.get_rect(
            topleft=(self.screen_width // 10, y_offset + len(modes_text) * 30 + 20)
        )
        screen.blit(ai_title, ai_rect)

        # Text explaining the AI training
        ai_text = [
            "• Supervised AI: Pre-trained neural network with fixed behavior",
            "• Learning AI: Starts dumb and learns every frame during gameplay",
            "• Watch the Learning AI evolve from simple chase to smart tactics",
            "• Learning AI gets better at avoiding missiles and catching player",
            "• Each session starts fresh - AI learns from scratch every time",
            "• Real-time learning creates unique and engaging gameplay",
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
