import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        # Main menu options - simplified and hierarchical
        self.main_menu_options = ["Play", "Train", "Help", "Exit"]
        # Play submenu options
        self.play_submenu_options = ["Supervised AI", "Learning AI", "Back"]
        
        # Current menu state
        self.current_menu = "main"  # "main" or "play_submenu"
        self.selected_option = 0

        # Flag to show help screen; if True, the help screen is drawn instead of the main menu
        self.show_help = False

        # Store screen dimensions to position menu items correctly
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors for text rendering and background
        self.font_title = pygame.font.Font(None, 80)  # Slightly smaller title for better balance
        self.font_option = pygame.font.Font(None, 48)  # Reduced font size for cleaner look
        self.font_submenu = pygame.font.Font(None, 36)  # Smaller font for submenu indicator
        self.color_background = (135, 206, 235)  # Light blue background color
        self.color_title = (0, 51, 102)  # Dark blue for title text
        self.color_option = (245, 245, 245)  # Light gray for unselected options
        self.color_selected = (255, 223, 0)  # Yellow for the currently selected option
        self.option_rects = {}  # Store clickable rects for each menu option

    def handle_menu_events(self, event):
        """
        Handle user input for the menu.
        If self.show_help is True, handle help screen.
        Otherwise, process keyboard/mouse events to navigate or select menu options.
        """

        # If the help screen is currently displayed
        if self.show_help:
            # Only check if user pressed ESC or ENTER to exit help
            if event.type == pygame.KEYDOWN and event.key in [
                pygame.K_ESCAPE,
                pygame.K_RETURN,
            ]:
                self.show_help = False
            return None

        # If user presses ENTER (or NUMPAD ENTER) on the menu
        elif event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            return self._handle_selection()

        # Handle arrow keys / WASD for navigating menu options
        if event.type == pygame.KEYDOWN:
            current_options = self._get_current_menu_options()
            if event.key in [pygame.K_UP, pygame.K_w]:
                # Move selection up, wrapping around with modulo
                self.selected_option = (self.selected_option - 1) % len(current_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                # Move selection down, wrapping around with modulo
                self.selected_option = (self.selected_option + 1) % len(current_options)
            elif event.key == pygame.K_ESCAPE:
                # ESC key - go back or exit
                if self.current_menu == "play_submenu":
                    self.current_menu = "main"
                    self.selected_option = 0
                else:
                    return "exit"
            # Pressing ENTER again here is a fallback check
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self._handle_selection()

        # Mouse click detection
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Left mouse click
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Check if the click is within any of the option rectangles
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_option = index
                    return self._handle_selection()

        # If no relevant event, return None
        return None

    def _get_current_menu_options(self):
        """Return the current menu options based on menu state."""
        if self.current_menu == "main":
            return self.main_menu_options
        elif self.current_menu == "play_submenu":
            return self.play_submenu_options
        return []

    def _handle_selection(self):
        """Handle the current menu selection."""
        if self.current_menu == "main":
            chosen = self.main_menu_options[self.selected_option]
            if chosen == "Play":
                # Enter play submenu
                self.current_menu = "play_submenu"
                self.selected_option = 0
                return None
            elif chosen == "Train":
                return "train"
            elif chosen == "Help":
                self.show_help = True
                return None
            elif chosen == "Exit":
                return "exit"
        
        elif self.current_menu == "play_submenu":
            chosen = self.play_submenu_options[self.selected_option]
            if chosen == "Supervised AI":
                return "play_supervised"
            elif chosen == "Learning AI":
                return "play_learning"
            elif chosen == "Back":
                # Return to main menu
                self.current_menu = "main"
                self.selected_option = 0
                return None
        
        return None

    def draw(self, screen):
        """
        Draw the menu on the given Pygame screen.
        If show_help is True, the help screen is drawn instead.
        Otherwise, the main menu is drawn with the title and list of options.
        """

        # If the help screen is active, draw that instead
        if self.show_help:
            self.draw_help(screen)
            return
            
        # Fill the screen background
        screen.fill(self.color_background)

        # Render the appropriate title based on current menu
        if self.current_menu == "main":
            title_text = "Pixel Pursuit"
        else:
            title_text = "Select AI Type"
            
        title_surface = self.font_title.render(title_text, True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 4)
        )
        screen.blit(title_surface, title_rect)

        # Show breadcrumb for submenu
        if self.current_menu == "play_submenu":
            breadcrumb = self.font_submenu.render("Play >", True, self.color_title)
            breadcrumb_rect = breadcrumb.get_rect(
                center=(self.screen_width // 2, self.screen_height // 4 + 50)
            )
            screen.blit(breadcrumb, breadcrumb_rect)

        # Render menu options with improved spacing
        current_options = self._get_current_menu_options()
        start_y = self.screen_height // 2
        spacing = 70  # Increased spacing for better readability
        
        for index, option in enumerate(current_options):
            # Choose highlight color if this option is currently selected
            color = (
                self.color_selected
                if index == self.selected_option
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, start_y + index * spacing)
            )
            # Store this rect for click detection
            self.option_rects[index] = option_rect
            # Draw the menu option on screen
            screen.blit(option_surface, option_rect)


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
            "• Press Escape to quit help or game"
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
            "• Play > Supervised AI: Fight pre-trained neural network",
            "• Play > Learning AI: Watch AI learn and improve in real-time!"
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
            "• Real-time learning creates unique and engaging gameplay"
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