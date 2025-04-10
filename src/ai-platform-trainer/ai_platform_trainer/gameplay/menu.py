import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        # Menu options displayed in the main menu
        self.menu_options = ["Play", "Train", "Help", "Exit"]
        # Tracks which menu option is currently selected (for keyboard input)
        self.selected_option = 0

        # Flag to show help screen; if True, the help screen is drawn instead of the main menu
        self.show_help = False

        # Store screen dimensions to position menu items correctly
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors for text rendering and background
        self.font_title = pygame.font.Font(None, 100)  # Large font for the main title
        self.font_option = pygame.font.Font(None, 60)  # Font for menu options
        self.color_background = (135, 206, 235)  # Light blue background color
        self.color_title = (0, 51, 102)  # Dark blue for title text
        self.color_option = (245, 245, 245)  # Light gray for unselected options
        self.color_selected = (255, 223, 0)  # Yellow for the currently selected option
        self.option_rects = {}  # Store clickable rects for each menu option

    def handle_menu_events(self, event):
        """
        Handle user input for the menu.
        If self.show_help is True, only check for key presses to exit the help screen.
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

        # If user presses ENTER (or NUMPAD ENTER) on the main menu
        elif event.type == pygame.KEYDOWN and event.key in [
            pygame.K_RETURN,
            pygame.K_KP_ENTER,
        ]:
            chosen = self.menu_options[self.selected_option]
            if chosen == "Help":
                # Switch to the help screen
                self.show_help = True
                return None  # Don't return "help" because we handle it internally
            else:
                # Return the chosen option in lowercase (e.g., "play", "train", "exit")
                return chosen.lower()

        # Handle arrow keys / WASD for navigating menu options
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                # Move selection up, wrapping around with modulo
                self.selected_option = (self.selected_option - 1) % len(
                    self.menu_options
                )
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                # Move selection down, wrapping around with modulo
                self.selected_option = (self.selected_option + 1) % len(
                    self.menu_options
                )
            # Pressing ENTER again here is a fallback check
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self.menu_options[self.selected_option].lower()
            # We let the game itself handle ESC for exiting the program

        # Mouse click detection
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Left mouse click
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Check if the click is within any of the option rectangles
            for index, rect in self.option_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    self.selected_option = index
                    chosen = self.menu_options[self.selected_option]
                    if chosen == "Help":
                        self.show_help = True
                        return None
                    else:
                        return chosen.lower()

        # If no relevant event, return None
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

        # Render the game title
        title_surface = self.font_title.render("Pixel Pursuit", True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )
        screen.blit(title_surface, title_rect)

        # Render menu options
        mouse_x, mouse_y = (
            pygame.mouse.get_pos()
        )  # For any mouse-based highlighting, if desired
        for index, option in enumerate(self.menu_options):
            # Choose highlight color if this option is currently selected
            color = (
                self.color_selected
                if index == self.selected_option
                else self.color_option
            )
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80)
            )
            # Store this rect for click detection
            self.option_rects[index] = option_rect
            # Draw the menu option on screen
            screen.blit(option_surface, option_rect)

    def draw_help(self, screen):
        """
        Draw a dedicated Help / Controls screen.
        This screen instructs the user on game controls and how to return to the main menu.
        """

        # Clear the screen with the background color
        screen.fill(self.color_background)

        # Title for the Help screen
        help_title_surface = self.font_title.render(
            "Help / Controls", True, self.color_title
        )
        help_title_rect = help_title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5)
        )
        screen.blit(help_title_surface, help_title_rect)

        # Text lines describing the controls and how to exit
        controls_text = [
            "Press Arrow Keys or W/S to navigate menu.",
            "Control Player with Arrow Keys or WASD.",
            "Press Space to shoot missiles.",
            "Press Enter to select.",
            "Press F to toggle fullscreen.",
            "Press M to return to the menu.",
            "Press Escape to quit this help screen or the game.",
        ]

        # Draw each line of text, spacing them out vertically
        for i, line in enumerate(controls_text):
            help_surface = self.font_option.render(line, True, self.color_option)
            help_rect = help_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + i * 60)
            )
            screen.blit(help_surface, help_rect)
