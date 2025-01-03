import pygame


class Menu:
    def __init__(self, screen_width, screen_height):
        # Menu options
        self.menu_options = ["Play", "Train", "Exit"]
        self.selected_option = 0

        # Screen dimensions
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Fonts and colors
        self.font_title = pygame.font.Font(None, 100)  # Font for the title
        self.font_option = pygame.font.Font(None, 60)  # Font for menu options
        self.color_background = (135, 206, 235)  # Light blue background
        self.color_title = (0, 51, 102)  # Dark blue title color
        # Light gray for unselected options
        self.color_option = (245, 245, 245)
        self.color_selected = (255, 223, 0)  # Yellow for selected option

    def handle_menu_events(self, event):
        # Handle menu navigation using arrow keys or WASD
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_UP, pygame.K_w]:
                self.selected_option = (
                    self.selected_option - 1) % len(self.menu_options)
            elif event.key in [pygame.K_DOWN, pygame.K_s]:
                self.selected_option = (
                    self.selected_option + 1) % len(self.menu_options)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                return self.menu_options[self.selected_option].lower()
            elif event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
        return None

    def draw(self, screen):
        # Fill screen with background color
        screen.fill(self.color_background)

        # Render the title
        title_surface = self.font_title.render(
            "Pixel Pursuit", True, self.color_title)
        title_rect = title_surface.get_rect(
            center=(self.screen_width // 2, self.screen_height // 5))
        screen.blit(title_surface, title_rect)

        # Render menu options with consistent spacing
        for index, option in enumerate(self.menu_options):
            color = self.color_selected if index == self.selected_option else self.color_option
            option_surface = self.font_option.render(option, True, color)
            option_rect = option_surface.get_rect(
                center=(self.screen_width // 2, self.screen_height // 2 + index * 80))
            screen.blit(option_surface, option_rect)
