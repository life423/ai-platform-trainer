import pygame
import logging


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize the Renderer.

        :param screen: Pygame display surface
        """
        self.screen = screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue

    def render(self, menu, player, enemy, menu_active: bool) -> None:
        """
        Render the game elements on the screen.

        :param menu: Menu instance
        :param player: Player instance
        :param enemy: Enemy instance
        :param menu_active: Boolean indicating if the menu is active
        """
        try:
            self.screen.fill(self.BACKGROUND_COLOR)
            if menu_active:
                menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                player.draw(self.screen)
                enemy.draw(self.screen)
                logging.debug("Player and Enemy rendered.")
            pygame.display.flip()
            logging.debug("Frame updated on display.")
        except Exception as e:
            logging.error(f"Error during rendering: {e}")
