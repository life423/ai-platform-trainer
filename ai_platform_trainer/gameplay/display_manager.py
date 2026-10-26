"""
Display manager for the AI Platform Trainer.

This module handles Pygame display initialization and management.
"""
from typing import Tuple

import pygame

from ai_platform_trainer.gameplay.config import config


def init_pygame_display(fullscreen: bool = False) -> Tuple[pygame.Surface, int, int]:
    """
    Initialize Pygame, create the display surface, and return (surface, width, height).

    Args:
        fullscreen: Whether to start in fullscreen mode

    Returns:
        Tuple of (surface, width, height)
    """
    pygame.init()

    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(config.SCREEN_SIZE)

    width, height = screen.get_size()

    # Set window title
    pygame.display.set_caption(config.WINDOW_TITLE)

    return screen, width, height


def toggle_fullscreen_display(
    new_state: bool, windowed_size: Tuple[int, int] = config.SCREEN_SIZE
) -> Tuple[pygame.Surface, int, int]:
    """
    Toggle fullscreen on/off, returning (surface, width, height).

    Args:
        new_state: True for fullscreen, False for windowed
        windowed_size: Size to use when in windowed mode

    Returns:
        Tuple of (surface, width, height)
    """
    if new_state:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(windowed_size)

    width, height = screen.get_size()
    return screen, width, height


class DisplayManager:
    """
    Manages the game display, providing methods for screen operations.
    """

    def __init__(self, fullscreen: bool = False):
        """
        Initialize the display manager.

        Args:
            fullscreen: Whether to start in fullscreen mode
        """
        self.screen, self.width, self.height = init_pygame_display(fullscreen)
        self.fullscreen = fullscreen

    def toggle_fullscreen(self) -> None:
        """Toggle between fullscreen and windowed mode."""
        self.fullscreen = not self.fullscreen
        self.screen, self.width, self.height = toggle_fullscreen_display(
            self.fullscreen, config.SCREEN_SIZE
        )

    def get_dimensions(self) -> Tuple[int, int]:
        """
        Get the current screen dimensions.

        Returns:
            Tuple of (width, height)
        """
        return self.width, self.height

    def get_screen(self) -> pygame.Surface:
        """
        Get the current screen surface.

        Returns:
            The pygame Surface object
        """
        return self.screen

    def flip(self) -> None:
        """Update the full display Surface to the screen."""
        pygame.display.flip()

    def present(self, world: pygame.Surface) -> None:
        """
        Scale the fixed-size game world onto the window, keeping its shape
        (black bars fill the rest), and show it.
        """
        win_w, win_h = self.screen.get_size()
        world_w, world_h = world.get_size()
        scale = min(win_w / world_w, win_h / world_h)
        draw_w, draw_h = int(world_w * scale), int(world_h * scale)
        offset = ((win_w - draw_w) // 2, (win_h - draw_h) // 2)

        self.screen.fill((0, 0, 0))
        self.screen.blit(pygame.transform.smoothscale(world, (draw_w, draw_h)), offset)
        pygame.display.flip()
