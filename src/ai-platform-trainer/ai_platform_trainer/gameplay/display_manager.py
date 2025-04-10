# file: ai_platform_trainer/gameplay/display_manager.py
import pygame
from typing import Tuple

def init_pygame_display(fullscreen: bool) -> Tuple[pygame.Surface, int, int]:
    """
    Initialize Pygame, create the display surface, and return (surface, width, height).
    """
    pygame.init()
    if fullscreen:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        from ai_platform_trainer.gameplay.config import config
        screen = pygame.display.set_mode(config.SCREEN_SIZE)

    width, height = screen.get_size()
    return screen, width, height

def toggle_fullscreen_display(new_state: bool, windowed_size: Tuple[int, int]) -> Tuple[pygame.Surface, int, int]:
    """
    Helper that toggles fullscreen on/off, returning (surface, width, height).
    """
    if new_state:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        screen = pygame.display.set_mode(windowed_size)
    width, height = screen.get_size()
    return (screen, width, height)