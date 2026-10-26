"""The fixed 800x600 world is scaled onto the window, keeping its shape."""
import pygame

from ai_platform_trainer.gameplay.display_manager import DisplayManager


def test_world_is_scaled_onto_the_window_keeping_its_shape(monkeypatch):
    monkeypatch.setattr(pygame.display, "flip", lambda: None)
    display = DisplayManager.__new__(DisplayManager)
    display.screen = pygame.Surface((1600, 900))
    world = pygame.Surface((800, 600))
    world.fill((255, 0, 0))

    display.present(world)

    # 800x600 onto 1600x900 is 1200x900, centred, with black bars at the sides
    assert display.screen.get_at((800, 450))[:3] == (255, 0, 0)
    assert display.screen.get_at((10, 450))[:3] == (0, 0, 0)
    assert display.screen.get_at((1590, 450))[:3] == (0, 0, 0)
