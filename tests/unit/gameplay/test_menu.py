"""
The Play menu must start a game only when every AI it needs is trained, and
must say what is missing otherwise.
"""
import pygame
import pytest

from ai_platform_trainer.gameplay import menu as menu_module
from ai_platform_trainer.gameplay.menu import Menu


@pytest.fixture
def menu():
    pygame.init()
    return Menu(1280, 720)


def _enter():
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)


def _open_missile_menu(menu, model_key):
    menu.show_model_menu = True
    menu.selected_model_index = menu.model_keys.index(model_key)


def test_play_goes_straight_to_the_missile_menu(menu):
    menu.selected_option = 0  # "Play"

    assert menu.handle_menu_events(_enter()) is None

    assert menu.show_model_menu
    assert not hasattr(menu, "show_enemy_menu")


def test_everything_trained_starts_the_game(menu, monkeypatch):
    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: None)
    monkeypatch.setattr(menu_module.missile_ai_manager, "problem_with", lambda key: None)
    _open_missile_menu(menu, "sac")

    assert menu.handle_menu_events(_enter()) == (
        "play",
        {"model_choice": "sac"},
    )


def test_missing_agent_blocks_the_start_and_says_why(menu, monkeypatch):
    monkeypatch.setattr(
        menu_module, "enemy_agent_problem", lambda *a, **k: "AI Agent is not trained yet."
    )
    _open_missile_menu(menu, "sac")

    assert menu.handle_menu_events(_enter()) is None

    assert menu.show_model_menu
    assert "AI Agent is not trained yet" in menu.error_message


def test_missing_missile_model_blocks_the_start_and_says_why(menu, monkeypatch):
    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: None)
    monkeypatch.setattr(
        menu_module.missile_ai_manager,
        "problem_with",
        lambda key: "PPO is not trained yet. Train it with: ...",
    )
    _open_missile_menu(menu, "ppo")

    assert menu.handle_menu_events(_enter()) is None

    assert menu.error_message == "PPO is not trained yet. Train it with: ..."


def test_escape_backs_out_to_the_main_menu(menu):
    _open_missile_menu(menu, "sac")

    menu.handle_menu_events(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    assert not menu.show_model_menu


@pytest.mark.parametrize("show_error", [False, True])
def test_menus_draw_without_error(menu, show_error):
    surface = pygame.Surface((1280, 720))
    menu.draw(surface)

    menu.show_model_menu = True
    menu.error_message = "something is missing" if show_error else None
    menu.draw(surface)
