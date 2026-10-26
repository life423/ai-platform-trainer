"""
The Play menu must start a game only when every AI it needs is trained, and
must say what is missing otherwise. The game itself checks the same models.
"""
import pygame
import pytest

from ai_platform_trainer.ai.model_paths import MissingModelError
from ai_platform_trainer.gameplay import game_core
from ai_platform_trainer.gameplay import menu as menu_module
from ai_platform_trainer.gameplay.game_core import GameCore
from ai_platform_trainer.gameplay.menu import Menu


@pytest.fixture
def menu():
    pygame.init()
    return Menu(1280, 720)


def _enter():
    return pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)


def _choose_play(menu):
    menu.selected_option = 0  # "Play"


def test_everything_trained_starts_the_game(menu, monkeypatch):
    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: None)
    monkeypatch.setattr(
        menu_module.missile_ai_manager, "problem_with", lambda *a, **k: None
    )
    _choose_play(menu)

    assert menu.handle_menu_events(_enter()) == ("play", None)


def test_missing_agent_blocks_the_start_and_says_why(menu, monkeypatch):
    monkeypatch.setattr(
        menu_module, "enemy_agent_problem", lambda *a, **k: "AI Agent is not trained yet."
    )
    monkeypatch.setattr(
        menu_module.missile_ai_manager, "problem_with", lambda *a, **k: None
    )
    _choose_play(menu)

    assert menu.handle_menu_events(_enter()) is None

    assert "AI Agent is not trained yet" in menu.error_message


def test_missing_sac_missile_blocks_the_start_and_says_why(menu, monkeypatch):
    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: None)
    monkeypatch.setattr(
        menu_module.missile_ai_manager,
        "problem_with",
        lambda *a, **k: "SAC is not trained yet. Train it with: ...",
    )
    _choose_play(menu)

    assert menu.handle_menu_events(_enter()) is None

    assert menu.error_message == "SAC is not trained yet. Train it with: ..."


def test_a_successful_play_clears_an_earlier_error(menu, monkeypatch):
    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: "missing")
    _choose_play(menu)
    menu.handle_menu_events(_enter())
    assert menu.error_message == "missing"

    monkeypatch.setattr(menu_module, "enemy_agent_problem", lambda *a, **k: None)

    assert menu.handle_menu_events(_enter()) == ("play", None)
    assert menu.error_message is None


def test_escape_on_the_main_menu_quits(menu):
    event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)

    assert menu.handle_menu_events(event) == ("exit", None)


def test_escape_closes_help(menu):
    menu.show_help = True

    menu.handle_menu_events(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

    assert not menu.show_help


@pytest.mark.parametrize("show_error", [False, True])
def test_menus_draw_without_error(menu, show_error):
    surface = pygame.Surface((1280, 720))
    menu.error_message = "something is missing" if show_error else None
    menu.draw(surface)

    menu.show_help = True
    menu.draw(surface)


def test_start_game_refuses_without_the_sac_missile(monkeypatch):
    """The game checks the models too, not only the menu."""
    monkeypatch.setattr(game_core, "enemy_agent_problem", lambda *a, **k: None)
    monkeypatch.setattr(
        game_core.missile_ai_manager,
        "problem_with",
        lambda *a, **k: "SAC is not trained yet. Train it with: ...",
    )
    core = GameCore.__new__(GameCore)  # Checking the models needs no window

    with pytest.raises(MissingModelError, match="SAC is not trained yet"):
        core.start_game()


def test_a_refused_start_keeps_the_menu_up(monkeypatch):
    monkeypatch.setattr(game_core, "enemy_agent_problem", lambda *a, **k: "missing")
    core = GameCore.__new__(GameCore)
    core.menu_active = True

    with pytest.raises(MissingModelError):
        core.check_menu_selection("play", None)

    assert core.menu_active
