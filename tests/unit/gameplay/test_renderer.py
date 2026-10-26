"""The Play renderer: sprite caching, the overlay, missile trails and the AI overlay."""

import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from ai_platform_trainer.entities.missile import Missile  # noqa: E402
from ai_platform_trainer.gameplay.renderer import (  # noqa: E402
    TRAIL_LENGTH,
    Renderer,
    facing_for_heading,
)
from ai_platform_trainer.utils.sprite_manager import SpriteManager  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SPRITES_DIR = REPO_ROOT / "assets" / "sprites"


@pytest.fixture(scope="module")
def display_surface():
    pygame.init()
    # Sprites are converted to the display format, so a display mode must exist
    return pygame.display.set_mode((1, 1))


class _Display:
    def __init__(self):
        self.presented = 0

    def present(self, world):
        self.presented += 1


@pytest.fixture
def renderer(display_surface, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)  # the sprite manager looks for assets/ from here
    return Renderer(pygame.Surface((800, 600)), _Display())


def _player(missiles=()):
    return SimpleNamespace(
        position={"x": 100.0, "y": 100.0},
        size=50,
        facing_angle=0.0,
        missiles=list(missiles),
    )


def _enemy():
    return SimpleNamespace(
        pos={"x": 500.0, "y": 300.0},
        size=50,
        visible=True,
        fading_in=False,
        fade_alpha=255,
        _last_move=(0.0, 0.0),
    )


def test_sprites_are_cached_by_name_and_size(display_surface):
    manager = SpriteManager(str(SPRITES_DIR))

    big = manager.load_sprite("player", (50, 50))
    small = manager.load_sprite("player", (20, 20))

    assert big.get_size() == (50, 50)
    assert small.get_size() == (20, 20)
    assert manager.load_sprite("player", (50, 50)) is big


def test_rotated_sprites_are_cached_by_size_and_whole_degree(display_surface):
    manager = SpriteManager(str(SPRITES_DIR))

    turned = manager._get_rotated_sprite("missile", (14, 21), 45)

    assert manager._get_rotated_sprite("missile", (14, 21), 45) is turned
    assert manager._get_rotated_sprite("missile", (14, 21), 45.2) is turned
    assert manager._get_rotated_sprite("missile", (28, 42), 45) is not turned
    assert manager._get_rotated_sprite("missile", (14, 21), 0) is (
        manager.load_sprite("missile", (14, 21))
    )


def test_facing_follows_the_player_convention():
    assert facing_for_heading(0, -1) == 0  # moving up: the sprite is upright
    assert facing_for_heading(1, 0) == -90  # moving right
    assert abs(facing_for_heading(0, 1)) == 180  # moving down
    assert facing_for_heading(0, 0) == 0


def test_render_presents_the_world_and_reuses_the_background(renderer):
    background = renderer._background

    renderer.render(_player(), _enemy())
    renderer.render(_player(), _enemy())

    assert renderer._background is background
    assert renderer.display.presented == 2
    # The top of the arena is navy, so blue dominates red
    top = background.get_at((400, 5))
    assert top.b > top.r


def test_overlay_is_cleared_every_frame(renderer):
    missile = Missile(300, 200, vx=5, vy=0, birth_time=0, lifespan=10**9)
    player = _player([missile])

    # The missile's centre is at (305, 205); the player ship is elsewhere
    renderer.render(player, None)
    assert renderer.overlay.get_at((305, 205)).a > 0

    player.missiles.clear()
    renderer.render(player, None)
    assert renderer.overlay.get_at((305, 205)).a == 0


def test_trail_keeps_the_last_positions_and_is_dropped_with_the_missile(renderer):
    missile = Missile(100, 100, vx=5, vy=0, birth_time=0, lifespan=10**9)
    player = _player([missile])

    for _ in range(40):
        missile.pos["x"] += 5
        renderer.render(player, None)
    assert len(renderer._missile_trails[id(missile)].points) == TRAIL_LENGTH

    player.missiles.clear()
    renderer.render(player, None)
    assert renderer._missile_trails == {}


def test_trail_is_not_joined_across_a_screen_wrap(renderer):
    missile = Missile(100, 100, vx=5, vy=0, birth_time=0, lifespan=10**9)
    player = _player([missile])
    renderer.render(player, None)
    renderer.render(player, None)

    missile.pos["x"] = 700  # the missile wrapped to the other side of the screen
    renderer.render(player, None)

    assert len(renderer._missile_trails[id(missile)].points) == 1


def test_an_exploding_missile_bursts_once(renderer):
    missile = Missile(300, 200, vx=5, vy=0, birth_time=0, lifespan=10**9)
    missile.explode(pygame.time.get_ticks())
    player = _player([missile])

    renderer.render(player, None)
    renderer.render(player, None)

    assert len(renderer.effects.particles) == 14


def test_the_ai_overlay_adds_lines_and_labels_only_when_on(renderer):
    missile = Missile(300, 200, vx=5, vy=0, birth_time=0, lifespan=10**9)
    player = _player([missile])
    enemy = _enemy()
    enemy._last_move = (3.0, 0.0)

    renderer.show_ai_overlay = False
    renderer.render(player, enemy)
    off = pygame.mask.from_surface(renderer.overlay, 0).count()

    renderer.show_ai_overlay = True
    renderer.render(player, enemy)
    on = pygame.mask.from_surface(renderer.overlay, 0).count()

    assert on > off


def test_play_mode_impacts_reach_the_renderer_effects(renderer):
    # The hook only needs the game's renderer, so the mode is built without a model
    from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode

    mode = PlayLearningMode.__new__(PlayLearningMode)
    mode.game = SimpleNamespace(renderer=renderer)

    mode._show_impact((100, 100), (255, 90, 70), burst_count=30, ring_radius=50)

    assert len(renderer.effects.particles) == 30
    assert len(renderer.effects.rings) == 1


class _Keys:
    """Stands in for pygame.key.get_pressed(): only V can be held."""

    def __init__(self):
        self.v_down = False

    def __getitem__(self, key):
        return key == pygame.K_v and self.v_down


def test_v_toggles_the_ai_overlay_once_per_press(renderer):
    from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode

    mode = PlayLearningMode.__new__(PlayLearningMode)
    mode.game = SimpleNamespace(renderer=renderer)
    mode.v_pressed_last_frame = False
    keys = _Keys()

    keys.v_down = True
    mode._toggle_ai_overlay(keys)
    assert renderer.show_ai_overlay is True

    mode._toggle_ai_overlay(keys)  # still held: no second toggle
    assert renderer.show_ai_overlay is True

    keys.v_down = False
    mode._toggle_ai_overlay(keys)
    keys.v_down = True
    mode._toggle_ai_overlay(keys)  # a new press turns it off again
    assert renderer.show_ai_overlay is False
