"""The headless sim must reproduce Play mode frame for frame.

Each test drives the real pygame entity code and the arena sim with identical
inputs and requires identical results.
"""
import math
import random
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch

import pygame
import pytest

from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.sim import CAUGHT, HIT, ArenaSim
from ai_platform_trainer.arena.state import MissileState
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.smart_missile import SmartMissile
from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode

W, H = 1470, 956
CONFIG = ArenaConfig(width=W, height=H)


def ticks(frame: int) -> int:
    """pygame.time.get_ticks() at a given frame of a steady 60 FPS game."""
    return int(round(frame * 1000.0 / 60.0))


def pressed(dx: int, dy: int):
    keys = defaultdict(bool)
    keys[pygame.K_LEFT] = dx < 0
    keys[pygame.K_RIGHT] = dx > 0
    keys[pygame.K_UP] = dy < 0
    keys[pygame.K_DOWN] = dy > 0
    return keys


def basic_missile(**kwargs) -> SmartMissile:
    """A SmartMissile with no guidance model, i.e. the basic homing fallback."""
    return SmartMissile(
        kwargs["x"],
        kwargs["y"],
        target_x=kwargs["target_x"],
        target_y=kwargs["target_y"],
        speed=kwargs["speed"],
        vx=kwargs["vx"],
        vy=kwargs["vy"],
        birth_time=kwargs["birth_time"],
        lifespan=kwargs["lifespan"],
    )


def test_player_movement_and_wrap_match_play_mode():
    rng = random.Random(0)
    player = PlayerPlay(W, H)
    player.position["x"], player.position["y"] = W - 12, 7
    sim = ArenaSim(CONFIG, random.Random(0))
    sim.reset(player_xy=(W - 12.0, 7.0), enemy_xy=(500.0, 500.0))
    move, hold = (0, 0), 0
    for _ in range(4000):
        if hold <= 0:
            move, hold = (rng.choice((-1, 0, 1)), rng.choice((-1, 0, 1))), rng.randint(
                1, 150
            )
        hold -= 1
        with patch("pygame.key.get_pressed", return_value=pressed(*move)):
            player.handle_input()
        sim.player_phase(move[0], move[1], shoot=False)
        assert player.position["x"] == sim.state.player.x
        assert player.position["y"] == sim.state.player.y


def test_missile_homing_expiry_and_wrap_match_play_mode():
    rng = random.Random(3)
    for _ in range(30):
        life_frames = 3 * rng.randint(50, 60)  # whole milliseconds at 60 FPS
        x, y = rng.uniform(0.0, W), rng.uniform(0.0, H)
        angle = rng.uniform(-math.pi, math.pi)
        vx, vy = 5.0 * math.cos(angle), 5.0 * math.sin(angle)
        sim = ArenaSim(CONFIG, random.Random(0))
        sim.reset(
            player_xy=(0.0, 0.0),
            enemy_xy=(rng.uniform(0, W - 50), rng.uniform(0, H - 50)),
        )
        enemy = sim.state.enemy
        sim.state.missiles = [
            MissileState(
                x, y, vx, vy, life_frames, last_target_x=enemy.x, last_target_y=enemy.y
            )
        ]
        player = PlayerPlay(W, H)
        player.missiles = [
            basic_missile(
                x=x,
                y=y,
                target_x=enemy.x,
                target_y=enemy.y,
                speed=5.0,
                vx=vx,
                vy=vy,
                birth_time=0,
                lifespan=ticks(life_frames),
            )
        ]
        for frame in range(life_frames + 40):
            sim.enemy_phase(rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
            with patch("pygame.time.get_ticks", return_value=ticks(frame)):
                player.update_missiles({"x": enemy.x, "y": enemy.y})
            sim.missile_phase()
            assert len(player.missiles) == len(sim.state.missiles)
            if not player.missiles:
                break
            game, arena = player.missiles[0], sim.state.missiles[0]
            assert game.exploded == arena.exploded
            if not arena.exploded:
                assert game.pos["x"] == pytest.approx(arena.x, abs=1e-9)
                assert game.pos["y"] == pytest.approx(arena.y, abs=1e-9)
                assert game.vx == pytest.approx(arena.vx, abs=1e-9)
                assert game.vy == pytest.approx(arena.vy, abs=1e-9)
        assert (
            not player.missiles
        ), "the missile should have expired and finished exploding"


def test_shooting_rules_match_play_mode():
    made, launched = [], []

    def factory(**kwargs):
        made.append(basic_missile(**kwargs))
        return made[-1]

    player = PlayerPlay(W, H)
    player.position["x"], player.position["y"] = 400, 300
    enemy = {"x": 900.0, "y": 650.0}
    sim = ArenaSim(CONFIG, random.Random(1))
    sim.reset(player_xy=(400.0, 300.0), enemy_xy=(900.0, 650.0))
    with patch(
        "ai_platform_trainer.entities.player_play.create_smart_missile",
        side_effect=factory,
    ):
        for frame in range(120):
            before = len(player.missiles)
            with patch("pygame.time.get_ticks", return_value=10_000 + ticks(frame)):
                player.shoot_missile(enemy)
            game_fired = len(player.missiles) > before
            sim_fired = sim.try_fire()
            assert game_fired == sim_fired, f"frame {frame}"
            if sim_fired:
                m = sim.state.missiles[-1]
                launched.append((m.x, m.y, m.vx, m.vy, m.life_frames))
            sim.missile_phase()
    assert (
        len(made) == len(launched) == 3
    )  # frames 0, 30 and 60, then the 3-missile cap
    aim = math.atan2(650.0 - 325.0, 900.0 - 425.0)
    for game, (x, y, vx, vy, life_frames) in zip(made, launched):
        assert (game.pos["x"], game.pos["y"]) == (425, 325)
        assert (x, y) == (425.0, 325.0)
        for mvx, mvy in ((game.vx, game.vy), (vx, vy)):
            assert math.hypot(mvx, mvy) == pytest.approx(5.0)
            assert abs(math.atan2(mvy, mvx) - aim) <= 0.1 + 1e-9
        assert 2500 <= game.lifespan <= 3000
        assert 150 <= life_frames <= 180


def test_collision_rules_match_play_mode():
    rng = random.Random(5)
    outcomes = set()
    for _ in range(5000):
        px, py = rng.uniform(0.0, 300.0), rng.uniform(0.0, 300.0)
        ex, ey = px + rng.uniform(-80.0, 80.0), py + rng.uniform(-80.0, 80.0)
        mx, my = ex + rng.uniform(-40.0, 70.0), ey + rng.uniform(-40.0, 70.0)
        player = SimpleNamespace(position={"x": px, "y": py}, size=50)
        enemy = SimpleNamespace(pos={"x": ex, "y": ey}, size=50, visible=True)
        missile = SimpleNamespace(pos={"x": mx, "y": my}, size=10)
        if PlayLearningMode._entities_collide(None, player, enemy):
            expected = CAUGHT
        elif PlayLearningMode._missile_enemy_collide(None, missile, enemy):
            expected = HIT
        else:
            expected = ""
        sim = ArenaSim(CONFIG)
        sim.reset(player_xy=(px, py), enemy_xy=(ex, ey))
        sim.state.missiles = [MissileState(mx, my, 5.0, 0.0, 170)]
        assert sim.collision_phase() == expected
        outcomes.add(expected)
    assert outcomes == {CAUGHT, HIT, ""}
