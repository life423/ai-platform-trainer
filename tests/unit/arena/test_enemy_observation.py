"""The shared enemy encoder: layout, wrap awareness, and game/sim parity."""
import random

import numpy as np
import pytest

from ai_platform_trainer.arena.config import FRAME_MS, ArenaConfig
from ai_platform_trainer.arena.game_bridge import MotionTracker, snapshot_from_game
from ai_platform_trainer.arena.observations import (
    ENEMY_OBS_FEATURES,
    ENEMY_OBS_SIZE,
    OBS_CLIP,
    POSITION_SCALE,
    build_enemy_observation,
    enemy_action_to_displacement,
)
from ai_platform_trainer.arena.sim import ArenaSim
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.smart_missile import SmartMissile

W, H = 1470, 956
CONFIG = ArenaConfig(width=W, height=H)


def named(obs):
    return dict(zip(ENEMY_OBS_FEATURES, obs))


def test_observation_layout():
    state = ArenaSim(CONFIG, random.Random(0)).reset()
    obs = build_enemy_observation(state)
    assert obs.shape == (ENEMY_OBS_SIZE,)
    assert obs.dtype == np.float32
    assert len(ENEMY_OBS_FEATURES) == ENEMY_OBS_SIZE == len(set(ENEMY_OBS_FEATURES))
    assert np.all(np.abs(obs) <= OBS_CLIP)


def test_player_offset_takes_the_wrapped_short_way():
    sim = ArenaSim(CONFIG)
    state = sim.reset(player_xy=(10.0, 400.0), enemy_xy=(W - 30.0, 400.0))
    obs = named(build_enemy_observation(state))
    # across the right edge: (10 - (W - 30)) wrapped by W + 50 = +90 px
    assert obs["player_dx"] == pytest.approx(90.0 / POSITION_SCALE)
    assert obs["player_dy"] == 0.0


def test_missile_slots_are_nearest_first_and_skip_explosions():
    sim = ArenaSim(CONFIG)
    state = sim.reset(player_xy=(100.0, 100.0), enemy_xy=(700.0, 500.0))
    sim.spawn_missile(1100.0, 500.0, 0.0, 170)
    sim.spawn_missile(800.0, 500.0, 0.0, 170)
    sim.spawn_missile(720.0, 500.0, 0.0, 170).exploded = True
    obs = named(build_enemy_observation(state))
    # enemy center x = 725; missile centers x = 805 and 1105
    assert obs["missile0_dx"] == pytest.approx(80.0 / POSITION_SCALE)
    assert obs["missile1_dx"] == pytest.approx(380.0 / POSITION_SCALE)
    assert obs["missile0_present"] == obs["missile1_present"] == 1.0
    assert obs["missile2_present"] == 0.0
    assert obs["player_missile_count"] == pytest.approx(
        1.0
    )  # explosions count toward the cap


def test_action_mapping_matches_the_player_speed_cap():
    assert enemy_action_to_displacement([1.0, -1.0], CONFIG) == (5.0, -5.0)
    assert enemy_action_to_displacement([3.0, 0.5], CONFIG) == (5.0, 2.5)


def test_motion_tracker_measures_displacement_like_the_sim():
    sim = ArenaSim(CONFIG)
    sim.reset(player_xy=(W - 3.0, 5.0), enemy_xy=(700.0, 500.0))
    tracker = MotionTracker(CONFIG)
    tracker.update(sim.state.player.x, sim.state.player.y)
    for _ in range(60):  # crosses the right and top edges
        sim.player_phase(1, -1, shoot=False)
        velocity = tracker.update(sim.state.player.x, sim.state.player.y)
        assert velocity == (sim.state.player.vx, sim.state.player.vy)


def test_game_snapshot_encodes_identically_to_sim_state():
    rng = random.Random(11)
    for _ in range(200):
        sim = ArenaSim(CONFIG, rng)
        state = sim.reset()
        for _ in range(rng.randint(0, 3)):
            missile = sim.spawn_missile(
                rng.uniform(0, W),
                rng.uniform(0, H),
                rng.uniform(-3, 3),
                rng.randint(150, 180),
            )
            missile.age_frames = rng.randint(0, 149)
            missile.exploded = rng.random() < 0.2
        state.cooldown_left = rng.randint(0, 30)
        state.player.vx, state.player.vy = rng.choice((-5.0, 0.0, 5.0)), rng.choice(
            (-5.0, 0.0, 5.0)
        )
        state.enemy.vx, state.enemy.vy = rng.uniform(-5, 5), rng.uniform(-5, 5)

        # The same situation expressed as live pygame entities, timed in ms.
        now = 600 * FRAME_MS
        player = PlayerPlay(W, H)
        player.position["x"], player.position["y"] = state.player.x, state.player.y
        player.last_missile_time = (
            now - (CONFIG.missile_cooldown_frames - state.cooldown_left) * FRAME_MS
        )
        player.missiles = []
        for m in state.missiles:
            game_missile = SmartMissile(
                m.x,
                m.y,
                speed=5.0,
                vx=m.vx,
                vy=m.vy,
                birth_time=now - m.age_frames * FRAME_MS,
                lifespan=m.life_frames * FRAME_MS,
            )
            game_missile.exploded = m.exploded
            player.missiles.append(game_missile)
        snapshot = snapshot_from_game(
            CONFIG,
            player,
            (state.enemy.x, state.enemy.y),
            (state.player.vx, state.player.vy),
            (state.enemy.vx, state.enemy.vy),
            now,
        )
        np.testing.assert_allclose(
            build_enemy_observation(snapshot), build_enemy_observation(state), atol=1e-6
        )
