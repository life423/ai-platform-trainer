"""The Gymnasium training environment."""
import math

import numpy as np
import pytest

from ai_platform_trainer.ai.envs.enemy_env import (
    GAMMA,
    TIME_COST_PER_FRAME,
    EnemyArenaEnv,
)
from ai_platform_trainer.arena.observations import ENEMY_DECISION_FRAMES


def test_env_passes_the_stable_baselines3_checker():
    from stable_baselines3.common.env_checker import check_env

    check_env(EnemyArenaEnv(), warn=True)


def test_env_is_deterministic_for_a_seed():
    first, second = EnemyArenaEnv(), EnemyArenaEnv()
    obs_a, _ = first.reset(seed=123)
    obs_b, _ = second.reset(seed=123)
    np.testing.assert_array_equal(obs_a, obs_b)
    actions = np.random.default_rng(0).uniform(-1, 1, size=(600, 2)).astype(np.float32)
    for action in actions:
        step_a, step_b = first.step(action), second.step(action)
        np.testing.assert_array_equal(step_a[0], step_b[0])
        assert step_a[1:4] == step_b[1:4]
        if step_a[2] or step_a[3]:
            np.testing.assert_array_equal(first.reset()[0], second.reset()[0])


def quiet_env(seed):
    env = EnemyArenaEnv(randomize=False, bot_styles=("idle",), preload_prob=0.0)
    env.reset(seed=seed)
    state = env.episode.state
    state.missiles.clear()
    return env, state


def test_catching_the_player_pays_plus_one_and_respawns_the_player():
    env, state = quiet_env(0)
    state.enemy.x, state.enemy.y = state.player.x + 30.0, state.player.y
    env._potential = env._phi(state)
    enemy_before = (state.enemy.x, state.enemy.y)
    _, reward, terminated, truncated, info = env.step(
        np.zeros(2, dtype=np.float32)
    )
    assert info["catches"] == 1 and info["event"] == "caught"
    assert not terminated and not truncated
    assert (state.enemy.x, state.enemy.y) == enemy_before
    away = math.hypot(state.player.x - state.enemy.x, state.player.y - state.enemy.y)
    assert away > state.config.min_spawn_distance
    # The jump is not shaped, so only the discounted potential of the new position
    # counts; the round goes on, so all four frames of time are charged.
    phi_after = env._phi(state)
    assert reward == pytest.approx(
        1.0 + env.shaping * (GAMMA - 1.0) * phi_after - 4 * TIME_COST_PER_FRAME
    )


def test_missile_hit_costs_one_hides_the_enemy_and_the_round_goes_on():
    env, state = quiet_env(1)
    state.player.x, state.player.y = 100.0, 100.0
    state.enemy.x, state.enemy.y = 800.0, 600.0
    env.episode.sim.spawn_missile(810.0, 610.0, 0.0, 170)
    env._potential = env._phi(state)
    potential_before = env._potential
    _, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["hits"] == 1 and info["event"] == "hit"
    assert not terminated and not truncated
    assert not state.enemy_visible
    phi_after = env._phi(state)
    assert reward == pytest.approx(
        -1.0 + env.shaping * (GAMMA * phi_after - potential_before)
        - 4 * TIME_COST_PER_FRAME
    )


def test_a_hidden_enemy_reappears_away_from_the_player():
    env, state = quiet_env(2)
    state.player.x, state.player.y = 100.0, 100.0
    state.enemy.x, state.enemy.y = 800.0, 600.0
    state.enemy_visible = False
    state.enemy_respawn_frames = state.config.respawn_frames
    for _ in range(state.config.respawn_frames // ENEMY_DECISION_FRAMES + 2):
        env.step(np.zeros(2, dtype=np.float32))
        if state.enemy_visible:
            break
    assert state.enemy_visible
    gap = math.hypot(state.enemy.x - state.player.x, state.enemy.y - state.player.y)
    assert gap > state.config.min_spawn_distance
