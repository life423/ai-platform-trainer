"""The Gymnasium training environment."""
import numpy as np
import pytest

from ai_platform_trainer.ai.envs.enemy_env import TIME_COST_PER_FRAME, EnemyArenaEnv


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


def test_catching_the_player_pays_plus_one_and_ends_the_round():
    env, state = quiet_env(0)
    state.enemy.x, state.enemy.y = state.player.x + 30.0, state.player.y
    env._potential = env._phi(state)  # 30 px apart
    _, reward, terminated, truncated, info = env.step(
        np.array([-1.0, 0.0], dtype=np.float32)
    )
    assert info["event"] == "caught" and terminated and not truncated
    # +1 for the catch, plus the shaping payback -phi(s) = 30 / 1000
    assert reward == pytest.approx(
        1.0 + env.shaping * 30.0 / 1000.0 - TIME_COST_PER_FRAME
    )


def test_missile_hit_costs_one_and_ends_the_round():
    env, state = quiet_env(1)
    state.player.x, state.player.y = 100.0, 100.0
    state.enemy.x, state.enemy.y = 800.0, 600.0
    env.episode.sim.spawn_missile(810.0, 610.0, 0.0, 170)
    env._potential = env._phi(state)
    potential = env._potential
    _, reward, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["event"] == "hit" and terminated
    # potential-based shaping pays back -phi(s) on termination
    assert reward == pytest.approx(-1.0 - env.shaping * potential - TIME_COST_PER_FRAME)
