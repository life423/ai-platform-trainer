"""ChaserEnv: the AI Agent's training environment on the canonical arena rules.

The environment tests use the real SAC missile model, so they are skipped if it is
not on disk. The shaping tests build arena states directly and need no model.
"""
import os

import numpy as np
import pytest

from ai_platform_trainer.ai.envs.chaser_env import ChaserEnv, missile_threat, shaping_potential
from ai_platform_trainer.ai.model_paths import MISSILE_SAC_MODEL
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.observations import ENEMY_OBS_SIZE
from ai_platform_trainer.arena.state import ArenaState, Body, MissileState

needs_sac = pytest.mark.skipif(
    not os.path.exists(MISSILE_SAC_MODEL), reason="SAC missile model not on disk"
)


def _state(player, enemy, missiles=(), visible=True):
    return ArenaState(
        config=ArenaConfig(),
        player=Body(*player),
        enemy=Body(*enemy),
        missiles=list(missiles),
        enemy_visible=visible,
    )


@needs_sac
def test_reset_returns_an_enemy_observation():
    env = ChaserEnv()

    obs, info = env.reset(seed=1)

    assert obs.shape == (ENEMY_OBS_SIZE,)
    assert obs.dtype == np.float32
    assert "bot" in info


@needs_sac
def test_step_returns_reward_flags_and_event_counts():
    env = ChaserEnv()
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))

    assert obs.shape == (ENEMY_OBS_SIZE,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert terminated is False
    assert isinstance(truncated, bool)
    assert isinstance(info["catches"], int) and isinstance(info["hits"], int)


def test_shaping_is_zero_while_the_enemy_is_hidden():
    state = _state(player=(100.0, 100.0), enemy=(400.0, 300.0), visible=False)

    assert shaping_potential(state) == 0.0


def test_closer_to_the_player_means_a_higher_potential():
    near = _state(player=(420.0, 300.0), enemy=(400.0, 300.0))
    far = _state(player=(700.0, 300.0), enemy=(400.0, 300.0))

    assert shaping_potential(near) > shaping_potential(far)


def test_missile_threat_is_zero_without_missiles():
    assert missile_threat(_state(player=(100.0, 100.0), enemy=(400.0, 300.0))) == 0.0


def test_a_missile_aimed_at_the_enemy_is_a_threat():
    # 100 px to the left of the enemy, flying straight at it at full missile speed.
    speed = ArenaConfig().missile_speed
    missile = MissileState(x=300.0, y=300.0, vx=speed, vy=0.0, life_frames=600)
    state = _state(player=(100.0, 100.0), enemy=(400.0, 300.0), missiles=[missile])

    threat = missile_threat(state)

    assert threat > 0.0
    assert threat == pytest.approx(1.0 - 100.0 / 250.0)
