"""The arena's SAC guidance must steer a missile exactly as the game does.

These tests use the real SAC model, so they are skipped if it is not on disk.
"""
import math
import os

import numpy as np
import pytest

from ai_platform_trainer.ai.model_paths import MISSILE_SAC_MODEL
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.observations import sac_missile_observation
from ai_platform_trainer.arena.state import MissileState
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities.smart_missile import SmartMissile

W, H = 800, 600
SAC_PATH = MISSILE_SAC_MODEL

needs_sac = pytest.mark.skipif(
    not os.path.exists(SAC_PATH), reason="SAC missile model not on disk"
)


@pytest.fixture(autouse=True)
def world_screen():
    """The game's world is the fixed 800x600 screen."""
    ScreenContext.initialize(W, H)


def test_arena_default_is_the_fixed_world():
    config = ArenaConfig()

    assert (config.width, config.height) == (W, H)


def test_game_and_arena_build_the_same_sac_input():
    missile = SmartMissile(
        300, 200, target_x=520.0, target_y=410.0, speed=5.0, vx=3.0, vy=4.0,
        rl_model=object(),
        rl_algorithm="sac",
    )
    target = {"x": 520.0, "y": 410.0}

    game_input = missile._create_sac_observation(target, 1.5, -2.0)
    arena_input = sac_missile_observation(
        300.0, 200.0, 3.0, 4.0, 520.0, 410.0, 1.5, -2.0, W, H
    )

    np.testing.assert_allclose(game_input, arena_input, atol=1e-7)
    assert arena_input.shape == (11,)


@needs_sac
def test_numpy_actor_matches_sb3_on_random_inputs():
    from ai_platform_trainer.arena.missile_policies import SacGuidance

    guidance = SacGuidance(SAC_PATH)
    rng = np.random.default_rng(3)
    for _ in range(200):
        obs = rng.uniform(-1.0, 1.0, size=(11,)).astype(np.float32)
        expected, _ = guidance.model.predict(obs, deterministic=True)
        np.testing.assert_allclose(guidance.actor.act(obs), expected, atol=1e-5)


@needs_sac
def test_arena_sac_guidance_steers_like_the_game():
    """One frame of guidance: the arena and the game must agree on the new velocity."""
    from ai_platform_trainer.arena.missile_policies import SacGuidance
    from stable_baselines3 import SAC

    config = ArenaConfig(width=W, height=H)
    guidance = SacGuidance(SAC_PATH)
    sac_model = SAC.load(SAC_PATH, device="cpu")
    target = {"x": 600.0, "y": 300.0}
    start = {"x": 200.0, "y": 350.0, "vx": 4.0, "vy": -2.0}

    game = SmartMissile(
        start["x"], start["y"], target_x=target["x"], target_y=target["y"],
        speed=config.missile_speed, vx=start["vx"], vy=start["vy"],
        rl_model=sac_model, rl_algorithm="sac",
    )
    arena = MissileState(
        start["x"], start["y"], start["vx"], start["vy"], life_frames=600,
        last_target_x=target["x"], last_target_y=target["y"],
    )

    game.update_with_ai(target)
    guidance(arena, target["x"], target["y"], config)

    assert arena.vx == pytest.approx(game.vx, abs=1e-5)
    assert arena.vy == pytest.approx(game.vy, abs=1e-5)
    assert math.hypot(arena.vx, arena.vy) == pytest.approx(config.missile_speed)


@needs_sac
def test_sac_guidance_is_the_default_for_training_and_benchmarks():
    from ai_platform_trainer.ai.envs.enemy_env import EnemyArenaEnv

    env = EnemyArenaEnv()

    assert env.guidance.name == "sac"
