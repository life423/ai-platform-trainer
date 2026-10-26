"""Pieces of the AI Agent trainer: demonstrations, behaviour cloning and the card.

train() itself runs for many minutes, so it is not run here.
"""
import math
import os

import numpy as np
import pytest

from ai_platform_trainer.ai.envs.chaser_env import HIT_REWARD, THREAT_WEIGHT, ChaserEnv
from ai_platform_trainer.ai.model_paths import (
    ENEMY_MODEL,
    MISSILE_SAC_MODEL,
    load_model_card,
    model_card_path,
)
from ai_platform_trainer.ai.training import train_enemy
from ai_platform_trainer.arena.observations import ENEMY_OBS_SIZE

needs_sac = pytest.mark.skipif(
    not os.path.exists(MISSILE_SAC_MODEL), reason="SAC missile model not on disk"
)


@needs_sac
def test_demonstrations_pair_observations_with_moves():
    observations, actions = train_enemy.collect_demonstrations(ChaserEnv(), rounds=2)

    assert observations.shape[0] == actions.shape[0] > 0
    assert observations.shape[1:] == (ENEMY_OBS_SIZE,)
    assert actions.shape[1:] == (2,)
    assert np.all(np.abs(actions) <= 1.0)


@needs_sac
def test_behaviour_cloning_runs_and_reports_a_finite_loss(monkeypatch):
    from stable_baselines3 import PPO

    monkeypatch.setattr(train_enemy, "BC_EPOCHS", 1)
    env = ChaserEnv()
    observations, actions = train_enemy.collect_demonstrations(env, rounds=2)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, device="cpu", seed=0)

    loss = train_enemy.clone_behaviour(model, observations, actions)

    assert math.isfinite(loss)
    predicted, _ = model.predict(observations[:5], deterministic=True)
    assert predicted.shape == (5, 2)


def test_write_card_round_trips_through_the_loader(tmp_path):
    card = {
        "name": "AI Agent",
        "algorithm": "behaviour cloning of the chase enemy, then PPO",
        "decision_frames": 4,
        "timesteps": 1000,
        "eval_arena": {"width": 800, "height": 600},
    }
    model_path = str(tmp_path / "enemy.zip")

    train_enemy.write_card(model_card_path(model_path), card)

    assert os.path.exists(os.path.join(tmp_path, "enemy.json"))
    assert load_model_card(model_path) == card


def test_cli_defaults_match_the_deployed_model():
    from ai_platform_trainer.cli import build_parser

    args = build_parser().parse_args(["train-enemy"])

    assert (args.rounds, args.timesteps, args.out) == (300, 1_000_000, ENEMY_MODEL)
    assert (args.hit_reward, args.threat_weight) == (HIT_REWARD, THREAT_WEIGHT)
