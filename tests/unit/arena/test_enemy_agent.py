"""The AI Agent in Play mode: shared encoder, movement cap, respawn, model card checks."""
import json
import os
import shutil

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

import ai_platform_trainer.arena.enemy_policies as enemy_policies
from ai_platform_trainer.ai.model_paths import ENEMY_MODEL, TRAINING_COMMANDS
from ai_platform_trainer.arena.enemy_policies import SB3Enemy
from ai_platform_trainer.arena.observations import ENEMY_DECISION_FRAMES, ENEMY_OBS_SIZE
from ai_platform_trainer.entities.enemy_agent import (
    ArenaEnemyAgent,
    enemy_agent_problem,
    model_card_summary,
)
from ai_platform_trainer.entities.player_play import PlayerPlay

W, H = 1470, 956
CARD_ALGORITHM = "behaviour cloning of the chase enemy, then PPO"
CARD_SUMMARY_HEAD = "Neural-network policy, trained by copying the chase enemy, then PPO"


class _Spec(gymnasium.Env):
    """Only the spaces matter here: the agent tests need a policy of the right shape."""

    def __init__(self):
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(ENEMY_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(ENEMY_OBS_SIZE, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(ENEMY_OBS_SIZE, dtype=np.float32), 0.0, True, False, {}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    from stable_baselines3 import PPO

    path = tmp_path_factory.mktemp("agent") / os.path.basename(ENEMY_MODEL)
    PPO("MlpPolicy", _Spec(), n_steps=64, batch_size=64, device="cpu", seed=0).save(path)
    return str(path)


def make_agent(model_path):
    player = PlayerPlay(W, H)
    player.position["x"], player.position["y"] = 200, 200
    agent = ArenaEnemyAgent(W, H, player_provider=lambda: player, model_path=model_path)
    agent.set_position(900.0, 600.0)
    return agent, player


def test_agent_moves_within_the_enemy_speed_cap(model_path):
    agent, player = make_agent(model_path)
    top = agent.config.enemy_speed
    for frame in range(50):
        x, y = agent.pos["x"], agent.pos["y"]
        agent.update_movement(200, 200, 5, 10_000 + frame * 16, player.missiles)
        assert abs(agent.pos["x"] - x) <= top + 1e-9
        assert abs(agent.pos["y"] - y) <= top + 1e-9


def test_agent_decides_through_the_shared_encoder(model_path, monkeypatch):
    seen = []
    real = enemy_policies.build_enemy_observation

    def spy(state):
        seen.append(state)
        return real(state)

    monkeypatch.setattr(enemy_policies, "build_enemy_observation", spy)
    agent, player = make_agent(model_path)
    agent.update_movement(200, 200, 5, 10_000, player.missiles)
    assert len(seen) == 1
    assert (seen[0].enemy.x, seen[0].player.x) == (900.0, 200.0)


def test_hidden_agent_waits_and_reports_stats(model_path):
    agent, player = make_agent(model_path)
    agent.hide()
    agent.update_movement(200, 200, 5, 10_000, player.missiles)
    assert (agent.pos["x"], agent.pos["y"]) == (900.0, 600.0)
    assert agent.panel_stats()["title"] == "AI Agent (PPO)"


def test_models_run_at_their_trained_decision_rate(model_path, tmp_path):
    custom = tmp_path / "custom.zip"
    shutil.copy(model_path, custom)
    (tmp_path / "custom.json").write_text(json.dumps({"decision_frames": 3}))
    assert SB3Enemy.load(str(custom)).decision_frames == 3
    assert SB3Enemy.load(model_path).decision_frames == ENEMY_DECISION_FRAMES


def test_panel_summarizes_the_model_card(tmp_path):
    card = {
        "algorithm": CARD_ALGORITHM,
        "timesteps": 1_000_448,
        "decision_frames": 4,
        "evaluation": {"net_per_min": 7.3, "rounds": 160},
    }
    (tmp_path / "enemy.json").write_text(json.dumps(card))
    assert model_card_summary(str(tmp_path / "enemy.zip")) == [
        CARD_SUMMARY_HEAD,
        "Trained 1.0M decisions",
        "Decides 15 times per second",
        "Benchmark: +7.30 net catches/min over 160 rounds",
    ]


def test_panel_has_no_benchmark_line_without_an_evaluation(tmp_path):
    card = {"algorithm": CARD_ALGORITHM, "timesteps": 2_500_000, "decision_frames": 4}
    (tmp_path / "enemy.json").write_text(json.dumps(card))
    assert model_card_summary(str(tmp_path / "enemy.zip")) == [
        CARD_SUMMARY_HEAD,
        "Trained 2.5M decisions",
        "Decides 15 times per second",
    ]


def _write_model(tmp_path, card=None):
    """A model file (only its presence is checked) and, optionally, its card."""
    model = tmp_path / "enemy.zip"
    model.write_bytes(b"")
    if card is not None:
        (tmp_path / "enemy.json").write_text(json.dumps(card))
    return str(model)


GOOD_CARD = {"algorithm": CARD_ALGORITHM, "eval_arena": {"width": 800, "height": 600}}


def test_agent_refuses_a_missing_model(tmp_path):
    problem = enemy_agent_problem(str(tmp_path / "enemy.zip"))
    assert "AI Agent is not trained yet" in problem
    assert TRAINING_COMMANDS["enemy"] in problem


def test_agent_refuses_a_model_without_a_card(tmp_path):
    problem = enemy_agent_problem(_write_model(tmp_path))
    assert "no model card" in problem
    assert TRAINING_COMMANDS["enemy"] in problem


def test_agent_refuses_a_card_without_an_algorithm(tmp_path):
    old_card = {"eval_arena": {"width": 800, "height": 600}}
    problem = enemy_agent_problem(_write_model(tmp_path, old_card))
    assert "older training run" in problem
    assert "python -m ai_platform_trainer train-enemy" in problem


def test_agent_refuses_a_policy_from_another_arena_size(tmp_path):
    card = {"algorithm": CARD_ALGORITHM, "eval_arena": {"width": 1470, "height": 956}}
    problem = enemy_agent_problem(_write_model(tmp_path, card))
    assert "1470x956" in problem
    assert "800x600" in problem


def test_agent_accepts_a_current_card(tmp_path):
    assert enemy_agent_problem(_write_model(tmp_path, GOOD_CARD)) is None


@pytest.mark.skipif(not os.path.exists(ENEMY_MODEL), reason="AI Agent model not on disk")
def test_deployed_agent_passes_the_checks():
    assert enemy_agent_problem() is None
