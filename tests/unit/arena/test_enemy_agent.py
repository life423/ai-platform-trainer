"""The in-game PPO agent: shared encoder, movement cap, respawn behavior."""
import json
import shutil

import pytest

import ai_platform_trainer.arena.enemy_policies as enemy_policies
from ai_platform_trainer.ai.envs.enemy_env import EnemyArenaEnv
from ai_platform_trainer.arena.enemy_policies import SB3Enemy
from ai_platform_trainer.arena.observations import ENEMY_DECISION_FRAMES
from ai_platform_trainer.entities.enemy_agent import ArenaEnemyAgent, model_card_summary
from ai_platform_trainer.entities.player_play import PlayerPlay

W, H = 1470, 956


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    from stable_baselines3 import PPO

    path = tmp_path_factory.mktemp("agent") / "enemy_ppo.zip"
    PPO(
        "MlpPolicy", EnemyArenaEnv(), n_steps=64, batch_size=64, device="cpu", seed=0
    ).save(path)
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


def test_panel_summarizes_the_model_card(tmp_path):
    card = {
        "timesteps": 10_000_000,
        "decision_frames": 4,
        "evaluation": {"net_per_min": 1.234},
    }
    (tmp_path / "enemy_ppo.json").write_text(json.dumps(card))
    assert model_card_summary(str(tmp_path / "enemy_ppo.zip")) == [
        "Neural-network policy, no scripted rules",
        "Trained 10.0M steps with PPO",
        "Decides 15 times per second",
        "Benchmark: +1.23 net catches/min",
    ]


def test_models_run_at_their_trained_decision_rate(model_path, tmp_path):
    custom = tmp_path / "custom.zip"
    shutil.copy(model_path, custom)
    (tmp_path / "custom.json").write_text(json.dumps({"decision_frames": 3}))
    assert SB3Enemy.load(str(custom)).decision_frames == 3
    assert SB3Enemy.load(model_path).decision_frames == ENEMY_DECISION_FRAMES
