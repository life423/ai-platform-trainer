"""The in-game PPO agent: shared encoder, movement cap, respawn behavior."""
import pytest

import ai_platform_trainer.arena.enemy_policies as enemy_policies
from ai_platform_trainer.ai.envs.enemy_env import EnemyArenaEnv
from ai_platform_trainer.entities.enemy_agent import ArenaEnemyAgent
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


def test_agent_moves_within_the_player_speed_cap(model_path):
    agent, player = make_agent(model_path)
    for frame in range(60):
        x, y = agent.pos["x"], agent.pos["y"]
        agent.update_movement(200, 200, 5, 10_000 + frame * 16, player.missiles)
        assert abs(agent.pos["x"] - x) <= 5.0 + 1e-9
        assert abs(agent.pos["y"] - y) <= 5.0 + 1e-9


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
    assert agent.get_learning_stats()["stage"] == "PPO agent"
