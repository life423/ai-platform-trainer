"""
Play must fail loudly when a trained model is missing or broken.

These tests pin that there is no scripted or basic-homing fallback left in
the game's missiles or enemy.
"""
import pytest

from ai_platform_trainer.ai.missile_ai_loader import MissileAIManager
from ai_platform_trainer.ai.model_paths import MissingModelError
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.entities import enemy_agent
from ai_platform_trainer.entities.smart_missile import SmartMissile


def _missile(**overrides) -> SmartMissile:
    kwargs = dict(
        x=100,
        y=100,
        target_x=300.0,
        target_y=300.0,
        speed=5.0,
        vx=5.0,
        vy=0.0,
        birth_time=0,
        lifespan=5000,
    )
    kwargs.update(overrides)
    return SmartMissile(**kwargs)


@pytest.fixture
def no_disk_models(monkeypatch):
    """A manager that does not read model files, so tests control what is loaded."""
    monkeypatch.setattr(MissileAIManager, "_load_models", lambda self: None)
    return MissileAIManager()


def test_missile_without_a_model_is_an_error():
    with pytest.raises(MissingModelError):
        _missile()


def test_missile_has_no_basic_homing_to_fall_back_to():
    assert not hasattr(SmartMissile, "_update_with_basic_homing")


def test_missile_rejects_an_unknown_algorithm():
    with pytest.raises(ValueError):
        _missile(rl_model=object(), rl_algorithm="basic")


def test_missile_model_errors_are_not_swallowed():
    class Broken:
        def predict(self, observation, deterministic=True):
            raise ValueError("boom")

    ScreenContext.initialize(1280, 720)
    missile = _missile(rl_model=Broken(), rl_algorithm="sac")

    with pytest.raises(ValueError, match="boom"):
        missile.update_with_ai({"x": 0.0, "y": 0.0}, {"x": 300.0, "y": 300.0})


def test_untrained_missile_model_is_refused(no_disk_models):
    with pytest.raises(MissingModelError, match="SAC is not trained yet"):
        no_disk_models.create_smart_missile(0, 0, model_choice="sac")


def test_unknown_missile_model_is_an_error(no_disk_models):
    with pytest.raises(ValueError):
        no_disk_models.create_smart_missile(0, 0, model_choice="basic")


def test_model_that_fails_to_load_is_reported(no_disk_models, monkeypatch):
    monkeypatch.setattr(
        no_disk_models, "_find", lambda filename: "models/missile_sac_model_final.zip"
    )

    def corrupt_loader(path):
        raise ValueError("corrupt zip")

    loaded = no_disk_models._load_rl_model(
        corrupt_loader, "missile_sac_model_final.zip", "sac"
    )

    assert loaded is None
    assert "corrupt zip" in no_disk_models.problem_with("sac")
    with pytest.raises(MissingModelError, match="corrupt zip"):
        no_disk_models.create_smart_missile(0, 0, model_choice="sac")


def test_agent_enemy_is_required(monkeypatch):
    monkeypatch.setattr(enemy_agent, "is_enemy_agent_available", lambda *a, **k: False)

    with pytest.raises(MissingModelError, match="AI Agent is not trained yet"):
        enemy_agent.create_trained_enemy(800, 600, player_provider=lambda: None)
