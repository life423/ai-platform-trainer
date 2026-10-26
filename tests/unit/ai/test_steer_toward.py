"""
Unit tests for steer_toward, the deterministic controller that labels training data.

It takes plain objects with pos, vx, vy, speed and last_action, so these tests
don't need pygame or a model.
"""

import math
from types import SimpleNamespace

import pytest

from ai_platform_trainer.ai.inference.missile_controller import steer_toward


def _missile(vx: float = 10.0, vy: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        pos={"x": 0.0, "y": 0.0}, vx=vx, vy=vy, speed=10.0, last_action=0.0
    )


def test_turn_is_capped_at_max_rate():
    missile = _missile()
    # The enemy is straight up, 90 degrees away from the missile's heading
    turn = steer_toward(missile, {"x": 0.0, "y": 100.0}, max_turn_rate=5.0)

    assert turn == pytest.approx(5.0)
    assert math.degrees(math.atan2(missile.vy, missile.vx)) == pytest.approx(5.0)


def test_small_error_is_turned_exactly():
    missile = _missile()
    angle = math.radians(2.0)
    enemy = {"x": 100.0 * math.cos(angle), "y": 100.0 * math.sin(angle)}

    turn = steer_toward(missile, enemy, max_turn_rate=5.0)

    assert turn == pytest.approx(2.0)


def test_records_the_label_on_the_missile():
    missile = _missile()
    turn = steer_toward(missile, {"x": 0.0, "y": 100.0}, max_turn_rate=5.0)

    assert missile.last_action == pytest.approx(turn)


def test_keeps_speed_constant():
    missile = _missile()
    steer_toward(missile, {"x": 0.0, "y": 100.0}, max_turn_rate=5.0)

    assert math.hypot(missile.vx, missile.vy) == pytest.approx(10.0)


def test_intercepts_a_stationary_enemy():
    missile = _missile()
    enemy = {"x": 0.0, "y": 100.0}

    for _ in range(40):
        steer_toward(missile, enemy, max_turn_rate=5.0)

    heading = math.atan2(missile.vy, missile.vx)
    assert heading == pytest.approx(math.pi / 2, abs=1e-9)
