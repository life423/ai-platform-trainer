"""
Unit tests for the supervised model's input vector.

Play and training both call missile_features(), so these tests pin the order
and scaling that the trained model depends on.
"""

import math

import pytest

from ai_platform_trainer.ai.missile_features import FEATURE_COUNT, missile_features
from ai_platform_trainer.core.screen_context import ScreenContext


@pytest.fixture
def screen():
    """A known 1280x720 screen, so normalized values are easy to check."""
    return ScreenContext.initialize(1280, 720)


def test_returns_nine_features_in_model_order(screen):
    features = missile_features(
        {"x": 640.0, "y": 360.0},
        {"x": 1280.0, "y": 720.0},
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 10.0},
    )

    assert len(features) == FEATURE_COUNT
    assert features[0] == pytest.approx(0.5)  # player_x
    assert features[1] == pytest.approx(0.5)  # player_y
    assert features[2] == pytest.approx(1.0)  # target_x
    assert features[3] == pytest.approx(1.0)  # target_y
    assert features[4] == pytest.approx(0.0)  # missile_x
    assert features[5] == pytest.approx(0.0)  # missile_y


def test_heading_is_the_direction_before_the_turn(screen):
    features = missile_features(
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 0.0},
        {"x": 0.0, "y": 10.0},
    )

    assert features[6] == pytest.approx(math.pi / 2)


def test_distance_is_normalized_to_the_screen(screen):
    features = missile_features(
        {"x": 0.0, "y": 0.0},
        {"x": 1280.0, "y": 720.0},
        {"x": 0.0, "y": 0.0},
        {"x": 1.0, "y": 0.0},
    )

    assert features[7] == pytest.approx(math.sqrt(2))


def test_last_feature_is_always_zero(screen):
    """The collision flag used to sit here. The game can't know it, so it is zero."""
    features = missile_features(
        {"x": 100.0, "y": 100.0},
        {"x": 300.0, "y": 300.0},
        {"x": 120.0, "y": 120.0},
        {"x": 5.0, "y": 5.0},
    )

    assert features[8] == 0.0
