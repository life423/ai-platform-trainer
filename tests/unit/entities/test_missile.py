"""
Unit tests for the Missile class.

Tests the initialization, movement, and collision detection of missiles.
"""
from unittest.mock import Mock, patch

import pygame
import pytest

from ai_platform_trainer.entities.missile import Missile


@pytest.fixture
def missile():
    """Return a standard missile instance for testing."""
    return Missile(
        x=100, y=200, speed=5.0, vx=5.0, vy=0.0, birth_time=1000, lifespan=2000
    )


class TestMissile:
    """Tests for the Missile class."""

    def test_initialization(self, missile):
        """Test that a missile initializes with the correct properties."""
        assert missile.pos == {"x": 100, "y": 200}
        assert missile.speed == 5.0
        assert missile.vx == 5.0
        assert missile.vy == 0.0
        assert missile.birth_time == 1000
        assert missile.lifespan == 2000
        assert missile.size == 10
        assert missile.color == (255, 255, 0)  # Yellow

    def test_update_position(self, missile):
        """Test that update correctly changes the missile's position."""
        initial_x = missile.pos["x"]
        initial_y = missile.pos["y"]

        missile.update()

        # Position should change according to velocity
        assert missile.pos["x"] == initial_x + missile.vx
        assert missile.pos["y"] == initial_y + missile.vy

        # Change velocity and test again
        missile.vx = 3.0
        missile.vy = 4.0
        missile.update()

        assert missile.pos["x"] == initial_x + 5.0 + 3.0
        assert missile.pos["y"] == initial_y + 0.0 + 4.0

    def test_get_rect(self, missile):
        """Test that get_rect returns a correctly sized and positioned rectangle."""
        rect = missile.get_rect()

        # Rect should be centered on the missile position with 2x size width/height
        assert rect.centerx == missile.pos["x"]
        assert rect.centery == missile.pos["y"]
        assert rect.width == missile.size * 2
        assert rect.height == missile.size * 2

        # Check that the rect correctly encompasses the missile
        assert rect.left == missile.pos["x"] - missile.size
        assert rect.top == missile.pos["y"] - missile.size
        assert rect.right == missile.pos["x"] + missile.size
        assert rect.bottom == missile.pos["y"] + missile.size

    def test_draw(self, missile):
        """Test that draw calls the correct Pygame drawing function."""
        # Missile is rendered as a triangle pointing in its direction of travel
        with patch("pygame.draw.polygon") as mock_draw:
            # Create a mock surface
            mock_surface = Mock(spec=pygame.Surface)

            # Call the draw method
            missile.draw(mock_surface)

            # Verify pygame.draw.polygon was called with the missile's color
            # and a 3-point triangle
            mock_draw.assert_called_once()
            args, _ = mock_draw.call_args
            assert args[0] is mock_surface
            assert args[1] == missile.color
            assert len(args[2]) == 3

    def test_diagonal_movement(self):
        """Test that diagonal movement works correctly."""
        # Create a missile with diagonal movement
        diagonal_missile = Missile(x=100, y=100, speed=5.0, vx=3.0, vy=4.0)

        diagonal_missile.update()

        # Position should update according to both components
        assert diagonal_missile.pos["x"] == 103.0
        assert diagonal_missile.pos["y"] == 104.0

        # Speed remains constant
        assert diagonal_missile.speed == 5.0

        # Velocity components shouldn't change from the update
        assert diagonal_missile.vx == 3.0
        assert diagonal_missile.vy == 4.0

    def test_lifespan_settings(self):
        """Test different lifespan settings."""
        # Create missiles with different lifespans
        short_missile = Missile(
            x=100, y=100, birth_time=1000, lifespan=500  # Short lifespan
        )

        long_missile = Missile(
            x=100, y=100, birth_time=1000, lifespan=5000  # Long lifespan
        )

        assert short_missile.birth_time == 1000
        assert short_missile.lifespan == 500

        assert long_missile.birth_time == 1000
        assert long_missile.lifespan == 5000
