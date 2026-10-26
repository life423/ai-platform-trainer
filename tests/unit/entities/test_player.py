"""
Unit tests for the Player classes.

Tests PlayerPlay, focusing on
movement, missile management, and input handling.
"""
from collections import defaultdict
from unittest.mock import Mock, patch

import pygame
import pytest

from ai_platform_trainer.entities.player_play import PlayerPlay


@pytest.fixture
def player_play():
    """Return a fresh PlayerPlay instance for testing."""
    return PlayerPlay(screen_width=800, screen_height=600)


class TestPlayerPlay:
    """Tests for the PlayerPlay class."""

    def test_initialization(self, player_play):
        """Test that a player initializes with the correct properties."""
        assert player_play.screen_width == 800
        assert player_play.screen_height == 600
        assert player_play.size == 50
        assert player_play.position == {"x": 800 // 4, "y": 600 // 2}
        assert player_play.step == 5
        assert player_play.missiles == []

    def test_reset(self, player_play):
        """Test that reset correctly resets the player state."""
        # Change position and add a missile
        player_play.position = {"x": 100, "y": 100}
        player_play.missiles.append(Mock())

        # Reset
        player_play.reset()

        # Should be back to initial state
        assert player_play.position == {"x": 800 // 4, "y": 600 // 2}
        assert player_play.missiles == []

    def test_shoot_missile(self, player_play):
        """Test missile creation when shooting."""
        # Mock pygame time
        with patch("pygame.time.get_ticks") as mock_time:
            mock_time.return_value = 1000

            # Shoot a missile with no target
            player_play.shoot_missile()

            # Should have one missile
            assert len(player_play.missiles) == 1

            # Missile should have correct properties
            missile = player_play.missiles[0]
            assert missile.birth_time == 1000
            assert missile.pos["x"] == player_play.position["x"] + player_play.size // 2
            assert missile.pos["y"] == player_play.position["y"] + player_play.size // 2
            assert missile.vx > 0  # Default direction is right
            assert missile.vy == 0

            # Try to shoot again immediately
            player_play.shoot_missile()

            # Still only one missile (can't fire while one is active)
            assert len(player_play.missiles) == 1

    def test_missiles_are_requested_from_the_sac_model(self, player_play):
        """Play has one missile AI, so a shot asks for it without choosing a model."""
        with patch("pygame.time.get_ticks", return_value=1000), patch(
            "ai_platform_trainer.entities.player_play.create_smart_missile"
        ) as make_missile:
            player_play.shoot_missile()

        assert "model_choice" not in make_missile.call_args.kwargs
        assert not hasattr(player_play, "model_choice")

    def test_shoot_missile_with_target(self, player_play):
        """Test missile creation when shooting at a target."""
        # Mock pygame time
        with patch("pygame.time.get_ticks") as mock_time:
            mock_time.return_value = 1000

            # Target position
            target_pos = {"x": 600, "y": 400}

            # Shoot a missile at the target
            player_play.shoot_missile(enemy_pos=target_pos)

            # Should have one missile directed at the target
            assert len(player_play.missiles) == 1

            missile = player_play.missiles[0]

            # Direction should be toward target (positive x and y)
            assert missile.vx > 0
            assert missile.vy > 0

    def test_update_missiles(self, player_play):
        """Test updating missile positions."""
        # Create a missile
        with patch("pygame.time.get_ticks") as mock_time:
            mock_time.return_value = 1000
            player_play.shoot_missile()

        # Get initial position
        missile = player_play.missiles[0]
        initial_x = missile.pos["x"]
        initial_y = missile.pos["y"]

        # Update missiles
        player_play.update_missiles()

        # Missile should have moved
        assert missile.pos["x"] > initial_x
        assert missile.pos["y"] == initial_y  # No y movement by default

    def test_update_missiles_expiration(self, player_play):
        """Test that missiles explode after their lifespan, then disappear."""
        # Create a missile with a short lifespan
        with patch("pygame.time.get_ticks") as mock_time:
            # Start time
            mock_time.return_value = 1000
            player_play.shoot_missile()

            # Make missile's lifespan very short
            missile = player_play.missiles[0]
            missile.lifespan = 500

            # Update time to be after lifespan
            mock_time.return_value = 1501  # 1000 + 501

            # First update after expiring: the missile explodes in place
            # rather than vanishing immediately.
            player_play.update_missiles()
            assert len(player_play.missiles) == 1
            assert missile.exploded is True

            # Once the explosion animation has played out, it's removed.
            mock_time.return_value += missile.EXPLOSION_DURATION_MS + 1
            player_play.update_missiles()
            assert len(player_play.missiles) == 0

    def test_handle_input(self, player_play):
        """Test that keyboard input changes player position."""
        # Arrow keys (e.g. K_LEFT = 1073741904) use SDL scancode-derived
        # constants far outside a small range, so a defaultdict is needed
        # rather than a fixed-size dict comprehension.
        keys = defaultdict(bool)

        # Mock pygame.key.get_pressed
        with patch("pygame.key.get_pressed", return_value=keys):
            # Initial position
            initial_pos = player_play.position.copy()

            # No keys pressed - should not move
            player_play.handle_input()
            assert player_play.position == initial_pos

            # Press right arrow
            keys[pygame.K_RIGHT] = True
            player_play.handle_input()
            assert player_play.position["x"] > initial_pos["x"]
            assert player_play.position["y"] == initial_pos["y"]

            # Reset and press left arrow
            player_play.position = initial_pos.copy()
            keys[pygame.K_RIGHT] = False
            keys[pygame.K_LEFT] = True
            player_play.handle_input()
            assert player_play.position["x"] < initial_pos["x"]
            assert player_play.position["y"] == initial_pos["y"]

            # Test WASD keys (e.g., W for up)
            player_play.position = initial_pos.copy()
            keys[pygame.K_LEFT] = False
            keys[pygame.K_w] = True
            player_play.handle_input()
            assert player_play.position["x"] == initial_pos["x"]
            assert player_play.position["y"] < initial_pos["y"]

    def test_wrap_around(self, player_play):
        """Test screen wrapping when player goes off-screen."""
        # handle_input() reads real keyboard state via pygame.key.get_pressed(),
        # which requires a display to be initialized. Mock it out since this
        # test only cares about the wrap-around logic, not key handling.
        with patch("pygame.key.get_pressed", return_value=defaultdict(bool)):
            # Move player off right edge
            player_play.position["x"] = player_play.screen_width + 1
            player_play.handle_input()  # Should wrap
            assert player_play.position["x"] == -player_play.size

            # Move player off left edge
            player_play.position["x"] = -player_play.size - 1
            player_play.handle_input()  # Should wrap
            assert player_play.position["x"] == player_play.screen_width

            # Move player off bottom edge
            player_play.position["y"] = player_play.screen_height + 1
            player_play.handle_input()  # Should wrap
            assert player_play.position["y"] == -player_play.size

            # Move player off top edge
            player_play.position["y"] = -player_play.size - 1
            player_play.handle_input()  # Should wrap
        assert player_play.position["y"] == player_play.screen_height
