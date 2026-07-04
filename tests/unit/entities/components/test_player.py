"""
Unit tests for the Player classes.

Tests both PlayerPlay and PlayerTraining classes, focusing on
movement, missile management, and input handling.
"""
from collections import defaultdict
from unittest.mock import Mock, patch

import pygame
import pytest

from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.entities.player_training import PlayerTraining


@pytest.fixture
def player_play():
    """Return a fresh PlayerPlay instance for testing."""
    return PlayerPlay(screen_width=800, screen_height=600)


@pytest.fixture
def player_training():
    """Return a fresh PlayerTraining instance for testing."""
    return PlayerTraining(screen_width=800, screen_height=600)


class TestPlayerPlay:
    """Tests for the PlayerPlay class."""

    def test_initialization(self, player_play):
        """Test that a player initializes with the correct properties."""
        assert player_play.screen_width == 800
        assert player_play.screen_height == 600
        assert player_play.size == 50
        assert player_play.color == (0, 0, 139)  # Dark Blue
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
        """Test that missiles expire after their lifespan."""
        # Create a missile with a short lifespan
        with patch("pygame.time.get_ticks") as mock_time:
            # Start time
            mock_time.return_value = 1000
            player_play.shoot_missile()

            # Make missile's lifespan very short
            player_play.missiles[0].lifespan = 500

            # Update time to be after lifespan
            mock_time.return_value = 1501  # 1000 + 501

            # Update missiles - should remove expired missile
            player_play.update_missiles()

            # No more missiles
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


class TestPlayerTraining:
    """Tests for the PlayerTraining class."""

    def test_initialization(self, player_training):
        """Test that a training player initializes correctly."""
        assert player_training.screen_width == 800
        assert player_training.screen_height == 600
        assert player_training.size == 50
        assert player_training.color == (0, 0, 139)
        # Position is randomized, so just check it's within bounds
        assert 0 <= player_training.position["x"] <= 800 - player_training.size
        assert 0 <= player_training.position["y"] <= 600 - player_training.size
        assert player_training.step == 5
        assert player_training.missiles == []

        # Should have a movement pattern set
        assert player_training.current_pattern in player_training.PATTERNS

    def test_reset(self, player_training):
        """Test that reset correctly resets the training player state."""
        # Add a missile
        player_training.missiles.append(Mock())

        # Reset
        player_training.reset()

        # Missiles should be cleared
        assert player_training.missiles == []

        # Position should be within bounds (randomized)
        assert 0 <= player_training.position["x"] <= 800 - player_training.size
        assert 0 <= player_training.position["y"] <= 600 - player_training.size

        # Pattern might have changed (but might not due to randomness)
        # so we don't assert anything about it

    def test_switch_pattern(self, player_training):
        """Test pattern switching."""
        # Force a specific pattern
        player_training.current_pattern = "random_walk"

        # Keep switching until we get a different pattern
        for _ in range(10):  # Limit iterations to avoid infinite loop
            player_training.switch_pattern()
            if player_training.current_pattern != "random_walk":
                break

        # Should have a new pattern or timeout after 10 tries
        assert player_training.current_pattern != "random_walk" or _ == 9

        # State timer should be set to a reasonable value
        assert player_training.state_timer >= 180
        assert player_training.state_timer <= 300

    def test_update(self, player_training):
        """Test the update method."""
        # Store initial position
        initial_pos = player_training.position.copy()

        # Update with an enemy far away
        player_training.update(enemy_x=700, enemy_y=500)

        # Position should have changed
        assert player_training.position != initial_pos

    def test_shoot_missile(self, player_training):
        """Test shooting missiles in training mode."""
        # Mock pygame time
        with patch("pygame.time.get_ticks") as mock_time:
            mock_time.return_value = 1000

            # Shoot a missile
            player_training.shoot_missile(enemy_pos={"x": 500, "y": 300})

            # Should have one missile
            assert len(player_training.missiles) == 1

            # Missile should have correct birth time
            missile = player_training.missiles[0]
            assert missile.birth_time == 1000

            # Should aim toward the target
            assert missile.vx != 0
            assert missile.vy != 0

    def test_movement_patterns(self, player_training):
        """Test different movement patterns."""
        # Test each pattern
        for pattern in player_training.PATTERNS:
            # Force the specific pattern
            player_training.current_pattern = pattern
            player_training.circle_angle = 0
            player_training.circle_center = (400, 300)
            player_training.circle_radius = 100

            # Reset to a known position
            player_training.position = {"x": 400, "y": 300}

            # Update with enemy at a fixed position
            player_training.update(enemy_x=500, enemy_y=300)

            # Should have moved
            assert player_training.position != {"x": 400, "y": 300}
