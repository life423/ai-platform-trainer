"""
Integration tests for core game mechanics.

These tests check the interaction between various components of the game,
ensuring that they work together correctly in a simulated game loop.
"""
from collections import defaultdict
from unittest.mock import patch

import pygame
import pytest

from ai_platform_trainer.entities.player_play import PlayerPlay


@pytest.fixture
def player():
    """Create a player instance for testing."""
    return PlayerPlay(screen_width=800, screen_height=600)


@pytest.fixture
def mock_time():
    """Mock pygame time.get_ticks to control game timing."""
    with patch("pygame.time.get_ticks") as mock:
        mock.return_value = 1000  # Start at 1000ms
        yield mock


@pytest.fixture
def mock_keys():
    """Mock pygame.key.get_pressed to simulate keyboard input."""
    with patch("pygame.key.get_pressed") as mock:
        # Arrow keys (e.g. K_LEFT = 1073741904) use SDL scancode-derived
        # constants far outside a small range, so a defaultdict is needed
        # rather than a fixed-size dict comprehension.
        keys = defaultdict(bool)  # Any unset key reads as not pressed
        mock.return_value = keys
        yield keys


class TestGameMechanics:
    """Integration tests for game mechanics."""

    def test_player_missile_firing(self, player, mock_time):
        """Test that the player can fire missiles correctly."""
        # No missiles at start
        assert len(player.missiles) == 0

        # Fire a missile
        player.shoot_missile(enemy_pos={"x": 500, "y": 300})

        # Should have one missile now
        assert len(player.missiles) == 1

        # Missile should have the correct properties
        missile = player.missiles[0]
        assert missile.birth_time == 1000  # The mocked time
        assert (
            missile.pos["x"] > player.position["x"]
        )  # Should start at player position
        assert missile.pos["y"] > player.position["y"]

        # Can't fire again immediately (only one active missile)
        player.shoot_missile(enemy_pos={"x": 500, "y": 300})
        assert len(player.missiles) == 1  # Still only one missile

        # Update missile position
        player.update_missiles()

        # Missile should have moved
        assert missile.pos != {"x": player.position["x"], "y": player.position["y"]}

    def test_player_input_handling(self, player, mock_keys):
        """Test player input handling."""
        # Set initial position
        initial_pos = {"x": 200, "y": 200}
        player.position = initial_pos.copy()

        # Simulate right key press
        mock_keys[pygame.K_RIGHT] = True

        # Handle input
        player.handle_input()

        # Player should have moved right
        assert player.position["x"] > initial_pos["x"]
        assert player.position["y"] == initial_pos["y"]  # Y position shouldn't change

        # Reset and simulate left key press
        player.position = initial_pos.copy()
        mock_keys[pygame.K_RIGHT] = False
        mock_keys[pygame.K_LEFT] = True

        # Handle input
        player.handle_input()

        # Player should have moved left
        assert player.position["x"] < initial_pos["x"]
        assert player.position["y"] == initial_pos["y"]  # Y position shouldn't change
