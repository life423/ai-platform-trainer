"""
Integration tests for core game mechanics.

These tests check the interaction between various components of the game,
ensuring that they work together correctly in a simulated game loop.
"""
from collections import defaultdict
from unittest.mock import Mock, patch

import pygame
import pytest
import torch

from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.gameplay.collisions import handle_missile_collisions


@pytest.fixture
def mock_pygame_setup():
    """Mock pygame setup to avoid actual window creation during tests."""
    # Mock pygame.init
    with patch("pygame.init"):
        # Mock display setup
        with patch("pygame.display.set_mode") as mock_set_mode:
            # Mock display.set_caption
            with patch("pygame.display.set_caption"):
                # Mock Surface
                mock_surface = Mock(spec=pygame.Surface)
                mock_set_mode.return_value = mock_surface
                yield mock_surface


@pytest.fixture
def mock_model():
    """Create a mock enemy AI model."""
    model = Mock()
    # Create a mock output tensor
    output_tensor = torch.tensor([[1.0, 0.0]])  # Move right
    model.return_value = output_tensor
    # Add load_state_dict method that just returns
    model.load_state_dict = Mock(return_value=None)
    model.eval = Mock(return_value=None)
    return model


@pytest.fixture
def player():
    """Create a player instance for testing."""
    return PlayerPlay(screen_width=800, screen_height=600)


@pytest.fixture
def enemy(mock_model):
    """Create an enemy instance for testing."""
    return EnemyPlay(screen_width=800, screen_height=600, model=mock_model)


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

    def test_missile_enemy_collision(self, player, enemy, mock_time):
        """Test collision detection between missiles and enemies."""
        # Position enemy and player for a clean test
        player.position = {"x": 100, "y": 100}
        enemy.pos = {"x": 500, "y": 100}
        enemy.visible = True

        # Fire a missile directly at the enemy. shoot_missile() checks a
        # cooldown against pygame.time.get_ticks(), so the mocked clock
        # (mock_time) must be active for the shot to actually fire.
        player.shoot_missile(enemy_pos=enemy.pos)

        # Should have one missile
        assert len(player.missiles) == 1

        # Mock the missile to be at the enemy position to guarantee collision
        missile = player.missiles[0]
        missile.pos = {"x": enemy.pos["x"], "y": enemy.pos["y"]}

        # Create a mock respawn callback
        respawn_callback = Mock()

        # Check for collision
        handle_missile_collisions(player, enemy, respawn_callback)

        # Missile should be removed
        assert len(player.missiles) == 0

        # Enemy should be hidden
        assert not enemy.visible

        # Respawn callback should have been called
        respawn_callback.assert_called_once()

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
