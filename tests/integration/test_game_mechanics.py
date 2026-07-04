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

from ai_platform_trainer.entities.behaviors.enemy_ai_controller import EnemyAIController
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

    def test_enemy_movement(self, enemy, mock_time):
        """Test that the enemy can move correctly."""
        # Set initial position
        initial_pos = {"x": 200, "y": 200}
        enemy.pos = initial_pos.copy()
        enemy.visible = True

        # Create controller directly for testing
        controller = EnemyAIController()

        # Update enemy movement
        with patch("time.time") as mock_time_func:
            mock_time_func.return_value = 1000  # Ensure we're past throttle time
            # last_action_time was seeded from the real clock at construction;
            # reset it relative to the mocked clock so the throttle check
            # in update_enemy_movement doesn't treat this as "too soon".
            controller.last_action_time = 1000 - controller.action_interval - 0.01

            controller.update_enemy_movement(
                enemy=enemy,
                player_x=300,  # Player to the right of enemy
                player_y=200,
                player_speed=5.0,
                current_time=1000,
            )

        # Enemy should have moved
        assert enemy.pos != initial_pos

        # With our mock model (1.0, 0.0), enemy should have moved right
        assert enemy.pos["x"] > initial_pos["x"]
        assert enemy.pos["y"] == initial_pos["y"]  # Y position shouldn't change

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

    def test_enemy_chasing_player(self, enemy, mock_time):
        """Test that the enemy follows the player."""
        # Position enemy and player
        enemy.pos = {"x": 200, "y": 200}
        enemy.visible = True
        player_pos = {"x": 300, "y": 300}  # Player is down and right

        # Create controller directly for testing
        controller = EnemyAIController()

        # Update enemy movement multiple times, overriding the model
        # to test the actual movement logic
        with patch("time.time") as mock_time_func:
            with patch.object(controller, "_get_nn_action") as mock_nn:
                # Setup mock to return direction toward player
                mock_nn.return_value = (0.7071, 0.7071)  # 45 degree angle (normalized)
                mock_time_func.return_value = 1000  # Ensure we're past throttle time
                # Reset last_action_time relative to the mocked clock (it was
                # seeded from the real clock at construction, before the patch
                # was active) so the throttle check doesn't block every update.
                controller.last_action_time = 1000 - controller.action_interval - 0.01

                # Update multiple times
                for _ in range(5):
                    mock_time_func.return_value += 100  # Advance time
                    controller.update_enemy_movement(
                        enemy=enemy,
                        player_x=player_pos["x"],
                        player_y=player_pos["y"],
                        player_speed=5.0,
                        current_time=mock_time_func.return_value,
                    )

        # Enemy should have moved toward player
        assert enemy.pos["x"] > 200
        assert enemy.pos["y"] > 200
