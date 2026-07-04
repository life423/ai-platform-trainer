"""
Unit tests for the EnemyAIController class.

Tests the enemy AI controller behavior, including movement strategies,
stuck detection, and recovery behaviors.
"""
import math
import random
from unittest.mock import Mock, patch

import pytest
import torch

from ai_platform_trainer.entities.behaviors.enemy_ai_controller import EnemyAIController

# Use a fixed seed for deterministic tests
random.seed(42)
torch.manual_seed(42)


@pytest.fixture
def controller():
    """Return a fresh EnemyAIController instance for each test."""
    return EnemyAIController()


@pytest.fixture
def mock_enemy():
    """Create a mock enemy for testing."""
    enemy = Mock()
    enemy.pos = {"x": 100, "y": 100}
    enemy.visible = True
    # Mock (not a plain lambda) so tests can assert on how it was called
    enemy.wrap_position = Mock(side_effect=lambda x, y: (x, y))
    enemy.model = Mock()
    # Plain Mocks auto-vivify any attribute access, so without this,
    # `hasattr(enemy, 'game') and hasattr(enemy.game, 'player')` in the
    # controller would be True and pull in the missile-evasion code path.
    enemy.game = None

    # Setup the model to return a specific direction
    mock_action = torch.tensor([[1.0, 0.0]])  # Move right
    enemy.model.return_value = mock_action

    return enemy


class TestEnemyAIController:
    """Tests for the EnemyAIController class."""

    def test_initialization(self, controller):
        """Test that the controller initializes with expected default values."""
        assert controller.last_action_time is not None
        assert controller.action_interval == 0.01
        assert controller.smoothing_factor == 0.4
        assert controller.prev_dx == 0
        assert controller.prev_dy == 0
        assert controller.stuck_counter == 0
        assert len(controller.prev_positions) == 0
        assert controller.max_positions == 10

    def test_normalize_vector(self, controller):
        """Test that vector normalization works correctly."""
        # Test with normal vector
        dx, dy = controller._normalize_vector(3.0, 4.0)
        # Should be normalized to length 1
        assert math.isclose(dx, 0.6, abs_tol=1e-6)
        assert math.isclose(dy, 0.8, abs_tol=1e-6)
        assert math.isclose(dx**2 + dy**2, 1.0, abs_tol=1e-6)

        # Test with zero vector - should return random direction
        with patch.object(controller, "_get_random_direction") as mock_random:
            mock_random.return_value = (0.5, 0.5)
            dx, dy = controller._normalize_vector(0.0, 0.0)
            mock_random.assert_called_once()
            assert dx == 0.5
            assert dy == 0.5

    def test_get_random_direction(self, controller):
        """Test that random direction returns a unit vector."""
        for _ in range(10):  # Test multiple times to account for randomness
            dx, dy = controller._get_random_direction()
            # Vector should have length 1
            assert math.isclose(dx**2 + dy**2, 1.0, abs_tol=1e-6)

    def test_update_position_history(self, controller):
        """Test position history tracking."""
        # Add positions
        controller._update_position_history({"x": 10, "y": 20})
        assert len(controller.prev_positions) == 1
        assert controller.prev_positions[0] == {"x": 10, "y": 20}

        # Add more positions up to the max
        for i in range(1, controller.max_positions + 2):
            controller._update_position_history({"x": i, "y": i})

        # Should maintain max length
        assert len(controller.prev_positions) == controller.max_positions
        # The oldest position should have been removed
        assert controller.prev_positions[0] != {"x": 1, "y": 1}

    def test_is_enemy_stuck_not_enough_positions(self, controller):
        """Test stuck detection when not enough position history."""
        # Without enough position history, should not be stuck
        controller._update_position_history({"x": 10, "y": 10})
        assert not controller._is_enemy_stuck()

    def test_is_enemy_stuck_detected(self, controller):
        """Test that enemy stuck state is detected correctly."""
        # Add positions with very little movement
        for _ in range(controller.max_positions):
            # Random jitter within 5 pixels
            x = 100 + random.uniform(-5, 5)
            y = 100 + random.uniform(-5, 5)
            controller._update_position_history({"x": x, "y": y})

        # First check might not detect stuck due to counter
        for _ in range(5):  # Call multiple times to increase counter
            is_stuck = controller._is_enemy_stuck()
            if is_stuck:
                break

        # Should eventually detect as stuck
        assert is_stuck

    def test_is_enemy_not_stuck_with_movement(self, controller):
        """Test that enemy with normal movement is not detected as stuck."""
        # Add positions with significant movement
        for i in range(controller.max_positions):
            controller._update_position_history({"x": 100 + i * 10, "y": 100})

        assert not controller._is_enemy_stuck()

    def test_handle_stuck_enemy(self, controller):
        """Test the stuck handler provides appropriate escape behavior."""
        dx, dy = controller._handle_stuck_enemy(
            player_x=50, player_y=50, enemy_pos={"x": 100, "y": 100}
        )

        # Direction should point away from player
        assert dx > 0  # Player at x=50, enemy at x=100
        assert dy > 0  # Player at y=50, enemy at y=100
        # Should be a unit vector
        assert math.isclose(dx**2 + dy**2, 1.0, abs_tol=1e-6)

    def test_get_nn_action_normal(self, controller, mock_enemy):
        """Test neural network action retrieval."""
        action_dx, action_dy = controller._get_nn_action(
            mock_enemy, player_x=50, player_y=50
        )

        # With our mock setup, should return (1.0, 0.0)
        assert math.isclose(action_dx, 1.0, abs_tol=1e-6)
        assert math.isclose(action_dy, 0.0, abs_tol=1e-6)

        # Check that model received correct input
        mock_enemy.model.assert_called_once()
        # Get the first positional argument (tensor)
        args, _ = mock_enemy.model.call_args
        tensor_arg = args[0]

        # Verify tensor shape and content
        assert tensor_arg.shape == (1, 5)
        assert tensor_arg[0][0].item() == 50  # player_x
        assert tensor_arg[0][1].item() == 50  # player_y
        assert tensor_arg[0][2].item() == 100  # enemy_x
        assert tensor_arg[0][3].item() == 100  # enemy_y
        # The last value should be the distance
        distance = math.sqrt((50 - 100) ** 2 + (50 - 100) ** 2)
        assert math.isclose(tensor_arg[0][4].item(), distance, abs_tol=1e-6)

    def test_get_nn_action_error(self, controller, mock_enemy):
        """Test neural network error handling."""
        # Make the model raise an exception
        mock_enemy.model.side_effect = RuntimeError("Model error")

        # Should fall back to random direction
        with patch.object(controller, "_get_random_direction") as mock_random:
            mock_random.return_value = (0.5, 0.5)
            dx, dy = controller._get_nn_action(mock_enemy, player_x=50, player_y=50)
            mock_random.assert_called_once()
            assert dx == 0.5
            assert dy == 0.5

    def test_get_nn_action_zero_output(self, controller, mock_enemy):
        """Test neural network returning zero vector."""
        # Make the model return a zero vector
        mock_enemy.model.return_value = torch.tensor([[0.0, 0.0]])

        # Should fall back to random direction
        with patch.object(controller, "_get_random_direction") as mock_random:
            mock_random.return_value = (0.5, 0.5)
            dx, dy = controller._get_nn_action(mock_enemy, player_x=50, player_y=50)
            mock_random.assert_called_once()
            assert dx == 0.5
            assert dy == 0.5

    @patch("time.time")
    def test_update_enemy_movement_throttling(self, mock_time, controller, mock_enemy):
        """Test throttling of updates for performance."""
        # Setup time to trigger throttling
        mock_time.return_value = 100.0
        # Comfortably less than action_interval regardless of its exact value
        controller.last_action_time = 100.0 - (controller.action_interval / 2)

        controller.update_enemy_movement(
            mock_enemy, player_x=50, player_y=50, player_speed=5.0, current_time=1000
        )

        # Enemy position should not change due to throttling
        assert mock_enemy.pos == {"x": 100, "y": 100}

        # Now with enough time passed
        mock_time.return_value = 100.0 + controller.action_interval + 0.01

        # Mock other methods to isolate the test
        with patch.object(controller, "_get_nn_action") as mock_nn:
            mock_nn.return_value = (1.0, 0.0)  # Move right
            with patch.object(controller, "_normalize_vector") as mock_norm:
                mock_norm.return_value = (1.0, 0.0)  # Keep direction

                controller.update_enemy_movement(
                    mock_enemy,
                    player_x=50,
                    player_y=50,
                    player_speed=5.0,
                    current_time=1000,
                )

                # Position should change now
                assert mock_enemy.pos != {"x": 100, "y": 100}

    def test_update_enemy_movement_invisible(self, controller, mock_enemy):
        """Test that invisible enemies don't update."""
        mock_enemy.visible = False

        controller.update_enemy_movement(
            mock_enemy, player_x=50, player_y=50, player_speed=5.0, current_time=1000
        )

        # Position should not change
        assert mock_enemy.pos == {"x": 100, "y": 100}

    def test_update_enemy_movement_stuck(self, controller, mock_enemy):
        """Test stuck detection and handling during movement update."""
        # Setup to detect as stuck
        controller.stuck_counter = 4  # Above threshold
        with patch.object(controller, "_is_enemy_stuck") as mock_stuck:
            mock_stuck.return_value = True
            with patch.object(controller, "_handle_stuck_enemy") as mock_handle:
                mock_handle.return_value = (0.0, 1.0)  # Move up
                with patch.object(controller, "_normalize_vector") as mock_norm:
                    mock_norm.return_value = (0.0, 1.0)  # Keep direction
                    with patch("time.time") as mock_time:
                        # Ensure we're past throttle time
                        mock_time.return_value = controller.last_action_time + 1.0

                        controller.update_enemy_movement(
                            mock_enemy,
                            player_x=50,
                            player_y=50,
                            player_speed=5.0,
                            current_time=1000,
                        )

                        # Should have called the stuck handler
                        mock_handle.assert_called_once_with(50, 50, mock_enemy.pos)

                        # Position should change according to stuck handler (up)
                        assert mock_enemy.pos["y"] > 100

    def test_update_enemy_movement_normal(self, controller, mock_enemy):
        """Test normal enemy movement update."""
        with patch.object(controller, "_get_nn_action") as mock_nn:
            mock_nn.return_value = (1.0, 0.0)  # Move right
            with patch.object(controller, "_normalize_vector") as mock_norm:
                mock_norm.return_value = (1.0, 0.0)  # Keep direction
                with patch.object(controller, "_is_enemy_stuck") as mock_stuck:
                    mock_stuck.return_value = False
                    with patch("time.time") as mock_time:
                        # Ensure we're past throttle time
                        mock_time.return_value = controller.last_action_time + 1.0

                        controller.update_enemy_movement(
                            mock_enemy,
                            player_x=50,
                            player_y=50,
                            player_speed=5.0,
                            current_time=1000,
                        )

                        # Position should change according to NN action (right)
                        assert mock_enemy.pos["x"] > 100
                        assert mock_enemy.pos["y"] == 100

                        # Should have called wrap_position
                        mock_enemy.wrap_position.assert_called_once()
