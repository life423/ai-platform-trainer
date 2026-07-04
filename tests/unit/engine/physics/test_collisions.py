"""
Unit tests for collision detection.

Tests collision detection and handling for missiles and enemies.
"""
from unittest.mock import Mock

import pygame
import pytest

from ai_platform_trainer.gameplay.collisions import handle_missile_collisions


@pytest.fixture
def player():
    """Create a mock player with missiles for testing collisions."""
    player = Mock()

    # Setup missiles list with one active missile
    # handle_missile_collisions reads missile.pos directly (not get_rect()),
    # so pos must be set for the mock to behave like a real Missile.
    missile1 = Mock()
    missile1.pos = {"x": 100, "y": 100}
    missile_rect = pygame.Rect(100, 100, 20, 20)
    missile1.get_rect.return_value = missile_rect

    player.missiles = [missile1]

    return player


@pytest.fixture
def enemy():
    """Create a mock enemy for testing collisions."""
    enemy = Mock()
    enemy.pos = {"x": 90, "y": 90}
    enemy.size = 30
    enemy.visible = True

    return enemy


@pytest.fixture
def respawn_callback():
    """Create a mock callback function for respawning."""
    return Mock()


class TestCollisions:
    """Tests for collision handling."""

    def test_handle_missile_collisions_hit(self, player, enemy, respawn_callback):
        """Test collision detection when a missile hits the enemy."""
        # Setup collision: missile at (100,100) with size 20x20,
        # enemy at (90,90) with size 30x30 - these overlap
        missile = player.missiles[0]

        # Call the function under test
        handle_missile_collisions(player, enemy, respawn_callback)

        # Verify the missile was removed
        assert missile not in player.missiles
        assert len(player.missiles) == 0

        # Verify the enemy was hidden
        enemy.hide.assert_called_once()

        # Verify respawn callback was called
        respawn_callback.assert_called_once()

    def test_handle_missile_collisions_no_hit(self, player, enemy, respawn_callback):
        """Test collision detection when a missile does not hit the enemy."""
        # Move the enemy away from the missile
        enemy.pos = {"x": 300, "y": 300}  # Far from missile

        # Call the function under test
        handle_missile_collisions(player, enemy, respawn_callback)

        # Verify the missile was not removed
        assert len(player.missiles) == 1

        # Verify the enemy was not hidden
        enemy.hide.assert_not_called()

        # Verify respawn callback was not called
        respawn_callback.assert_not_called()

    def test_handle_missile_collisions_invisible_enemy(
        self, player, enemy, respawn_callback
    ):
        """Test collision detection with an invisible enemy."""
        # Make the enemy invisible
        enemy.visible = False

        # Even though positions would cause a collision, the enemy is invisible
        handle_missile_collisions(player, enemy, respawn_callback)

        # Verify nothing happened since enemy is invisible
        assert len(player.missiles) == 1
        enemy.hide.assert_not_called()
        respawn_callback.assert_not_called()

    def test_handle_missile_collisions_multiple_missiles(
        self, player, enemy, respawn_callback
    ):
        """Test collision detection with multiple missiles."""
        # Add a second missile that won't collide
        missile2 = Mock()
        missile2.pos = {"x": 300, "y": 300}
        missile2_rect = pygame.Rect(300, 300, 20, 20)
        missile2.get_rect.return_value = missile2_rect
        player.missiles.append(missile2)

        # Call the function under test
        handle_missile_collisions(player, enemy, respawn_callback)

        # Verify only the first missile was removed
        assert len(player.missiles) == 1
        assert player.missiles[0] == missile2

        # Verify the enemy was hidden
        enemy.hide.assert_called_once()

        # Verify respawn callback was called once
        respawn_callback.assert_called_once()

    def test_handle_missile_collisions_multiple_hits(
        self, player, enemy, respawn_callback
    ):
        """Test handling multiple missiles hitting in a single frame."""
        # Add a second missile that also collides
        missile2 = Mock()
        missile2.pos = {"x": 95, "y": 95}
        missile2_rect = pygame.Rect(95, 95, 20, 20)  # Also collides
        missile2.get_rect.return_value = missile2_rect
        player.missiles.append(missile2)

        # Run the collision detection
        handle_missile_collisions(player, enemy, respawn_callback)

        # Both missiles should be removed
        assert len(player.missiles) == 0

        # Enemy should only be hidden once and respawn called once
        # (even though both missiles hit in the same frame)
        enemy.hide.assert_called_once()
        respawn_callback.assert_called_once()

    def test_handle_missile_collisions_no_missiles(self, enemy, respawn_callback):
        """Test collision detection when there are no missiles."""
        # Create player with empty missiles list
        player = Mock()
        player.missiles = []

        # Call the function under test
        handle_missile_collisions(player, enemy, respawn_callback)

        # Nothing should happen
        enemy.hide.assert_not_called()
        respawn_callback.assert_not_called()
        # Missiles list should still be empty
        assert player.missiles == []
