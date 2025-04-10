"""
Enemy manager for the AI Platform Trainer.

This module provides a manager class for handling multiple enemies,
including spawning, updating, and collision detection.
"""
import random
import logging
import math
import os
import pygame
from typing import List, Dict, Optional, Tuple

from ai_platform_trainer.entities.enemy_play import EnemyPlay
from ai_platform_trainer.ai_model.model_definition.enemy_movement_model import EnemyMovementModel
from ai_platform_trainer.gameplay.config import config


class EnemyManager:
    """
    Manages multiple enemy entities in the game.
    
    This class is responsible for creating, updating, and tracking
    multiple enemy ships, as well as handling collisions and respawning.
    """
    
    def __init__(self, screen_width: int, screen_height: int, model: EnemyMovementModel):
        """
        Initialize the enemy manager.
        
        Args:
            screen_width: Width of the game screen
            screen_height: Height of the game screen
            model: Neural network model for enemy behavior
        """
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model = model
        
        # Create a list to store multiple enemies
        self.enemies: List[EnemyPlay] = []
        
        # Track respawn timers for each enemy
        self.respawn_timers: Dict[int, int] = {}
        
        # Flag to track if we're currently respawning enemies
        self.is_respawning = False
        
        # Delay before respawning in milliseconds
        self.respawn_delay = 1000
        
        # Create a single enemy to start (for backward compatibility)
        self.primary_enemy = self._create_enemy()
        self.enemies.append(self.primary_enemy)
        
    def _create_enemy(self) -> EnemyPlay:
        """
        Create a new enemy with the model.
        
        Returns:
            A new EnemyPlay instance
        """
        enemy = EnemyPlay(
            self.screen_width,
            self.screen_height,
            self.model
        )
        
        # Try to load RL model if available
        rl_model_path = "models/enemy_rl/final_model.zip"
        if os.path.exists(rl_model_path):
            try:
                success = enemy.load_rl_model(rl_model_path)
                if success:
                    logging.info("Using RL model for enemy behavior")
                else:
                    logging.warning("RL model exists but couldn't be loaded.")
                    logging.warning("Falling back to neural network.")
            except Exception as e:
                logging.error(f"Error loading RL model: {e}.")
                logging.error("Using neural network instead.")
        
        return enemy
    
    def spawn_enemies(self, count: int, player_pos: Dict[str, float]) -> None:
        """
        Spawn multiple enemies at random positions, ensuring minimum
        distance from the player and from each other.
        
        Args:
            count: Number of enemies to spawn
            player_pos: Player position dictionary {"x": x, "y": y}
        """
        # Clear existing enemies
        self.enemies.clear()
        
        # Keep the primary enemy as the first enemy for compatibility
        self.primary_enemy = self._create_enemy()
        self.enemies.append(self.primary_enemy)
        
        # Position the primary enemy
        self._position_enemy(
            self.primary_enemy, 
            player_pos, 
            []
        )
        
        # Create additional enemies if needed
        for _ in range(1, count):
            enemy = self._create_enemy()
            
            # Get positions of existing enemies
            existing_positions = [
                (e.pos["x"], e.pos["y"]) for e in self.enemies
            ]
            
            # Position the new enemy
            self._position_enemy(
                enemy,
                player_pos,
                existing_positions
            )
            
            # Add to the enemy list
            self.enemies.append(enemy)
    
    def _position_enemy(
        self,
        enemy: EnemyPlay,
        player_pos: Dict[str, float],
        other_enemy_positions: List[tuple]
    ) -> None:
        """
        Position an enemy away from the player and other enemies.
        
        Args:
            enemy: The enemy to position
            player_pos: Player's position {"x": x, "y": y}
            other_enemy_positions: List of (x, y) tuples for other enemies
        """
        max_attempts = 50
        for _ in range(max_attempts):
            # Generate random position
            x = random.randint(0, self.screen_width - enemy.size)
            y = random.randint(0, self.screen_height - enemy.size)
            
            # Calculate distance to player
            dist_to_player = math.sqrt(
                (x - player_pos["x"])**2 + 
                (y - player_pos["y"])**2
            )
            
            # Check if it's far enough from player
            if dist_to_player < config.MIN_DISTANCE:
                continue
                
            # Check distance to other enemies
            too_close = False
            for other_x, other_y in other_enemy_positions:
                dist = math.sqrt(
                    (x - other_x)**2 + 
                    (y - other_y)**2
                )
                if dist < config.MIN_DISTANCE:
                    too_close = True
                    break
                    
            if not too_close:
                # Found a good position
                enemy.set_position(x, y)
                return
                
        # If we get here, we couldn't find a good position
        # Just use a random position as a fallback
        x = random.randint(0, self.screen_width - enemy.size)
        y = random.randint(0, self.screen_height - enemy.size)
        enemy.set_position(x, y)
    
    def update(
        self,
        player_x: float,
        player_y: float,
        player_step: float,
        current_time: int
    ) -> None:
        """
        Update all enemies.
        
        Args:
            player_x: Player's x position
            player_y: Player's y position
            player_step: Player's movement speed
            current_time: Current game time in milliseconds
        """
        # Handle enemy respawning
        self._handle_respawns(current_time)
        
        # Update each enemy
        for enemy in self.enemies:
            if enemy.visible:
                try:
                    enemy.update_movement(
                        player_x,
                        player_y,
                        player_step,
                        current_time
                    )
                except Exception as e:
                    logging.error(f"Error updating enemy movement: {e}")
    
    def _handle_respawns(self, current_time: int) -> None:
        """
        Check and handle any pending enemy respawns.
        
        Args:
            current_time: Current game time in milliseconds
        """
        # Check each respawn timer
        for enemy_id in list(self.respawn_timers.keys()):
            if current_time >= self.respawn_timers[enemy_id]:
                # Time to respawn this enemy
                if enemy_id < len(self.enemies):
                    self.enemies[enemy_id].show(current_time)
                    # Second show call intentional for fade-in effect
                    self.enemies[enemy_id].show(current_time)
                # Remove timer after respawning
                del self.respawn_timers[enemy_id]
                
        # Update is_respawning flag based on any remaining timers
        self.is_respawning = bool(self.respawn_timers)
    
    def check_collision_with_player(
        self,
        player_rect: pygame.Rect
    ) -> Tuple[bool, Optional[EnemyPlay]]:
        """
        Check if any visible enemy collides with the player.
        
        Args:
            player_rect: Rectangle representing the player's hitbox
            
        Returns:
            Tuple of (collision_detected, colliding_enemy)
        """
        for enemy in self.enemies:
            if not enemy.visible:
                continue
                
            enemy_rect = pygame.Rect(
                enemy.pos["x"],
                enemy.pos["y"],
                enemy.size,
                enemy.size
            )
            
            if player_rect.colliderect(enemy_rect):
                return True, enemy
                
        return False, None
    
    def handle_missile_collision(
        self,
        missile_rect: pygame.Rect,
        current_time: int
    ) -> Tuple[bool, Optional[EnemyPlay]]:
        """
        Check if a missile collides with any enemy.
        
        Args:
            missile_rect: Rectangle representing the missile's hitbox
            current_time: Current game time in milliseconds
            
        Returns:
            Tuple of (hit_detected, hit_enemy)
        """
        for i, enemy in enumerate(self.enemies):
            if not enemy.visible:
                continue
                
            enemy_rect = pygame.Rect(
                enemy.pos["x"],
                enemy.pos["y"],
                enemy.size,
                enemy.size
            )
            
            if missile_rect.colliderect(enemy_rect):
                # Hide the enemy
                enemy.hide()
                
                # Set up respawn timer
                self.respawn_timers[i] = current_time + self.respawn_delay
                self.is_respawning = True
                
                logging.info(f"Missile hit enemy {i}, will respawn in {self.respawn_delay}ms")
                return True, enemy
                
        return False, None
