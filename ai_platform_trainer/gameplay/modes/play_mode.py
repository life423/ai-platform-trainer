"""
Play mode game logic for AI Platform Trainer.

This module handles the play mode game loop and mechanics.
"""
import logging
import pygame
import math

from ai_platform_trainer.gameplay.missile_ai_controller import update_missile_ai


class PlayMode:
    def __init__(self, game):
        """
        Holds 'play' mode logic for the game.
        """
        self.game = game

    def update(self, current_time: int) -> None:
        """
        The main update loop for 'play' mode, replacing old play_update() logic in game.py.
        """

        # 1) Player movement & input
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.game.running = False
            return

        # 2) Enemy movement - using enemy manager for multiple enemies if available
        if hasattr(self.game, 'enemy_manager') and self.game.enemy_manager:
            try:
                self.game.enemy_manager.update(
                    self.game.player.position["x"],
                    self.game.player.position["y"],
                    self.game.player.step,
                    current_time
                )
                logging.debug("Multiple enemies updated in play mode.")
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.game.running = False
                return
        # Fallback to single enemy for backward compatibility
        elif self.game.enemy:
            try:
                self.game.enemy.update_movement(
                    self.game.player.position["x"],
                    self.game.player.position["y"],
                    self.game.player.step,
                    current_time,
                )
                logging.debug("Single enemy updated in play mode.")
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.game.running = False
                return

        # 3) Player-Enemy collision - using enemy manager for multiple enemies
        player_rect = pygame.Rect(
            self.game.player.position["x"],
            self.game.player.position["y"],
            self.game.player.size,
            self.game.player.size
        )
        
        # Check for collisions with enemy manager
        if hasattr(self.game, 'enemy_manager') and self.game.enemy_manager:
            collision, enemy = self.game.enemy_manager.check_collision_with_player(player_rect)
            if collision:
                # Enemy is already hidden by the manager
                self.game.is_respawning = True
                self.game.respawn_timer = current_time + self.game.respawn_delay
                logging.info("Player collided with enemy (managed). Respawning.")
        # Fallback to legacy collision detection
        elif self.game.check_collision():
            if self.game.enemy:
                self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay
            logging.info("Player collided with enemy (legacy). Respawning.")

        # 4) Update obstacle collisions
        if hasattr(self.game, 'obstacle_manager') and self.game.obstacle_manager:
            obstacle_collision, _ = self.game.obstacle_manager.check_collision(player_rect)
            if obstacle_collision:
                logging.debug("Player collided with obstacle.")

        # 5) Missile AI targeting
        if (
            self.game.missile_model
            and self.game.player
            and self.game.player.missiles
        ):
            # Get target position - prefer closest enemy from manager if available
            target_pos = None
            has_manager = (hasattr(self.game, 'enemy_manager') and 
                          self.game.enemy_manager and 
                          self.game.enemy_manager.enemies)
            if has_manager:
                target_pos = self._get_closest_enemy_position()
            elif self.game.enemy:
                target_pos = self.game.enemy.pos

            update_missile_ai(
                self.game.player.missiles,
                self.game.player.position,
                target_pos,
                self.game._missile_input,
                self.game.missile_model
            )

        # 6) Misc updates
        # Respawn logic
        self.game.handle_respawn(current_time)

        # Update enemy fade-in (for single enemy backward compatibility)
        if self.game.enemy and hasattr(self.game.enemy, 'fading_in') and self.game.enemy.fading_in:
            self.game.enemy.update_fade_in(current_time)

        # Update missiles
        if self.game.player:
            self.game.player.update_missiles()

        # Check if missiles collide with enemies
        self.check_missile_collisions(current_time)

    def _get_closest_enemy_position(self) -> dict:
        """Find the closest visible enemy to target with missiles."""
        if not self.game.enemy_manager or not self.game.player:
            # Fallback to single enemy position
            return self.game.enemy.pos if self.game.enemy else {"x": 0, "y": 0}
            
        closest_dist = float('inf')
        closest_pos = {"x": 0, "y": 0}
        
        for enemy in self.game.enemy_manager.enemies:
            if not enemy.visible:
                continue
                
            # Calculate distance to this enemy
            dx = enemy.pos["x"] - self.game.player.position["x"]
            dy = enemy.pos["y"] - self.game.player.position["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_pos = enemy.pos
                
        # If no visible enemies found, use position of first enemy as fallback
        if closest_dist == float('inf') and self.game.enemy_manager.enemies:
            return self.game.enemy_manager.enemies[0].pos
            
        return closest_pos

    def check_missile_collisions(self, current_time: int) -> None:
        """
        Check for collisions between missiles and enemies or obstacles.
        This is an updated version that works with multiple enemies and obstacles.
        """
        if not self.game.player:
            return

        # Check each missile
        for missile in self.game.player.missiles[:]:  # Copy to avoid modification during iteration
            if not missile.active:
                continue

            # Create a rect for the missile
            missile_rect = pygame.Rect(
                missile.position["x"],
                missile.position["y"],
                missile.size,
                missile.size
            )

            # Check collision with enemies from enemy manager
            if self.game.enemy_manager:
                hit, _ = self.game.enemy_manager.handle_missile_collision(
                    missile_rect, 
                    current_time
                )
                
                if hit:
                    missile.active = False
                    logging.info("Missile hit enemy from enemy manager")
                    continue  # Don't check other collisions if already hit
            # Fallback to single enemy collision
            elif self.game.enemy and self.game.enemy.visible:
                enemy_rect = pygame.Rect(
                    self.game.enemy.pos["x"],
                    self.game.enemy.pos["y"],
                    self.game.enemy.size,
                    self.game.enemy.size
                )
                
                if missile_rect.colliderect(enemy_rect):
                    missile.active = False
                    self.game.enemy.hide()
                    self.game.is_respawning = True
                    self.game.respawn_timer = current_time + self.game.respawn_delay
                    logging.info("Missile hit single enemy")
                    continue
                
            # Check collision with obstacles
            if hasattr(self.game, 'obstacle_manager') and self.game.obstacle_manager:
                obstacle_hit, _ = self.game.obstacle_manager.check_collision(missile_rect)
                if obstacle_hit:
                    missile.active = False
                    logging.debug("Missile hit obstacle")
