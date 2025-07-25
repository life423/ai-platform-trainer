"""
Play mode game logic for AI Platform Trainer.

This module handles the play mode game loop and mechanics.
"""
import logging
import pygame

from ai_platform_trainer.ai.inference.missile_controller import update_missile_ai


class PlayMode:
    def __init__(self, game):
        """
        Holds 'play' mode logic for the game.
        """
        self.game = game
        self.space_pressed_last_frame = False

    def update(self, current_time: int) -> None:
        """
        The main update loop for 'play' mode, replacing old play_update() logic in game.py.
        """
        # Check for space bar press to shoot missile (with key repeat prevention)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not self.space_pressed_last_frame and self.game.player and self.game.enemy:
            self.game.player.shoot_missile(self.game.enemy.pos)
            logging.debug("Space bar pressed - shooting missile")
        self.space_pressed_last_frame = keys[pygame.K_SPACE]

        # 1) Player movement & input
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit. Exiting game.")
            self.game.running = False
            return

        # 2) Enemy movement
        if self.game.enemy:
            try:
                self.game.enemy.update_movement(
                    self.game.player.position["x"],
                    self.game.player.position["y"],
                    self.game.player.step,
                    current_time,
                )
                logging.debug("Enemy movement updated in play mode.")
            except Exception as e:
                logging.error(f"Error updating enemy movement: {e}")
                self.game.running = False
                return

        # 3) Player-Enemy collision
        if self.game.check_collision():
            logging.info("Collision detected between player and enemy.")
            if self.game.enemy:
                self.game.enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay
            logging.info("Player-Enemy collision in play mode.")

        # 4) Update missiles
        if self.game.player:
            # Pass enemy position for smart missile tracking
            enemy_pos = self.game.enemy.pos if self.game.enemy and self.game.enemy.visible else None
            self.game.player.update_missiles(enemy_pos)

        # 5) Missile AI
        if (
            self.game.missile_model
            and self.game.player
            and self.game.player.missiles
            and self.game.enemy
        ):
            update_missile_ai(
                self.game.player.missiles,
                self.game.player.position,
                self.game.enemy.pos,
                self.game._missile_input,
                self.game.missile_model
            )

        # 6) Misc updates
        # Respawn logic
        self.game.handle_respawn(current_time)

        # If enemy is fading in, keep updating alpha
        if self.game.enemy and hasattr(self.game.enemy, 'fading_in') and self.game.enemy.fading_in:
            self.game.enemy.update_fade_in(current_time)

        # Check if missiles collide with the enemy
        self.game.check_missile_collisions()