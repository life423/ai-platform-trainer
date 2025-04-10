# file: ai_platform_trainer/gameplay/modes/play_mode.py

import logging
import pygame
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

        # 4) Missile AI
        if (
            self.game.missile_model
            and self.game.player
            and self.game.player.missiles
        ):
            update_missile_ai(
                self.game.player.missiles,
                self.game.player.position,
                self.game.enemy.pos if self.game.enemy else None,
                self.game._missile_input,
                self.game.missile_model
            )

        # 5) Misc updates
        # Respawn logic
        self.game.handle_respawn(current_time)

        # If enemy is fading in, keep updating alpha
        if self.game.enemy and self.game.enemy.fading_in:
            self.game.enemy.update_fade_in(current_time)

        # Update missiles
        # If you need the enemy position, pass it. Some players do "self.game.player.update_missiles()" with no arguments.
        self.game.player.update_missiles()

        # Check if missiles collide with the enemy
        self.game.check_missile_collisions()