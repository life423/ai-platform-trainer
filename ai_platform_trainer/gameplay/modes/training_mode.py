import math
import logging
import random
import pygame


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_lifespan = {}
        self.missile_sequences = {}

    def update(self):
        current_time = pygame.time.get_ticks()

        if self.game.enemy and self.game.player:
            enemy_x = self.game.enemy.pos["x"]
            enemy_y = self.game.enemy.pos["y"]

            self.game.player.update(enemy_x, enemy_y)
            self.game.enemy.update_movement(
                self.game.player.position["x"],
                self.game.player.position["y"],
                self.game.player.step,
            )
            
            # Check for player-enemy collision
            if self.game.check_collision():
                logging.info("Training mode: Collision detected between player and enemy.")
                # Log collision data
                self.log_collision_data(current_time, True)
                
                # Handle collision (hide enemy and set respawn)
                if self.game.enemy:
                    self.game.enemy.hide()
                self.game.is_respawning = True
                self.game.respawn_timer = current_time + self.game.respawn_delay
            else:
                # Log normal (non-collision) data
                self.log_collision_data(current_time, False)

            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            if random.random() < getattr(self, "missile_fire_prob", 0.1):
                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile(enemy_x, enemy_y)
                    self.missile_cooldown = 120

                    if self.game.player.missiles:
                        missile = self.game.player.missiles[0]
                        self.missile_lifespan[missile] = (
                            missile.birth_time,
                            missile.lifespan,
                        )
                        self.missile_sequences[missile] = []

        current_missiles = self.game.player.missiles[:]
        for missile in current_missiles:
            if missile in self.missile_lifespan:
                birth_time, lifespan = self.missile_lifespan[missile]
                if current_time - birth_time >= lifespan:
                    self.finalize_missile_sequence(missile, success=False)
                    self.game.player.missiles.remove(missile)
                    del self.missile_lifespan[missile]
                    logging.debug("Training mode: Missile removed (lifespan expiry).")
                else:
                    missile.update()

                    if missile in self.missile_sequences:
                        player_x = self.game.player.position["x"]
                        player_y = self.game.player.position["y"]
                        enemy_x = self.game.enemy.pos["x"]
                        enemy_y = self.game.enemy.pos["y"]

                        # Calculate missile angle and get last action
                        missile_angle = math.atan2(missile.vy, missile.vx)
                        missile_action = getattr(missile, "last_action", 0.0)

                        self.missile_sequences[missile].append(
                            {
                                "player_x": player_x,
                                "player_y": player_y,
                                "enemy_x": enemy_x,
                                "enemy_y": enemy_y,
                                "missile_x": missile.pos["x"],
                                "missile_y": missile.pos["y"],
                                "missile_angle": missile_angle,
                                "missile_collision": False,
                                "missile_action": missile_action,
                                "timestamp": current_time,
                                "collision": False,  # Field for player-enemy collision
                            }
                        )

                    if self.game.enemy:
                        enemy_rect = pygame.Rect(
                            self.game.enemy.pos["x"],
                            self.game.enemy.pos["y"],
                            self.game.enemy.size,
                            self.game.enemy.size,
                        )
                        if missile.get_rect().colliderect(enemy_rect):
                            logging.info("Missile hit the enemy (training mode).")
                            self.finalize_missile_sequence(missile, success=True)
                            self.game.player.missiles.remove(missile)
                            del self.missile_lifespan[missile]

                            self.game.enemy.hide()
                            self.game.is_respawning = True
                            self.game.respawn_timer = (
                                current_time + self.game.respawn_delay
                            )
                            break

                    if not (
                        0 <= missile.pos["x"] <= self.game.screen_width
                        and 0 <= missile.pos["y"] <= self.game.screen_height
                    ):
                        self.finalize_missile_sequence(missile, success=False)
                        self.game.player.missiles.remove(missile)
                        del self.missile_lifespan[missile]
                        logging.debug("Training Mode: Missile left the screen.")

        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            if self.game.enemy:
                self.game.handle_respawn(current_time)

    def finalize_missile_sequence(self, missile, success: bool) -> None:
        """
        Called when a missile's life ends or collision occurs.
        Logs each frame's data with a final 'missile_collision' outcome.
        """
        if missile not in self.missile_sequences:
            return

        outcome_val = success
        frames = self.missile_sequences[missile]

        for frame_data in frames:
            frame_data["missile_collision"] = outcome_val
            if self.game.data_logger:
                self.game.data_logger.log(frame_data)

        del self.missile_sequences[missile]
        logging.debug(
            f"Finalized missile sequence with success={success}, frames={len(frames)}"
        )
        
    def log_collision_data(self, current_time, is_collision):
        """
        Log data about player and enemy positions with collision information.
        
        Args:
            current_time: Current game time in milliseconds
            is_collision: Boolean indicating whether a player-enemy collision occurred
        """
        if not self.game.data_logger or not self.game.player or not self.game.enemy:
            return
            
        # Calculate distance between player and enemy
        player_x = self.game.player.position["x"]
        player_y = self.game.player.position["y"]
        enemy_x = self.game.enemy.pos["x"]
        enemy_y = self.game.enemy.pos["y"]
        
        dx = player_x - enemy_x
        dy = player_y - enemy_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Prepare position dictionaries in the format needed for validation
        player_pos = {"x": player_x, "y": player_y}
        enemy_pos = {"x": enemy_x, "y": enemy_y}
        collision_val = 1 if is_collision else 0
        
        # Use the new log_data method with validation
        self.game.data_logger.log_data(
            current_time, 
            player_pos, 
            enemy_pos, 
            distance, 
            collision_val
        )
