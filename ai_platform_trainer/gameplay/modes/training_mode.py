
import math
import logging
import random
import pygame
from ai_platform_trainer.gameplay.post_training_processor import post_training_processor


class TrainingMode:
    def __init__(self, game):
        self.game = game
        self.missile_cooldown = 0
        self.missile_lifespan = {}
        self.missile_sequences = {}
        self.last_save_time = 0
        self.save_interval = 30000  # Auto-save every 30 seconds

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

            # Log data every frame for continuous data collection
            self._log_frame_data(current_time)
            
            # Auto-save data periodically
            if current_time - self.last_save_time >= self.save_interval:
                self._auto_save_data()
                self.last_save_time = current_time

            if self.missile_cooldown > 0:
                self.missile_cooldown -= 1

            if random.random() < getattr(self, "missile_fire_prob", 0.1):
                if self.missile_cooldown <= 0 and len(self.game.player.missiles) == 0:
                    self.game.player.shoot_missile({"x": enemy_x, "y": enemy_y})
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

                        # Example distances

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

                            }
                        )

                    if self.game.enemy:
                        enemy_rect = pygame.Rect(
                            self.game.enemy.pos["x"],
                            self.game.enemy.pos["y"],
                            self.game.enemy.size,
                            self.game.enemy.size,
                        )
                        # Create missile rect manually since get_rect() might not work
                        missile_rect = pygame.Rect(
                            missile.pos["x"] - 5,  # Missile size/2
                            missile.pos["y"] - 5,
                            10, 10  # Missile size
                        )
                        if missile_rect.colliderect(enemy_rect):
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
        
        # If this was the last active missile, process the collected data
        if not self.missile_sequences and not self.game.player.missiles:
            self.process_collected_data()
            
    def process_collected_data(self) -> None:
        """
        Process all collected training data:
        1. Get the data from the data logger
        2. Validate and append it to the existing dataset
        3. Retrain the AI models with the combined data
        """
        if not self.game.data_logger or not hasattr(self.game.data_logger, 'data'):
            logging.warning("No data logger available or no data collected")
            return
            
        # Get the collected data
        collected_data = self.game.data_logger.data
        
        if not collected_data:
            logging.warning("No training data was collected during this session")
            return
            
        logging.info(f"Processing {len(collected_data)} data points collected in training")
        
        # Use our validator/trainer to process the data and retrain the models
        success = post_training_processor.process_training_sequence(collected_data)
        
        if success:
            logging.info("Successfully validated data, updated dataset, and retrained models")
            # Reset the data logger for the next training session
            self.game.data_logger.data = []
        else:
            logging.error("Failed to process collected data and retrain models")
            
    def _log_frame_data(self, current_time: int) -> None:
        """
        Log frame data for continuous data collection.
        Collects comprehensive state information every frame.
        """
        if not self.game.data_logger:
            return
            
        player_x = self.game.player.position["x"]
        player_y = self.game.player.position["y"]
        enemy_x = self.game.enemy.pos["x"]
        enemy_y = self.game.enemy.pos["y"]
        
        # Calculate distances and angles
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        distance_to_enemy = math.sqrt(dx * dx + dy * dy)
        angle_to_enemy = math.atan2(dy, dx)
        
        # Check for active missiles
        has_missile = len(self.game.player.missiles) > 0
        missile_x = missile_y = missile_vx = missile_vy = 0.0
        if has_missile:
            missile = self.game.player.missiles[0]
            missile_x = missile.pos["x"]
            missile_y = missile.pos["y"]
            missile_vx = missile.vx
            missile_vy = missile.vy
        
        # Collect comprehensive frame data
        frame_data = {
            "timestamp": current_time,
            "player_x": player_x,
            "player_y": player_y,
            "enemy_x": enemy_x,
            "enemy_y": enemy_y,
            "distance_to_enemy": distance_to_enemy,
            "angle_to_enemy": angle_to_enemy,
            "has_missile": has_missile,
            "missile_x": missile_x,
            "missile_y": missile_y,
            "missile_vx": missile_vx,
            "missile_vy": missile_vy,
            "enemy_visible": self.game.enemy.visible,
            "player_step": self.game.player.step
        }
        
        self.game.data_logger.log(frame_data)
        
    def _auto_save_data(self) -> None:
        """
        Auto-save collected data to prevent data loss.
        """
        if self.game.data_logger and self.game.data_logger.data:
            self.game.data_logger.save()
            data_count = len(self.game.data_logger.data)
            logging.info(f"Auto-saved {data_count} data points to training dataset")
