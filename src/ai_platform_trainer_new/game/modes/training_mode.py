"""
Clean Training Mode

This module provides a simplified training mode for data collection.
"""
import pygame
import random
import logging
import math

from config.settings import settings
from ..entities.player import Player
from ..entities.enemy import Enemy
from ...utils.data_logger import DataLogger


class TrainingMode:
    """Training mode for collecting gameplay data."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create entities
        self.player = Player(screen_width, screen_height)
        self.enemy = Enemy(screen_width, screen_height)
        
        # Data collection
        self.data_logger = DataLogger()
        
        # Respawn system
        self.respawn_delay = settings.RESPAWN_DELAY
        self.respawn_timer = 0
        self.is_respawning = False
        
        # Auto missile firing
        self.missile_cooldown = 0
        self.missile_fire_prob = 0.1
        
        # Auto-save
        self.last_save_time = 0
        self.save_interval = settings.AUTO_SAVE_INTERVAL
        
        # Initialize positions
        self._spawn_entities()
        
        logging.info("Training mode initialized")
    
    def handle_event(self, event):
        """Handle training mode events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.shoot_missile(self.enemy.pos)
    
    def update(self):
        """Update training mode."""
        current_time = pygame.time.get_ticks()
        
        # Update entities
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])
        self.enemy.update_movement(
            self.player.position["x"],
            self.player.position["y"],
            self.player.step
        )
        
        # Log data every frame
        self._log_frame_data(current_time)
        
        # Auto-save periodically
        if current_time - self.last_save_time >= self.save_interval:
            self._auto_save_data()
            self.last_save_time = current_time
        
        # Auto-fire missiles occasionally
        self._handle_auto_missile_fire()
        
        # Check collisions
        self._check_collisions(current_time)
        
        # Handle respawning
        if self.is_respawning and current_time >= self.respawn_timer:
            self._respawn_enemy()
    
    def _spawn_entities(self):
        """Spawn player and enemy at initial positions."""
        # Player in center
        self.player.position = {
            "x": self.screen_width // 2,
            "y": self.screen_height // 2
        }
        
        # Enemy at random position away from player
        while True:
            x = random.randint(0, self.screen_width - self.enemy.size)
            y = random.randint(0, self.screen_height - self.enemy.size)
            
            # Check distance from player
            dx = x - self.player.position["x"]
            dy = y - self.player.position["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > settings.MIN_SPAWN_DISTANCE:
                self.enemy.set_position(x, y)
                break
    
    def _handle_auto_missile_fire(self):
        """Handle automatic missile firing."""
        if self.missile_cooldown > 0:
            self.missile_cooldown -= 1
        
        if (random.random() < self.missile_fire_prob and 
            self.missile_cooldown <= 0 and 
            len(self.player.missiles) == 0):
            
            self.player.shoot_missile(self.enemy.pos)
            self.missile_cooldown = 120  # 2 seconds at 60 FPS
    
    def _check_collisions(self, current_time: int):
        """Check for collisions."""
        # Player-Enemy collision
        if self._entities_collide(self.player, self.enemy):
            logging.info("Player-Enemy collision")
            self._trigger_respawn(current_time)
        
        # Missile-Enemy collisions
        for missile in self.player.missiles[:]:
            if self._missile_enemy_collide(missile, self.enemy):
                logging.info("Missile hit enemy")
                self.player.missiles.remove(missile)
                self._trigger_respawn(current_time)
                break
    
    def _entities_collide(self, entity1, entity2) -> bool:
        """Check collision between two entities."""
        if not entity2.visible:
            return False
        
        rect1 = pygame.Rect(entity1.position["x"], entity1.position["y"], 
                           entity1.size, entity1.size)
        rect2 = pygame.Rect(entity2.pos["x"], entity2.pos["y"], 
                           entity2.size, entity2.size)
        return rect1.colliderect(rect2)
    
    def _missile_enemy_collide(self, missile, enemy) -> bool:
        """Check collision between missile and enemy."""
        if not enemy.visible:
            return False
        
        missile_rect = pygame.Rect(missile.pos["x"], missile.pos["y"], 
                                  missile.size, missile.size)
        enemy_rect = pygame.Rect(enemy.pos["x"], enemy.pos["y"], 
                                enemy.size, enemy.size)
        return missile_rect.colliderect(enemy_rect)
    
    def _trigger_respawn(self, current_time: int):
        """Trigger enemy respawn."""
        self.enemy.hide()
        self.is_respawning = True
        self.respawn_timer = current_time + self.respawn_delay
    
    def _respawn_enemy(self):
        """Respawn the enemy at a new location."""
        while True:
            x = random.randint(0, self.screen_width - self.enemy.size)
            y = random.randint(0, self.screen_height - self.enemy.size)
            
            # Check distance from player
            dx = x - self.player.position["x"]
            dy = y - self.player.position["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > settings.MIN_SPAWN_DISTANCE:
                break
        
        self.enemy.set_position(x, y)
        self.enemy.show(pygame.time.get_ticks())
        self.is_respawning = False
    
    def _log_frame_data(self, current_time: int):
        """Log comprehensive frame data."""
        player_x = self.player.position["x"]
        player_y = self.player.position["y"]
        enemy_x = self.enemy.pos["x"]
        enemy_y = self.enemy.pos["y"]
        
        # Calculate distances and angles
        dx = enemy_x - player_x
        dy = enemy_y - player_y
        distance_to_enemy = math.sqrt(dx * dx + dy * dy)
        angle_to_enemy = math.atan2(dy, dx)
        
        # Check for active missiles
        has_missile = len(self.player.missiles) > 0
        missile_x = missile_y = missile_vx = missile_vy = 0.0
        if has_missile:
            missile = self.player.missiles[0]
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
            "enemy_visible": self.enemy.visible,
            "player_step": self.player.step
        }
        
        self.data_logger.log(frame_data)
    
    def _auto_save_data(self):
        """Auto-save collected data."""
        if self.data_logger.data:
            self.data_logger.save()
            data_count = len(self.data_logger.data)
            logging.info(f"Auto-saved {data_count} data points")
    
    def render(self, screen: pygame.Surface):
        """Render training mode."""
        from ..ui.renderer import Renderer
        renderer = Renderer(screen)
        
        # Render entities
        renderer.render_player(self.player)
        renderer.render_enemy(self.enemy)
        renderer.render_missiles(self.player.missiles)
        
        # Render UI
        renderer.render_text("TRAINING MODE - ESC to menu", (10, 10))
        renderer.render_text(f"Data Points: {len(self.data_logger.data)}", (10, 50))
        
        if self.is_respawning:
            # Use dark orange for warning text - good contrast on light blue background
            renderer.render_text("Enemy Respawning...", (10, 90), (255, 140, 0))