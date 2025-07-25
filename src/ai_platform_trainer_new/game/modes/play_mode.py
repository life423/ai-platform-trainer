"""
Clean Play Mode

This module provides a simplified play mode for human vs AI gameplay.
"""
import pygame
import random
import logging
import math

from config.settings import settings
from ..entities.player import Player
from ..entities.enemy import Enemy


class PlayMode:
    """Play mode for human vs AI gameplay."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create entities
        self.player = Player(screen_width, screen_height)
        self.enemy = Enemy(screen_width, screen_height)
        
        # Respawn system
        self.respawn_delay = settings.RESPAWN_DELAY
        self.respawn_timer = 0
        self.is_respawning = False
        
        # Initialize positions
        self._spawn_entities()
        
        logging.info("Play mode initialized")
    
    def handle_event(self, event):
        """Handle play mode events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.shoot_missile(self.enemy.pos)
    
    def update(self):
        """Update play mode."""
        current_time = pygame.time.get_ticks()
        
        # Update entities
        self.player.update(self.enemy.pos["x"], self.enemy.pos["y"])
        self.enemy.update_movement(
            self.player.position["x"],
            self.player.position["y"],
            self.player.step
        )
        
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
    
    def render(self, screen: pygame.Surface):
        """Render play mode."""
        from ..ui.renderer import Renderer
        renderer = Renderer(screen)
        
        # Render entities
        renderer.render_player(self.player)
        renderer.render_enemy(self.enemy)
        renderer.render_missiles(self.player.missiles)
        
        # Render UI
        renderer.render_text("PLAY MODE - ESC to menu", (10, 10))
        
        if self.is_respawning:
            # Use dark orange for warning text - good contrast on light blue background
            renderer.render_text("Enemy Respawning...", (10, 50), (255, 140, 0))