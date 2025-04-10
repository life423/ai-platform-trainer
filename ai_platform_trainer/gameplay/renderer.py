"""
Renderer module for AI Platform Trainer.

This handles drawing all game entities to the screen.
"""
import pygame
import logging
from typing import Dict, Optional, List

# Import sprite manager and background manager
from ai_platform_trainer.utils.sprite_manager import SpriteManager


class Renderer:
    """Handles rendering of all game entities."""

    def __init__(self, screen: pygame.Surface):
        """
        Initialize the renderer.

        Args:
            screen: The pygame Surface to render to
        """
        self.screen = screen
        
        # Initialize sprite and background managers
        self.sprite_manager = SpriteManager()
        
        # Text rendering setup
        self.font = pygame.font.Font(None, 24)
        self.score_font = pygame.font.Font(None, 36)

    def render(self, menu, player, enemy, menu_active):
        """
        Legacy render method for backward compatibility.
        
        Args:
            menu: The menu object
            player: The player object
            enemy: The enemy object
            menu_active: Whether the menu is active
        """
        # If menu is active, let the menu handle rendering
        if menu_active:
            return
        
        # Clear screen with sky blue
        self.screen.fill((135, 206, 235))
        
        # Render entities
        if player:
            self.render_player(player)
            
            # Render player missiles
            for missile in player.missiles:
                if missile.active:
                    self.render_missile(missile)
        
        if enemy and enemy.visible:
            self.render_enemy(enemy)

    def render_player(self, player):
        """
        Render the player entity with sprites.
        
        Args:
            player: Player entity to render
        """
        if not hasattr(player, 'position') or not hasattr(player, 'size'):
            return
            
        # Determine sprite size
        size = (player.size, player.size)
        
        # Render the player sprite
        self.sprite_manager.render(
            screen=self.screen,
            entity_type="player",
            position=player.position,
            size=size
        )
        
    def render_enemy(self, enemy):
        """
        Render the enemy entity with sprites.
        
        Args:
            enemy: Enemy entity to render
        """
        if not hasattr(enemy, 'pos') or not hasattr(enemy, 'size') or not enemy.visible:
            return
            
        # Determine sprite size
        size = (enemy.size, enemy.size)
        
        # Determine transparency/alpha if enemy has a fade_in property
        alpha = 255
        if hasattr(enemy, 'fade_in') and enemy.fade_in:
            current_time = pygame.time.get_ticks()
            # Calculate fade over 500ms
            if current_time - enemy.fade_start_time < 500:
                alpha = int(255 * (current_time - enemy.fade_start_time) / 500)
        
        # Render the enemy sprite
        sprite = self.sprite_manager.load_sprite("enemy", size)
        sprite.set_alpha(alpha)
        self.screen.blit(sprite, (enemy.pos["x"], enemy.pos["y"]))
        
    def render_missile(self, missile):
        """
        Render a missile entity with sprites.
        
        Args:
            missile: Missile entity to render
        """
        if not hasattr(missile, 'position') or not hasattr(missile, 'size') or not missile.active:
            return
            
        # Determine sprite size - make it a bit more elongated
        width = missile.size
        height = missile.size * 1.5
        size = (width, int(height))
        
        # Calculate rotation angle - missiles point in their movement direction
        angle = 0
        if hasattr(missile, 'direction'):
            # In radians, convert to degrees for pygame
            angle = -missile.direction * 180 / 3.14159
        
        # Render the missile sprite with rotation
        self.sprite_manager.render(
            screen=self.screen,
            entity_type="missile",
            position=missile.position,
            size=size,
            rotation=angle
        )

    def render_obstacle(self, obstacle):
        """
        Render an obstacle (wall) with sprites.
        
        Args:
            obstacle: Obstacle entity to render
        """
        if not hasattr(obstacle, 'position') or not hasattr(obstacle, 'width') or not obstacle.visible:
            return
            
        # Determine sprite size
        size = (obstacle.width, obstacle.height)
        
        # Determine sprite type based on orientation
        sprite_name = obstacle.sprite_name if hasattr(obstacle, 'sprite_name') else "wall_h"
        
        # Render the obstacle sprite
        self.sprite_manager.render(
            screen=self.screen,
            entity_type=sprite_name,
            position=obstacle.position,
            size=size
        )
        
    def render_score(self, score: int):
        """
        Render the player's score.
        
        Args:
            score: Player's current score
        """
        score_text = self.score_font.render(f"Score: {score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))
        
    def render_game_over(self):
        """Render the game over screen."""
        game_over_text = self.score_font.render("GAME OVER", True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(
            self.screen.get_width() // 2,
            self.screen.get_height() // 2
        ))
        self.screen.blit(game_over_text, text_rect)
        
        restart_text = self.font.render("Press R to restart or ESC to exit", True, (255, 255, 255))
        restart_rect = restart_text.get_rect(center=(
            self.screen.get_width() // 2,
            self.screen.get_height() // 2 + 50
        ))
        self.screen.blit(restart_text, restart_rect)
