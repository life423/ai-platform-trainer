"""
Clean Renderer System

This module provides simplified rendering for game entities.
"""
import pygame
from typing import Optional, List

from config.settings import settings


class Renderer:
    """Handles rendering of game entities."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
    
    def render_player(self, player):
        """Render the player entity."""
        if hasattr(player, 'position') and hasattr(player, 'size'):
            pygame.draw.rect(
                self.screen,
                settings.COLOR_PLAYER,
                (player.position["x"], player.position["y"], player.size, player.size)
            )
    
    def render_enemy(self, enemy):
        """Render the enemy entity."""
        if hasattr(enemy, 'pos') and hasattr(enemy, 'size') and getattr(enemy, 'visible', True):
            pygame.draw.rect(
                self.screen,
                settings.COLOR_ENEMY,
                (enemy.pos["x"], enemy.pos["y"], enemy.size, enemy.size)
            )
    
    def render_missiles(self, missiles: List):
        """Render missile entities."""
        for missile in missiles:
            if hasattr(missile, 'pos') and hasattr(missile, 'size'):
                pygame.draw.rect(
                    self.screen,
                    settings.COLOR_MISSILE,
                    (missile.pos["x"], missile.pos["y"], missile.size, missile.size)
                )
    
    def render_text(self, text: str, position: tuple, color: tuple = (255, 255, 255)):
        """Render text at the specified position."""
        text_surface = self.font.render(text, True, color)
        self.screen.blit(text_surface, position)
    
    def clear_screen(self):
        """Clear the screen with background color."""
        self.screen.fill(settings.COLOR_BACKGROUND)