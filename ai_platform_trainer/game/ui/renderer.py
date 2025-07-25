"""
Simple renderer for AI Platform Trainer.

Handles basic rendering operations.
"""
import pygame


class Renderer:
    """Simple renderer for game entities and UI."""
    
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
    
    def clear(self, color=(0, 0, 0)):
        """Clear screen with specified color."""
        self.screen.fill(color)
    
    def render_text(self, text: str, x: int, y: int, color=(255, 255, 255), font_size="normal"):
        """Render text at specified position."""
        font = self.font if font_size == "normal" else self.small_font
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, (x, y))
    
    def render_fps(self, clock: pygame.time.Clock):
        """Render FPS counter."""
        fps = int(clock.get_fps())
        self.render_text(f"FPS: {fps}", 10, 10, (255, 255, 0), "small")