"""
Sprite manager for AI Platform Trainer.

This module handles loading, scaling, rotating and caching sprites for game
entities. A sprite is scaled once per (name, size), and each rotated variant
is made once per (name, size, angle), so the work happens on the first frame
that needs it and is reused after that.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import pygame

# Type aliases
Color = Tuple[int, int, int]
Size = Tuple[int, int]
Position = Dict[str, float]  # {"x": x, "y": y}


def _as_size(size) -> Size:
    return (int(size[0]), int(size[1]))


class SpriteManager:
    """
    Manages sprite loading, caching, and rendering.

    This class handles the loading of sprite assets and creates
    placeholder sprites if assets are not available.
    """

    def __init__(self, sprites_dir: str = "assets/sprites"):
        """
        Initialize the sprite manager.

        Args:
            sprites_dir: Directory containing sprite assets
        """
        self.sprites_dir = sprites_dir
        # Decoded image files before any scaling; None marks a missing file
        self._source_images: Dict[str, Optional[pygame.Surface]] = {}
        self.sprites: Dict[Tuple[str, Size], pygame.Surface] = {}
        self.animations: Dict[Tuple[str, Size, int], List[pygame.Surface]] = {}
        self._rotated_sprites: Dict[Tuple[str, Size, int], pygame.Surface] = {}

        # Placeholder colors for entities
        self.placeholder_colors = {
            "player": (0, 128, 255),  # Blue
            "enemy": (255, 50, 50),  # Red
            "missile": (255, 255, 50),  # Yellow
        }

    def load_sprite(self, name: str, size: Size) -> pygame.Surface:
        """
        Load a sprite image scaled to size, or create a placeholder if not found.

        Sprites are cached by (name, size), so the same name at two sizes
        gives two surfaces.

        Args:
            name: Name of the sprite to load
            size: Size (width, height) of the sprite

        Returns:
            Loaded sprite surface or a placeholder
        """
        size = _as_size(size)
        key = (name, size)
        if key in self.sprites:
            return self.sprites[key]

        source = self._load_source(name)
        if source is not None:
            # smoothscale keeps edges clean when shrinking (scale() is jagged)
            sprite = pygame.transform.smoothscale(source, size)
        else:
            sprite = self._create_placeholder(name, size)
        self.sprites[key] = sprite
        return sprite

    def _load_source(self, name: str) -> Optional[pygame.Surface]:
        """Decode assets/sprites/<name>.png once, or None if the file is missing."""
        if name not in self._source_images:
            image = None
            path = os.path.join(self.sprites_dir, f"{name}.png")
            if os.path.exists(path):
                try:
                    image = pygame.image.load(path).convert_alpha()
                except pygame.error:
                    # If loading fails, the placeholder is used instead
                    image = None
            self._source_images[name] = image
        return self._source_images[name]

    def _create_placeholder(self, name: str, size: Size) -> pygame.Surface:
        """
        Create a placeholder sprite with specified color.

        Args:
            name: Name of the entity (determines color)
            size: Size (width, height) of the sprite

        Returns:
            Placeholder sprite surface
        """
        # Get the placeholder color for the entity type
        color = self.placeholder_colors.get(name, (200, 200, 200))  # Default to gray

        # Create a surface with alpha channel
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        if name == "player":
            # Triangle shape for player
            width, height = size
            points = [
                (width // 2, 0),  # Top point
                (0, height),  # Bottom left
                (width, height),  # Bottom right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif name == "enemy":
            # Pentagon shape for enemy
            width, height = size
            points = [
                (width // 2, 0),  # Top point
                (0, height // 2),  # Middle left
                (width // 4, height),  # Bottom left
                (3 * width // 4, height),  # Bottom right
                (width, height // 2),  # Middle right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif name == "missile":
            # Elongated pentagon for missile
            width, height = size
            points = [
                (width // 2, 0),  # Top point
                (width // 4, height // 4),  # Upper left
                (0, height),  # Bottom left
                (width, height),  # Bottom right
                (3 * width // 4, height // 4),  # Upper right
            ]
            pygame.draw.polygon(sprite, color, points)

        else:
            # Simple rectangle with slight transparency for other entities
            pygame.draw.rect(sprite, (*color, 220), pygame.Rect(0, 0, size[0], size[1]))

        return sprite

    def render(
        self,
        screen: pygame.Surface,
        entity_type: str,
        position: Union[Position, Tuple[float, float]],
        size: Size,
        rotation: float = 0,
        alpha: int = 255,
    ) -> None:
        """
        Render a sprite to the screen.

        Args:
            screen: Pygame surface to render to
            entity_type: Type of entity to render
            position: Top-left (x, y) of the entity's logical box
            size: Size (width, height) of the sprite
            rotation: Rotation angle in degrees, counter-clockwise
            alpha: Opacity from 0 to 255 (used for the fade-in)
        """
        # Get position as (x, y) tuple (top-left of the entity's bounding box)
        if isinstance(position, dict):
            pos = (position["x"], position["y"])
        else:
            pos = position

        center = (pos[0] + size[0] / 2, pos[1] + size[1] / 2)
        self.render_centered(screen, entity_type, center, size, rotation, alpha)

    def render_centered(
        self,
        screen: pygame.Surface,
        entity_type: str,
        center: Tuple[float, float],
        size: Size,
        rotation: float = 0,
        alpha: int = 255,
    ) -> None:
        """Render a sprite centred on center, rotated by rotation degrees."""
        sprite = self._get_rotated_sprite(entity_type, size, rotation)
        if alpha < 255:
            # Fade on a copy, so the cached surface keeps its full opacity
            sprite = sprite.copy()
            sprite.set_alpha(max(0, int(alpha)))

        # A rotated surface's bounding box is larger than the original
        # (especially at 45-degree angles), so it is centred on the entity's
        # logical box rather than drawn at its unrotated top-left - otherwise
        # it visibly drifts away from the entity's position.
        rect = sprite.get_rect(center=(int(round(center[0])), int(round(center[1]))))
        screen.blit(sprite, rect)

    def _get_rotated_sprite(
        self, entity_type: str, size: Size, rotation: float
    ) -> pygame.Surface:
        """
        Get a sprite rotated by the given angle, caching each rotated
        variant so repeated frames at the same angle (e.g. a player
        facing one of 8 fixed compass directions) don't re-rotate the
        surface every time. Angles are cached to the whole degree.
        """
        size = _as_size(size)
        base_sprite = self.load_sprite(entity_type, size)
        rotation_key = round(rotation) % 360
        if rotation_key == 0:
            return base_sprite

        cache_key = (entity_type, size, rotation_key)
        if cache_key not in self._rotated_sprites:
            self._rotated_sprites[cache_key] = pygame.transform.rotozoom(
                base_sprite, rotation_key, 1.0
            )
        return self._rotated_sprites[cache_key]

    def load_animation(
        self, name: str, size: Size, frames: int = 4
    ) -> List[pygame.Surface]:
        """
        Load or create a simple animation sequence.

        Args:
            name: Base name of the animation
            size: Size (width, height) of each frame
            frames: Number of frames in the animation

        Returns:
            List of animation frame surfaces
        """
        size = _as_size(size)
        animation_key = (name, size, frames)
        if animation_key in self.animations:
            return self.animations[animation_key]

        # Try to load animation frames from files
        animation_frames = []
        for i in range(frames):
            source = self._load_source(f"{name}_{i}")
            if source is not None:
                animation_frames.append(pygame.transform.smoothscale(source, size))
            else:
                # Create placeholder frame with slight variation
                animation_frames.append(
                    self._create_animation_placeholder(name, size, i, frames)
                )

        self.animations[animation_key] = animation_frames
        return animation_frames

    def _create_animation_placeholder(
        self, name: str, size: Size, frame_index: int, total_frames: int
    ) -> pygame.Surface:
        """
        Create a placeholder animation frame.

        Args:
            name: Name of the entity
            size: Size (width, height) of the sprite
            frame_index: Current frame index
            total_frames: Total number of frames

        Returns:
            Placeholder animation frame surface
        """
        # Get base sprite
        base_sprite = self._create_placeholder(name, size)

        # For animation, slightly modify the sprite based on frame index
        # This is a simple placeholder effect
        progress = frame_index / max(1, total_frames - 1)

        # Create a pulsing effect
        pulse_factor = 0.8 + 0.4 * abs((progress * 2) - 1)  # Values between 0.8 and 1.2

        # Apply the pulse effect
        width, height = size
        scaled_w = int(width * pulse_factor)
        scaled_h = int(height * pulse_factor)

        # Center the scaled sprite
        offset_x = (width - scaled_w) // 2
        offset_y = (height - scaled_h) // 2

        # Create a new surface for the animation frame
        frame = pygame.Surface(size, pygame.SRCALPHA)

        # Scale the base sprite and blit to the frame
        scaled_sprite = pygame.transform.smoothscale(base_sprite, (scaled_w, scaled_h))
        frame.blit(scaled_sprite, (offset_x, offset_y))

        return frame
