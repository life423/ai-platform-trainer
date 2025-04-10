"""
Sprite manager for AI Platform Trainer.

This module handles loading, caching, and rendering sprites for game entities.
It provides an abstraction layer over pygame's sprite handling to make the
code more modular and maintainable.
"""
import os
import pygame
from typing import Dict, Tuple, Union, List

# Type aliases
Color = Tuple[int, int, int]
Position = Dict[str, float]  # {"x": x, "y": y}


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
        self.sprites: Dict[str, pygame.Surface] = {}
        self.animations: Dict[str, List[pygame.Surface]] = {}

        # Placeholder colors for entities
        self.placeholder_colors = {
            "player": (0, 128, 255),    # Blue
            "enemy": (255, 50, 50),     # Red
            "missile": (255, 255, 50),  # Yellow
            "wall_h": (128, 128, 128),  # Gray
            "wall_v": (128, 128, 128)   # Gray
        }

        # Mapping of entity types to subdirectories
        self.sprite_subdirs = {
            "player": "player",
            "enemy": "enemies",
            "missile": "weapons",
            "explosion": "effects",
            "wall_h": "obstacles",
            "wall_v": "obstacles"
        }

    def load_sprite(self, name: str, size: Tuple[int, int]) -> pygame.Surface:
        """
        Load a sprite image or create a placeholder if not found.

        Args:
            name: Name of the sprite to load
            size: Size (width, height) of the sprite

        Returns:
            Loaded sprite surface or a placeholder
        """
        # Check if sprite is already cached
        cache_key = f"{name}_{size[0]}x{size[1]}"
        if cache_key in self.sprites:
            return self.sprites[cache_key]

        # Determine the appropriate subdirectory
        subdir = self.sprite_subdirs.get(name, "")
        
        # Try to load the sprite from file - check both new and old locations
        sprite_paths = []
        
        # New directory structure with subdirectories
        if subdir:
            sprite_paths.append(os.path.join(self.sprites_dir, subdir, f"{name}.png"))
            # For explosion frames and other indexed sprites
            if "_" in name and name.split("_")[0] in self.sprite_subdirs:
                base, index = name.split("_", 1)
                subdir = self.sprite_subdirs.get(base, "")
                sprite_paths.append(os.path.join(self.sprites_dir, subdir, f"{index}.png"))
        
        # Old/legacy path as fallback
        sprite_paths.append(os.path.join(self.sprites_dir, f"{name}.png"))
        
        # Try each path
        for sprite_path in sprite_paths:
            if os.path.exists(sprite_path):
                try:
                    # Load and scale the sprite
                    sprite = pygame.image.load(sprite_path).convert_alpha()
                    sprite = pygame.transform.scale(sprite, size)
                    self.sprites[cache_key] = sprite
                    return sprite
                except pygame.error:
                    # If loading fails, continue to the next path
                    continue

        # Create placeholder sprite
        sprite = self._create_placeholder(name, size)
        self.sprites[cache_key] = sprite
        return sprite

    def _create_placeholder(self, name: str, size: Tuple[int, int]) -> pygame.Surface:
        """
        Create a placeholder sprite with specified color.

        Args:
            name: Name of the entity (determines color)
            size: Size (width, height) of the sprite

        Returns:
            Placeholder sprite surface
        """
        # Get the placeholder color for the entity type
        # Strip any index suffix (like '_0' from 'explosion_0')
        base_name = name.split('_')[0] if '_' in name else name
        color = self.placeholder_colors.get(base_name, (200, 200, 200))  # Default to gray

        # Create a surface with alpha channel
        sprite = pygame.Surface(size, pygame.SRCALPHA)

        if base_name == "player":
            # Triangle shape for player
            width, height = size
            points = [
                (width // 2, 0),           # Top point
                (0, height),               # Bottom left
                (width, height)            # Bottom right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif base_name == "enemy":
            # Pentagon shape for enemy
            width, height = size
            points = [
                (width // 2, 0),             # Top point
                (0, height // 2),            # Middle left
                (width // 4, height),        # Bottom left
                (3 * width // 4, height),    # Bottom right
                (width, height // 2)         # Middle right
            ]
            pygame.draw.polygon(sprite, color, points)

        elif base_name == "missile":
            # Elongated pentagon for missile
            width, height = size
            points = [
                (width // 2, 0),             # Top point
                (width // 4, height // 4),   # Upper left
                (0, height),                 # Bottom left
                (width, height),             # Bottom right
                (3 * width // 4, height // 4)  # Upper right
            ]
            pygame.draw.polygon(sprite, color, points)
            
        elif base_name == "wall_h":
            # Horizontal wall - rectangle
            pygame.draw.rect(sprite, color, pygame.Rect(0, 0, size[0], size[1]))
            # Add edge lines for visual detail
            pygame.draw.rect(sprite, (color[0] // 2, color[1] // 2, color[2] // 2), 
                            pygame.Rect(0, 0, size[0], size[1]), 2)
            
        elif base_name == "wall_v":
            # Vertical wall - rectangle
            pygame.draw.rect(sprite, color, pygame.Rect(0, 0, size[0], size[1]))
            # Add edge lines for visual detail
            pygame.draw.rect(sprite, (color[0] // 2, color[1] // 2, color[2] // 2), 
                            pygame.Rect(0, 0, size[0], size[1]), 2)

        else:
            # Simple rectangle with slight transparency for other entities
            pygame.draw.rect(sprite, (*color, 220), pygame.Rect(0, 0, size[0], size[1]))

        return sprite

    def render(
        self,
        screen: pygame.Surface,
        entity_type: str,
        position: Union[Position, Tuple[float, float]],
        size: Tuple[int, int],
        rotation: float = 0
    ) -> None:
        """
        Render a sprite to the screen.

        Args:
            screen: Pygame surface to render to
            entity_type: Type of entity to render
            position: Position (x, y) coordinates
            size: Size (width, height) of the sprite
            rotation: Rotation angle in degrees
        """
        # Get position as (x, y) tuple
        if isinstance(position, dict):
            pos = (position["x"], position["y"])
        else:
            pos = position

        # Get the sprite surface
        sprite = self.load_sprite(entity_type, size)

        # Apply rotation if needed
        if rotation != 0:
            sprite = pygame.transform.rotate(sprite, rotation)

        # Blit to screen
        screen.blit(sprite, pos)

    def load_animation(
        self,
        name: str,
        size: Tuple[int, int],
        frames: int = 4
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
        # Check if animation is already cached
        animation_key = f"{name}_{frames}_{size[0]}x{size[1]}"
        if animation_key in self.animations:
            return self.animations[animation_key]

        # Try to load animation frames from files
        animation_frames = []
        subdir = self.sprite_subdirs.get(name, "")
        
        for i in range(frames):
            # Check both new and old paths
            frame_paths = []
            
            # New directory structure
            if subdir:
                frame_paths.append(os.path.join(self.sprites_dir, subdir, f"{i}.png"))
                frame_paths.append(os.path.join(self.sprites_dir, subdir, f"{name}_{i}.png"))
            
            # Old/legacy path
            frame_paths.append(os.path.join(self.sprites_dir, f"{name}_{i}.png"))
            
            frame_loaded = False
            for frame_path in frame_paths:
                if os.path.exists(frame_path):
                    try:
                        # Load and scale the frame
                        frame = pygame.image.load(frame_path).convert_alpha()
                        frame = pygame.transform.scale(frame, size)
                        animation_frames.append(frame)
                        frame_loaded = True
                        break
                    except pygame.error:
                        continue
            
            # If no frame was loaded, use placeholder
            if not frame_loaded:
                animation_frames.append(self._create_animation_placeholder(name, size, i, frames))

        self.animations[animation_key] = animation_frames
        return animation_frames

    def _create_animation_placeholder(
        self,
        name: str,
        size: Tuple[int, int],
        frame_index: int,
        total_frames: int
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
        scaled_sprite = pygame.transform.scale(base_sprite, (scaled_w, scaled_h))
        frame.blit(scaled_sprite, (offset_x, offset_y))

        return frame
