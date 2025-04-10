"""
Generate placeholder sprite images for AI Platform Trainer.

This script creates basic sprite images for the game entities using pygame
and saves them to the assets/sprites directory.
"""
import os
import pygame
import math
import sys

# Initialize pygame
pygame.init()

# Ensure output directory exists
SPRITES_DIR = "assets/sprites"
os.makedirs(SPRITES_DIR, exist_ok=True)

# Set up sprite sizes and parameters
SPRITE_SIZES = {
    "player": (64, 64),
    "enemy": (64, 64),
    "missile": (32, 48),
    "explosion": (64, 64),
    "wall_h": (128, 32),      # Horizontal wall
    "wall_v": (32, 128)       # Vertical wall
}

# Define color palettes
COLORS = {
    "player": {
        "body": (30, 144, 255),     # Dodger blue
        "highlight": (135, 206, 250),  # Light sky blue
        "shadow": (0, 0, 139)       # Dark blue
    },
    "enemy": {
        "body": (220, 20, 60),      # Crimson red
        "highlight": (255, 99, 71),  # Tomato red
        "shadow": (139, 0, 0)       # Dark red
    },
    "missile": {
        "body": (255, 215, 0),      # Gold
        "highlight": (255, 255, 0),  # Yellow
        "trail": (255, 69, 0)       # Red-orange
    },
    "explosion": {
        "outer": (255, 69, 0),      # Red-orange
        "middle": (255, 165, 0),    # Orange
        "inner": (255, 255, 0)      # Yellow
    },
    "wall": {
        "body": (128, 128, 128),    # Gray
        "edge": (64, 64, 64),       # Dark gray
        "highlight": (192, 192, 192)  # Light gray
    }
}


def create_player_sprite(size):
    """
    Create a player sprite - a spaceship/triangle with details.

    Args:
        size: Tuple of (width, height) for the sprite

    Returns:
        Pygame surface with the sprite
    """
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)

    # Define ship shape (triangular spaceship)
    points = [
        (width // 2, 0),                # Nose
        (width // 6, height * 3 // 4),  # Left wing
        (width // 3, height * 2 // 3),  # Left body
        (width // 3, height),           # Left back
        (width * 2 // 3, height),       # Right back
        (width * 2 // 3, height * 2 // 3), # Right body
        (width * 5 // 6, height * 3 // 4)  # Right wing
    ]

    # Draw ship body
    pygame.draw.polygon(surface, COLORS["player"]["body"], points)

    # Add cockpit (circle)
    cockpit_radius = width // 6
    pygame.draw.circle(
        surface,
        COLORS["player"]["highlight"],
        (width // 2, height // 3),
        cockpit_radius
    )

    # Add engine glow
    engine_points = [
        (width // 3 + 2, height - 5),
        (width * 2 // 3 - 2, height - 5),
        (width // 2, height + 10)
    ]
    pygame.draw.polygon(surface, COLORS["player"]["highlight"], engine_points)

    # Add details/contours
    pygame.draw.lines(surface, COLORS["player"]["shadow"], True, points, 2)

    return surface


def create_enemy_sprite(size):
    """
    Create an enemy sprite - a pentagon/spaceship with details.

    Args:
        size: Tuple of (width, height) for the sprite

    Returns:
        Pygame surface with the sprite
    """
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)

    # Define ship shape (pentagon-like alien ship)
    points = [
        (width // 2, 0),                # Top point
        (0, height // 2),               # Left point
        (width // 4, height),           # Bottom left
        (width * 3 // 4, height),       # Bottom right
        (width, height // 2)            # Right point
    ]

    # Draw ship body
    pygame.draw.polygon(surface, COLORS["enemy"]["body"], points)

    # Add center dome
    dome_radius = width // 4
    pygame.draw.circle(
        surface,
        COLORS["enemy"]["highlight"],
        (width // 2, height // 2),
        dome_radius
    )

    # Add "eye" in center
    eye_radius = width // 10
    pygame.draw.circle(
        surface,
        COLORS["enemy"]["shadow"],
        (width // 2, height // 2),
        eye_radius
    )

    # Add details/contours
    pygame.draw.lines(surface, COLORS["enemy"]["shadow"], True, points, 2)

    # Add some small circles for decoration
    for i in range(3):
        angle = i * 2 * math.pi / 3
        x = width // 2 + int(math.cos(angle) * width // 3)
        y = height // 2 + int(math.sin(angle) * height // 3)
        pygame.draw.circle(surface, COLORS["enemy"]["shadow"], (x, y), width // 12)

    return surface


def create_missile_sprite(size):
    """
    Create a missile sprite - an elongated shape with trail.

    Args:
        size: Tuple of (width, height) for the sprite

    Returns:
        Pygame surface with the sprite
    """
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)

    # Missile body (elongated hexagon)
    body_width = width // 2
    body_height = height * 2 // 3

    points = [
        (width // 2, 0),                      # Nose
        (width // 2 - body_width // 2, body_height // 4),  # Top left
        (width // 2 - body_width // 2, body_height),       # Bottom left
        (width // 2 + body_width // 2, body_height),       # Bottom right
        (width // 2 + body_width // 2, body_height // 4)   # Top right
    ]

    # Draw missile body
    pygame.draw.polygon(surface, COLORS["missile"]["body"], points)

    # Draw missile trail/flame
    trail_points = [
        (width // 2 - body_width // 4, body_height),
        (width // 2 + body_width // 4, body_height),
        (width // 2, height)
    ]
    pygame.draw.polygon(surface, COLORS["missile"]["trail"], trail_points)

    # Add highlight along nose
    pygame.draw.line(
        surface,
        COLORS["missile"]["highlight"],
        (width // 2, 0),
        (width // 2, body_height // 2),
        2
    )

    return surface


def create_explosion_sprite(size, frame_index, total_frames=4):
    """
    Create an explosion sprite frame.

    Args:
        size: Tuple of (width, height) for the sprite
        frame_index: Current frame of the explosion animation
        total_frames: Total number of frames in the explosion

    Returns:
        Pygame surface with the sprite
    """
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)

    # Calculate expansion factor based on frame
    progress = frame_index / (total_frames - 1)

    # Draw expanding circles for explosion effect
    radius_outer = int(width * 0.4 * (0.3 + progress * 0.7))
    radius_middle = int(width * 0.3 * (0.3 + progress * 0.7))
    radius_inner = int(width * 0.15 * (0.3 + progress * 0.7))

    # Transparency decreases with time
    alpha_outer = int(255 * (1 - progress * 0.7))
    alpha_middle = int(255 * (1 - progress * 0.5))
    alpha_inner = int(255 * (1 - progress * 0.3))

    # Draw explosion circles
    pygame.draw.circle(
        surface,
        (*COLORS["explosion"]["outer"], alpha_outer),
        (width // 2, height // 2),
        radius_outer
    )

    pygame.draw.circle(
        surface,
        (*COLORS["explosion"]["middle"], alpha_middle),
        (width // 2, height // 2),
        radius_middle
    )

    pygame.draw.circle(
        surface,
        (*COLORS["explosion"]["inner"], alpha_inner),
        (width // 2, height // 2),
        radius_inner
    )

    # Add some "spark" lines for later frames
    if frame_index > 0:
        for i in range(8):
            angle = i * math.pi / 4 + progress * math.pi / 8
            length = width * 0.5 * progress
            start_x = width // 2 + int(math.cos(angle) * radius_middle * 0.8)
            start_y = height // 2 + int(math.sin(angle) * radius_middle * 0.8)
            end_x = width // 2 + int(math.cos(angle) * (radius_middle * 0.8 + length))
            end_y = height // 2 + int(math.sin(angle) * (radius_middle * 0.8 + length))

            pygame.draw.line(
                surface,
                (*COLORS["explosion"]["outer"], alpha_outer),
                (start_x, start_y),
                (end_x, end_y),
                2
            )

    return surface


def create_wall_sprite(size, orientation="horizontal"):
    """
    Create a wall obstacle sprite.

    Args:
        size: Tuple of (width, height) for the sprite
        orientation: "horizontal" or "vertical" wall

    Returns:
        Pygame surface with the sprite
    """
    width, height = size
    surface = pygame.Surface(size, pygame.SRCALPHA)
    
    # Fill with base color
    pygame.draw.rect(
        surface,
        COLORS["wall"]["body"],
        pygame.Rect(0, 0, width, height)
    )
    
    # Add edge
    edge_width = max(2, min(width, height) // 8)
    pygame.draw.rect(
        surface,
        COLORS["wall"]["edge"],
        pygame.Rect(0, 0, width, height),
        edge_width
    )
    
    # Add some detail - either horizontal or vertical lines
    if orientation == "horizontal":
        # Horizontal wall gets vertical dividers
        for x in range(width // 16, width, width // 4):
            pygame.draw.line(
                surface,
                COLORS["wall"]["edge"],
                (x, 0),
                (x, height),
                max(1, edge_width // 2)
            )
    else:
        # Vertical wall gets horizontal dividers
        for y in range(height // 16, height, height // 4):
            pygame.draw.line(
                surface,
                COLORS["wall"]["edge"],
                (0, y),
                (width, y),
                max(1, edge_width // 2)
            )
    
    # Add highlight on one edge for 3D effect
    if orientation == "horizontal":
        # Top edge highlight for horizontal wall
        pygame.draw.line(
            surface,
            COLORS["wall"]["highlight"],
            (edge_width // 2, edge_width // 2),
            (width - edge_width // 2, edge_width // 2),
            max(1, edge_width // 2)
        )
    else:
        # Left edge highlight for vertical wall
        pygame.draw.line(
            surface,
            COLORS["wall"]["highlight"],
            (edge_width // 2, edge_width // 2),
            (edge_width // 2, height - edge_width // 2),
            max(1, edge_width // 2)
        )
        
    return surface



def save_sprite(surface, name):
    """
    Save a sprite surface to a file.

    Args:
        surface: Pygame surface to save
        name: Filename (without extension)
    """
    # Determine the appropriate subdirectory based on sprite name
    if name.startswith("player"):
        subdir = "player"
    elif name.startswith("enemy"):
        subdir = "enemies"
    elif name.startswith("missile"):
        subdir = "weapons"
    elif name.startswith("explosion"):
        subdir = "effects"
    elif name.startswith("wall"):
        subdir = "obstacles"
    else:
        subdir = ""  # Root sprites directory
        
    # Create the target directory if it doesn't exist
    target_dir = os.path.join(SPRITES_DIR, subdir)
    os.makedirs(target_dir, exist_ok=True)
    
    # Create the full path and save
    filename = name.split('_')[-1] if '_' in name else name
    path = os.path.join(target_dir, f"{filename}.png")
    pygame.image.save(surface, path)
    print(f"Saved sprite: {path}")


def generate_all_sprites():
    """Generate all sprite images for the game."""
    print("Generating sprites...")

    # Generate player sprite
    player = create_player_sprite(SPRITE_SIZES["player"])
    save_sprite(player, "player")

    # Generate enemy sprite
    enemy = create_enemy_sprite(SPRITE_SIZES["enemy"])
    save_sprite(enemy, "enemy")

    # Generate missile sprite
    missile = create_missile_sprite(SPRITE_SIZES["missile"])
    save_sprite(missile, "missile")

    # Generate explosion animation frames
    for i in range(4):
        explosion = create_explosion_sprite(SPRITE_SIZES["explosion"], i, 4)
        save_sprite(explosion, f"explosion_{i}")
        
    # Generate wall sprites (horizontal and vertical)
    wall_h = create_wall_sprite(SPRITE_SIZES["wall_h"], "horizontal")
    save_sprite(wall_h, "wall_h")
    
    wall_v = create_wall_sprite(SPRITE_SIZES["wall_v"], "vertical")
    save_sprite(wall_v, "wall_v")

    print("All sprites generated successfully!")


if __name__ == "__main__":
    generate_all_sprites()
    pygame.quit()
    sys.exit()
