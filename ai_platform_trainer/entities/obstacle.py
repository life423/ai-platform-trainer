"""
Obstacle entity for the AI Platform Trainer.

This module defines the Obstacle class for wall obstacles
that can block player and enemy movement.
"""
import pygame
from typing import Tuple, List, Optional


class Obstacle:
    """
    Represents a wall obstacle in the game.
    
    Obstacles can be horizontal or vertical walls that block
    the movement of players, enemies, and missiles.
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        orientation: str = "horizontal",
        width: int = 128,
        height: int = 32
    ):
        """
        Initialize an obstacle.
        
        Args:
            x: X-coordinate of the top-left corner
            y: Y-coordinate of the top-left corner
            orientation: "horizontal" or "vertical"
            width: Width of the obstacle
            height: Height of the obstacle
        """
        self.position = {"x": x, "y": y}
        self.orientation = orientation
        
        # Set dimensions based on orientation if not explicitly provided
        if orientation == "vertical" and width == 128 and height == 32:
            # Default vertical wall is taller than wide
            self.width = 32
            self.height = 128
        else:
            self.width = width
            self.height = height
            
        # Used by the renderer to determine which sprite to use
        self.sprite_name = f"wall_{orientation[0]}"
        
        # Flag for visibility
        self.visible = True
        
        # Create a rect for collision detection
        self.rect = pygame.Rect(x, y, self.width, self.height)
        
    def update(self):
        """
        Update the obstacle state.
        
        For static obstacles, this simply updates the collision rect
        to match the current position.
        """
        # Update the collision rectangle
        self.rect.x = self.position["x"]
        self.rect.y = self.position["y"]
        
    def set_position(self, x: int, y: int):
        """
        Set the position of the obstacle.
        
        Args:
            x: New X-coordinate
            y: New Y-coordinate
        """
        self.position["x"] = x
        self.position["y"] = y
        self.rect.x = x
        self.rect.y = y
        
    def collides_with(self, other_rect: pygame.Rect) -> bool:
        """
        Check if this obstacle collides with another entity's rectangle.
        
        Args:
            other_rect: Pygame.Rect of the other entity
            
        Returns:
            True if collision detected, False otherwise
        """
        return self.rect.colliderect(other_rect)
    
    def collides_with_point(self, x: int, y: int) -> bool:
        """
        Check if a point is inside this obstacle.
        
        Args:
            x: X-coordinate of the point
            y: Y-coordinate of the point
            
        Returns:
            True if the point is inside the obstacle, False otherwise
        """
        return self.rect.collidepoint(x, y)
    
    def hide(self):
        """Make the obstacle invisible."""
        self.visible = False
        
    def show(self):
        """Make the obstacle visible."""
        self.visible = True


class ObstacleManager:
    """
    Manages a collection of obstacles in the game.
    
    This class handles initializing, updating, and collision detection
    for multiple obstacles.
    """
    
    def __init__(self):
        """Initialize an empty obstacle collection."""
        self.obstacles: List[Obstacle] = []
        
    def add_obstacle(self, obstacle: Obstacle):
        """
        Add an obstacle to the collection.
        
        Args:
            obstacle: Obstacle to add
        """
        self.obstacles.append(obstacle)
        
    def add_horizontal_wall(self, x: int, y: int, width: int = 128):
        """
        Add a horizontal wall obstacle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            width: Width of the wall (default 128)
        """
        obstacle = Obstacle(x, y, "horizontal", width, 32)
        self.add_obstacle(obstacle)
        
    def add_vertical_wall(self, x: int, y: int, height: int = 128):
        """
        Add a vertical wall obstacle.
        
        Args:
            x: X-coordinate
            y: Y-coordinate
            height: Height of the wall (default 128)
        """
        obstacle = Obstacle(x, y, "vertical", 32, height)
        self.add_obstacle(obstacle)
        
    def clear(self):
        """Remove all obstacles."""
        self.obstacles.clear()
        
    def update(self):
        """Update all obstacles."""
        for obstacle in self.obstacles:
            obstacle.update()
            
    def check_collision(self, entity_rect: pygame.Rect) -> Tuple[bool, Optional[Obstacle]]:
        """
        Check if an entity collides with any obstacle.
        
        Args:
            entity_rect: Pygame.Rect of the entity to check
            
        Returns:
            Tuple of (collision_detected, first_colliding_obstacle)
            If no collision, returns (False, None)
        """
        for obstacle in self.obstacles:
            if obstacle.visible and obstacle.collides_with(entity_rect):
                return True, obstacle
        return False, None
    
    def get_visible_obstacles(self) -> List[Obstacle]:
        """
        Get the list of visible obstacles.
        
        Returns:
            List of visible obstacles
        """
        return [obs for obs in self.obstacles if obs.visible]
