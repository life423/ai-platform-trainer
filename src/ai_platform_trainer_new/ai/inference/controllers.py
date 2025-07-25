"""
Clean AI Controllers

This module provides AI controllers for game entities.
"""
import logging
from typing import Tuple

from ..models.enemy_model import EnemyAI
from ..models.missile_model import MissileAI


class AIControllerManager:
    """Manages AI controllers for different entities."""
    
    def __init__(self):
        self.enemy_ai = EnemyAI()
        self.missile_ai = MissileAI()
        logging.info("AI Controller Manager initialized")
    
    def get_enemy_movement(self, player_x: float, player_y: float, 
                          enemy_x: float, enemy_y: float) -> Tuple[float, float]:
        """
        Get AI-controlled enemy movement.
        
        Args:
            player_x: Player X position
            player_y: Player Y position
            enemy_x: Enemy X position
            enemy_y: Enemy Y position
            
        Returns:
            Tuple of (move_x, move_y) values
        """
        # Calculate distance
        distance = ((player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2) ** 0.5
        
        # Get AI prediction
        move_x, move_y = self.enemy_ai.predict_movement(
            player_x, player_y, enemy_x, enemy_y, distance
        )
        
        return move_x, move_y
    
    def get_missile_guidance(self, missile_x: float, missile_y: float,
                           missile_vx: float, missile_vy: float,
                           target_x: float, target_y: float,
                           player_x: float, player_y: float,
                           step: int) -> Tuple[float, float]:
        """
        Get AI-controlled missile guidance.
        
        Args:
            missile_x: Missile X position
            missile_y: Missile Y position
            missile_vx: Missile X velocity
            missile_vy: Missile Y velocity
            target_x: Target X position
            target_y: Target Y position
            player_x: Player X position
            player_y: Player Y position
            step: Current game step
            
        Returns:
            Tuple of (adjust_x, adjust_y) values
        """
        adjust_x, adjust_y = self.missile_ai.predict_guidance(
            missile_x, missile_y, missile_vx, missile_vy,
            target_x, target_y, player_x, player_y, step
        )
        
        return adjust_x, adjust_y


# Global AI controller instance
ai_controller = AIControllerManager()