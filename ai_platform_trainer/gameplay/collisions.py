# ai_platform_trainer/gameplay/collisions.py
import pygame
import logging


def handle_missile_collisions(player, enemy, respawn_callback):
    """
    Check for collisions between player missiles and the enemy.
    
    Args:
        player: The player entity with missiles
        enemy: The enemy entity
        respawn_callback: Function to call when enemy is hit
    """
    if not enemy.visible:
        return
        
    # Ensure enemy position is valid
    if not isinstance(enemy.pos, dict) or "x" not in enemy.pos or "y" not in enemy.pos:
        logging.error(f"Invalid enemy position format: {enemy.pos}")
        return
        
    try:
        enemy_rect = pygame.Rect(
            int(enemy.pos["x"]), 
            int(enemy.pos["y"]), 
            enemy.size, 
            enemy.size
        )
        
        for missile in player.missiles[:]:
            # Create missile rect manually since get_rect() might not work
            missile_rect = pygame.Rect(
                missile.pos["x"] - 5,  # Missile size/2
                missile.pos["y"] - 5,
                10, 10  # Missile size
            )
            if missile_rect.colliderect(enemy_rect):
                logging.info("Missile hit the enemy.")
                player.missiles.remove(missile)
                enemy.hide()
                respawn_callback()
                
    except (TypeError, ValueError) as e:
        logging.error(f"Error in missile collision detection: {e}")
        return