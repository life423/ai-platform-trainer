# ai_platform_trainer/gameplay/collisions.py
import pygame
import logging


def handle_missile_collisions(player, enemy, respawn_callback):
    if not enemy.visible:
        return
    enemy_rect = pygame.Rect(enemy.pos["x"], enemy.pos["y"], enemy.size, enemy.size)
    for missile in player.missiles[:]:
        if missile.get_rect().colliderect(enemy_rect):
            logging.info("Missile hit the enemy.")
            player.missiles.remove(missile)
            enemy.hide()
            respawn_callback()
