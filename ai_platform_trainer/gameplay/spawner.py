"""Places the player and the enemy at the start of a game, by the arena's reset rule."""
import logging
import random

from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.sim import random_spawn, spawn_away


def spawn_entities(game):
    """Spawn the player anywhere in the world, and the enemy away from the player."""
    if not game.player or not game.enemy:
        logging.error("Entities not initialized properly.")
        game.running = False
        return

    config = ArenaConfig(width=game.screen_width, height=game.screen_height)
    rng = random.Random()
    player_x, player_y = random_spawn(config, rng)
    enemy_x, enemy_y = spawn_away(config, rng, (player_x, player_y))

    game.player.position["x"], game.player.position["y"] = player_x, player_y
    game.enemy.pos["x"], game.enemy.pos["y"] = enemy_x, enemy_y

    logging.info(
        f"Spawned player at ({player_x:.0f}, {player_y:.0f}) "
        f"and enemy at ({enemy_x:.0f}, {enemy_y:.0f})."
    )
