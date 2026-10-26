# file: ai_platform_trainer/gameplay/spawner.py
import logging

from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.spawn_utils import find_valid_spawn_position


def spawn_entities(game):
    """
    Spawn the player and enemy at random positions at the start of the game,
    ensuring they maintain minimum distance constraints.
    """
    if not game.player or not game.enemy:
        logging.error("Entities not initialized properly.")
        game.running = False
        return

    player_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.player.size,
        margin=config.WALL_MARGIN,
        min_dist=0,
        other_pos=None,
    )

    enemy_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.enemy.size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=(game.player.position["x"], game.player.position["y"]),
    )

    game.player.position["x"], game.player.position["y"] = player_pos
    game.enemy.pos["x"], game.enemy.pos["y"] = enemy_pos

    logging.info(f"Spawned player at {player_pos} and enemy at {enemy_pos}.")
