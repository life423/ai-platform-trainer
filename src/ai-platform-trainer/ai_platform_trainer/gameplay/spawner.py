# file: ai_platform_trainer/gameplay/spawner.py
import logging
from ai_platform_trainer.gameplay.spawn_utils import (
    find_valid_spawn_position,
    find_enemy_spawn_position,
)
from ai_platform_trainer.gameplay.config import config


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


def respawn_enemy_at_position(game):
    """
    Find a valid position to respawn the enemy, respecting minimum distance
    from the player, and place the enemy there without showing it yet.
    This helper is used by other respawn functions.
    """
    new_pos = find_valid_spawn_position(
        screen_width=game.screen_width,
        screen_height=game.screen_height,
        entity_size=game.enemy.size,
        margin=config.WALL_MARGIN,
        min_dist=config.MIN_DISTANCE,
        other_pos=(game.player.position["x"], game.player.position["y"]),
    )
    game.enemy.set_position(new_pos[0], new_pos[1])
    return new_pos


def respawn_enemy(game):
    """
    Respawn the enemy at a valid position and show it immediately, without fade-in.
    """
    if not game.enemy or not game.player:
        return

    new_pos = respawn_enemy_at_position(game)
    game.enemy.show()
    game.is_respawning = False
    logging.info(f"Enemy respawned at {new_pos}.")


def respawn_enemy_with_fade_in(game, current_time):
    """
    Respawn the enemy at a valid position and show it with a fade-in effect.
    """
    if not game.enemy or not game.player:
        return

    new_pos = respawn_enemy_at_position(game)
    # Show the enemy with fade-in behavior
    game.enemy.show(current_time)
    # If the double show call is intentional for the fade effect, keep it:
    game.enemy.show(current_time)

    game.is_respawning = False
    logging.info(f"Enemy respawned at {new_pos} with fade-in.")