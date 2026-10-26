"""A catch respawns the player away from the enemy; the enemy stays where it is."""
import math
import os
from types import SimpleNamespace

import pygame
import pytest

from ai_platform_trainer.ai.model_paths import ENEMY_PPO_MODEL
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.gameplay.modes.play_learning_mode import PlayLearningMode

needs_agent = pytest.mark.skipif(
    not os.path.exists(ENEMY_PPO_MODEL), reason="AI Agent model not on disk"
)


@needs_agent
def test_a_catch_respawns_the_player_and_the_enemy_stays():
    pygame.init()
    game = SimpleNamespace(
        screen_width=800,
        screen_height=600,
        is_respawning=False,
        respawn_timer=0,
        respawn_delay=1000,
        player=PlayerPlay(800, 600),
        enemy=None,
    )
    mode = PlayLearningMode(game)
    enemy = mode.learning_enemy
    enemy.set_position(300, 300)
    enemy.visible = True
    game.player.position["x"], game.player.position["y"] = 300, 300  # on top of it

    mode._check_collisions(0)

    assert enemy.visible
    assert not game.is_respawning
    assert (enemy.pos["x"], enemy.pos["y"]) == (300, 300)
    away = math.hypot(game.player.position["x"] - 300, game.player.position["y"] - 300)
    assert away >= ArenaConfig().min_spawn_distance
