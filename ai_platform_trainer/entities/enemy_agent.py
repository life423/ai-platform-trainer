"""The arena-trained enemy agent, running in Play mode.

It sees the live game through exactly the encoder it was trained with:
pygame entities -> arena.game_bridge.snapshot_from_game() -> SB3Enemy
(build_enemy_observation, PPO policy, shared action mapping and decision
rate) -> geometry.wrapped_move(). The parity tests in tests/unit/arena pin
each step of that path to the headless training arena.
"""
import os
from typing import Any, Callable, List, Optional

from ai_platform_trainer.ai.model_paths import ENEMY_PPO_MODEL
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.enemy_policies import SB3Enemy
from ai_platform_trainer.arena.game_bridge import MotionTracker, snapshot_from_game
from ai_platform_trainer.arena.geometry import wrapped_move
from ai_platform_trainer.entities.enemy_play import (
    EnemyPlay,
    create_enemy_play,
    is_trained_enemy_available,
)


def is_enemy_agent_available(model_path: str = ENEMY_PPO_MODEL) -> bool:
    """Whether train-enemy has deployed a policy yet."""
    return os.path.exists(model_path)


def trained_enemy_description() -> Optional[str]:
    """What the Trained AI menu option will run, or None if nothing is trained."""
    if is_enemy_agent_available():
        return "PPO"
    if is_trained_enemy_available():
        return "Neural Network"
    return None


def create_trained_enemy(
    screen_width: int, screen_height: int, player_provider: Callable[[], Any]
) -> Optional[EnemyPlay]:
    """The best trained enemy available: the arena PPO agent, else the legacy network."""
    if is_enemy_agent_available():
        return ArenaEnemyAgent(screen_width, screen_height, player_provider)
    if is_trained_enemy_available():
        return create_enemy_play(screen_width, screen_height)
    return None


class ArenaEnemyAgent(EnemyPlay):
    """EnemyPlay (drawing, fade-in, respawn hooks) driven by the PPO policy."""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        player_provider: Callable[[], Any],
        model_path: str = ENEMY_PPO_MODEL,
    ) -> None:
        super().__init__(screen_width, screen_height, model=None)
        self.controller = SB3Enemy.load(model_path)
        self.config = ArenaConfig(width=screen_width, height=screen_height)
        self._player_provider = player_provider
        self._player_motion = MotionTracker(self.config)
        self._last_move = (0.0, 0.0)

    def update_movement(
        self,
        player_x: float,
        player_y: float,
        player_speed: float,
        current_time: int,
        missiles: Optional[List] = None,
    ) -> None:
        # Track the player every frame, even while this enemy is respawning.
        player_velocity = self._player_motion.update(player_x, player_y)
        player = self._player_provider()
        if not self.visible or player is None:
            self._last_move = (0.0, 0.0)
            return
        state = snapshot_from_game(
            self.config,
            player,
            (self.pos["x"], self.pos["y"]),
            player_velocity,
            self._last_move,
            current_time,
            self.visible,
        )
        dx, dy = self.controller.act(state)
        cfg = self.config
        self.pos["x"], self.pos["y"], vx, vy = wrapped_move(
            self.pos["x"], self.pos["y"], dx, dy, cfg.width, cfg.height, cfg.body_size
        )
        self._last_move = (vx, vy)
        if self.fading_in:
            self.update_fade_in(current_time)

    def set_position(self, x: float, y: float) -> None:
        super().set_position(x, y)
        self._last_move = (0.0, 0.0)
        self.controller.reset()

    def get_difficulty_level(self) -> float:
        return 1.0

    def get_learning_stats(self) -> dict:
        stats = super().get_learning_stats()
        stats["stage"] = "PPO agent"
        return stats
