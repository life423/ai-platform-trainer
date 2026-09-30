"""Bridge from live Play mode entities to ArenaState.

The in-game enemy agent snapshots the pygame entities every frame and feeds
the result to the same build_enemy_observation() it was trained with.
"""
import math
from typing import Any, Optional, Tuple

from ai_platform_trainer.arena.config import FRAME_MS, ArenaConfig, ms_to_frames
from ai_platform_trainer.arena.geometry import torus_delta
from ai_platform_trainer.arena.state import ArenaState, Body, MissileState

Vector = Tuple[float, float]


class MotionTracker:
    """Observed per-frame displacement of an entity, measured the same way the sim does."""

    def __init__(self, config: ArenaConfig) -> None:
        self.config = config
        self._last: Optional[Vector] = None

    def reset(self) -> None:
        self._last = None

    def update(self, x: float, y: float) -> Vector:
        cfg = self.config
        if self._last is None:
            velocity = (0.0, 0.0)
        else:
            velocity = (
                torus_delta(self._last[0], x, cfg.period_x),
                torus_delta(self._last[1], y, cfg.period_y),
            )
        self._last = (float(x), float(y))
        return velocity


def snapshot_from_game(
    config: ArenaConfig,
    player: Any,
    enemy_xy: Vector,
    player_velocity: Vector,
    enemy_velocity: Vector,
    now_ms: float,
    enemy_visible: bool = True,
) -> ArenaState:
    """Build an ArenaState from a PlayerPlay (and its SmartMissiles) plus the enemy."""
    missiles = [
        MissileState(
            x=float(m.pos["x"]),
            y=float(m.pos["y"]),
            vx=float(m.vx),
            vy=float(m.vy),
            life_frames=ms_to_frames(m.lifespan),
            age_frames=ms_to_frames(now_ms - m.birth_time),
            exploded=bool(getattr(m, "exploded", False)),
        )
        for m in player.missiles
    ]
    remaining_ms = player.missile_cooldown - (now_ms - player.last_missile_time)
    cooldown_left = max(0, int(math.ceil(remaining_ms / FRAME_MS - 1e-6)))
    return ArenaState(
        config=config,
        player=Body(
            float(player.position["x"]),
            float(player.position["y"]),
            float(player_velocity[0]),
            float(player_velocity[1]),
        ),
        enemy=Body(
            float(enemy_xy[0]),
            float(enemy_xy[1]),
            float(enemy_velocity[0]),
            float(enemy_velocity[1]),
        ),
        missiles=missiles,
        enemy_visible=enemy_visible,
        cooldown_left=cooldown_left,
    )
