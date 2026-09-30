"""Plain-data snapshot of one arena frame, shared by the sim and the game."""
from dataclasses import dataclass, field
from typing import List

from ai_platform_trainer.arena.config import ArenaConfig


@dataclass
class Body:
    """A 50x50 player or enemy. (x, y) is the top-left corner, as in pygame."""

    x: float
    y: float
    # Observed displacement over the most recent frame (wrap aware).
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class MissileState:
    """One missile. (x, y) is SmartMissile.pos; hits use a rect anchored there."""

    x: float
    y: float
    vx: float  # velocity per substep
    vy: float
    life_frames: int
    age_frames: int = 0
    exploded: bool = False
    explosion_age: int = 0
    last_target_x: float = 0.0
    last_target_y: float = 0.0

    @property
    def frames_left(self) -> int:
        return max(0, self.life_frames - self.age_frames)


@dataclass
class ArenaState:
    """Everything an agent may observe about one frame of Play mode."""

    config: ArenaConfig
    player: Body
    enemy: Body
    missiles: List[MissileState] = field(default_factory=list)
    enemy_visible: bool = True
    cooldown_left: int = 0  # frames until the player may fire again
    frame: int = 0
