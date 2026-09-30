"""Canonical rules of the Pixel Pursuit arena.

Every value here mirrors how Play mode behaves today, and the parity tests in
tests/unit/arena check the pygame entities against them. Training, evaluation
and gameplay all read this one definition, so a model can never be trained on
a different game than the one it is deployed into.
"""
from dataclasses import dataclass, replace
from typing import Tuple

FPS = 60
FRAME_MS = 1000.0 / FPS

# Fullscreen size of the machine this project is developed on; evaluation
# defaults to it, training randomizes around it.
DEFAULT_SCREEN = (1470, 956)


def ms_to_frames(ms: float) -> int:
    """Convert a millisecond game timer to whole frames at the fixed 60 FPS."""
    return int(round(ms / FRAME_MS))


@dataclass(frozen=True)
class ArenaConfig:
    """Geometry, speeds and timers for one arena.

    Speeds are pixels per frame. The player moves with the keyboard, so it
    covers player_speed on each axis (about 7.07 px per frame on diagonals).
    The enemy is 50 percent faster, where the original Adaptive AI hunting
    stages begin: at 1.2x or less, a player who kites while firing can never
    be caught. Every enemy in the benchmark gets the same cap, so comparisons
    stay fair.
    """

    width: int = DEFAULT_SCREEN[0]
    height: int = DEFAULT_SCREEN[1]
    body_size: int = 50
    player_speed: float = 5.0
    enemy_speed: float = 7.5
    missile_size: int = 10
    # One steer-and-move per frame at 5 px, as PlayerPlay.shoot_missile and
    # the Adaptive AI comments intend. Play mode used to update each missile
    # twice per frame by accident (10 px per frame, doubled turning); set
    # missile_substeps = 2 to reproduce that old feel.
    missile_speed: float = 5.0
    missile_substeps: int = 1
    missile_turn_deg: float = 12.0
    missile_prediction: float = 0.5
    missile_aim_jitter: float = 0.1
    missile_life_ms: Tuple[int, int] = (2500, 3000)
    missile_cooldown_frames: int = 30
    max_missiles: int = 3
    explosion_frames: int = 18
    respawn_frames: int = 60
    min_spawn_distance: float = 200.0

    @property
    def period_x(self) -> float:
        """Wrap period of a body on the x axis (see geometry.wrap_coordinate)."""
        return float(self.width + self.body_size)

    @property
    def period_y(self) -> float:
        return float(self.height + self.body_size)

    @property
    def max_missile_life_frames(self) -> int:
        return ms_to_frames(self.missile_life_ms[1])

    def with_size(self, width: int, height: int) -> "ArenaConfig":
        """The same rules on a different screen size."""
        return replace(self, width=int(width), height=int(height))
