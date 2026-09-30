"""Geometry helpers that reproduce Play mode exactly (see the parity tests)."""
import math
from typing import Tuple

# The 8 keyboard directions indexed by the octant of atan2(dy, dx), in screen
# coordinates where y grows downward.
OCTANT_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)


def wrap_coordinate(value: float, limit: float, size: float) -> float:
    """Play mode wrap: beyond -size jumps to limit, beyond limit to -size."""
    if value < -size:
        return float(limit)
    if value > limit:
        return float(-size)
    return value


def torus_delta(source: float, target: float, period: float) -> float:
    """Shortest signed offset from source to target on a wrapping axis."""
    delta = math.fmod(target - source, period)
    if delta > period / 2.0:
        delta -= period
    elif delta < -period / 2.0:
        delta += period
    return delta


def rects_overlap(
    ax: float, ay: float, asize: int, bx: float, by: float, bsize: int
) -> bool:
    """pygame.Rect(x, y, size, size).colliderect for float positions.

    pygame truncates float coordinates toward zero, and rectangles whose
    edges merely touch do not collide.
    """
    ax_i, ay_i, bx_i, by_i = int(ax), int(ay), int(bx), int(by)
    return (
        ax_i < bx_i + bsize
        and bx_i < ax_i + asize
        and ay_i < by_i + bsize
        and by_i < ay_i + asize
    )


def quantize_direction(dx: float, dy: float) -> Tuple[int, int]:
    """Snap a direction to the nearest keyboard direction, (0, 0) if none."""
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0, 0
    octant = int(round(math.atan2(dy, dx) / (math.pi / 4.0))) % 8
    return OCTANT_DIRECTIONS[octant]


def normalize_angle(angle: float) -> float:
    """Wrap an angle in radians into [-pi, pi], the way SmartMissile does."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def wrapped_move(
    x: float, y: float, dx: float, dy: float, width: int, height: int, size: int
) -> Tuple[float, float, float, float]:
    """Move a body, wrap it, and return (x, y, vx, vy), where v is the wrap-aware
    displacement an observer sees. The sim and the in-game agent both move this way."""
    new_x = wrap_coordinate(x + dx, width, size)
    new_y = wrap_coordinate(y + dy, height, size)
    return (
        new_x,
        new_y,
        torus_delta(x, new_x, width + size),
        torus_delta(y, new_y, height + size),
    )
