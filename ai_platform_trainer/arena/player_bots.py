"""Scripted stand-ins for the human player.

Each bot presses the same controls a person does (8-way movement plus fire)
and has human limits: it perceives the enemy 100-300 ms late, keeps a key
choice for a moment instead of re-deciding every frame, and now and then loses
focus. Its personal style (kiting, strafing, doubling back after each shot,
...) is sampled per round. A varied, human-like population keeps the enemy
from overfitting to one opponent, and the styles double as labeled behavior
for the opponent-modeling phase.
"""
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple, Type

from ai_platform_trainer.arena.geometry import (
    OCTANT_DIRECTIONS,
    quantize_direction,
    torus_delta,
)
from ai_platform_trainer.arena.state import ArenaState

Offset = Tuple[float, float]
Move = Tuple[int, int]
STOP: Move = (0, 0)


@dataclass(frozen=True)
class PlayerAction:
    """One frame of player input: dx and dy in {-1, 0, 1} plus the fire key."""

    dx: int = 0
    dy: int = 0
    shoot: bool = False

    @property
    def move_index(self) -> int:
        """Class id 0-8 of the movement keys (4 means standing still)."""
        return (self.dy + 1) * 3 + (self.dx + 1)


class PlayerBot:
    """Base bot: human reaction time, key holding, lapses and a firing habit."""

    style = "base"

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.reaction_frames = rng.randint(6, 18)  # 100-300 ms
        self.hold_frames = rng.randint(4, 10)  # a key choice is kept this long
        self.lapse_chance = rng.uniform(0.002, 0.01)  # per frame
        self.key_noise = rng.uniform(0.0, 0.05)
        self.fire_chance = rng.uniform(0.1, 0.9)
        self.fire_range = rng.uniform(250.0, 1000.0)
        self._seen: Deque[Offset] = deque(maxlen=self.reaction_frames + 1)
        self._move: Move = STOP
        self._next_decision = 0
        self._wander_move: Move = STOP
        self._wander_until = 0
        self.setup()

    def setup(self) -> None:
        """Sample style-specific tendencies (overridden by subclasses)."""

    def act(self, state: ArenaState) -> PlayerAction:
        cfg = state.config
        self._seen.append(
            (
                torus_delta(state.player.x, state.enemy.x, cfg.period_x),
                torus_delta(state.player.y, state.enemy.y, cfg.period_y),
            )
        )
        to_enemy = self._seen[0]  # what the player saw reaction_frames ago
        distance = math.hypot(to_enemy[0], to_enemy[1])
        if state.frame >= self._next_decision:
            self._decide(state, to_enemy, distance)
        shoot = (
            state.enemy_visible
            and distance <= self.fire_range
            and self.rng.random() < self.fire_chance
        )
        return PlayerAction(self._move[0], self._move[1], shoot)

    def _decide(self, state: ArenaState, to_enemy: Offset, distance: float) -> None:
        if self.rng.random() < self.lapse_chance * self.hold_frames:
            # Attention drifts (aiming, looking around): keep going or stop.
            self._move = self.rng.choice((self._move, STOP))
            self._next_decision = state.frame + self.rng.randint(15, 60)
            return
        if state.enemy_visible:
            move = self.move(state, to_enemy, distance)
        else:
            move = self.wander(state.frame, 20, 60, 0.5)
        if self.rng.random() < self.key_noise:
            move = OCTANT_DIRECTIONS[self.rng.randrange(8)]
        self._move = move
        self._next_decision = state.frame + self.hold_frames

    def move(self, state: ArenaState, to_enemy: Offset, distance: float) -> Move:
        return STOP

    # -- movement helpers ------------------------------------------------------

    @staticmethod
    def toward(to_enemy: Offset) -> Move:
        return quantize_direction(to_enemy[0], to_enemy[1])

    @staticmethod
    def away(to_enemy: Offset) -> Move:
        return quantize_direction(-to_enemy[0], -to_enemy[1])

    def wander(
        self, frame: int, min_hold: int, max_hold: int, stop_chance: float
    ) -> Move:
        if frame >= self._wander_until:
            if self.rng.random() < stop_chance:
                self._wander_move = STOP
            else:
                self._wander_move = OCTANT_DIRECTIONS[self.rng.randrange(8)]
            self._wander_until = frame + self.rng.randint(min_hold, max_hold)
        return self._wander_move


class IdleBot(PlayerBot):
    """Stands still and shoots."""

    style = "idle"


class WanderBot(PlayerBot):
    """Drifts in random directions, ignoring the enemy."""

    style = "wander"

    def move(self, state, to_enemy, distance):
        return self.wander(state.frame, 20, 90, 0.2)


class KiteBot(PlayerBot):
    """Flees when the enemy gets inside its comfort radius."""

    style = "kite"

    def setup(self) -> None:
        self.panic_radius = self.rng.uniform(250.0, 650.0)

    def move(self, state, to_enemy, distance):
        if distance < self.panic_radius:
            return self.away(to_enemy)
        return self.wander(state.frame, 10, 40, 0.5)


class StrafeBot(PlayerBot):
    """Circles sideways around the enemy, backing off when it gets close."""

    style = "strafe"

    def setup(self) -> None:
        self.spin = self.rng.choice((-1, 1))
        self.keep_away = self.rng.uniform(200.0, 500.0)

    def move(self, state, to_enemy, distance):
        tx = -to_enemy[1] * self.spin
        ty = to_enemy[0] * self.spin
        if distance < self.keep_away:
            tx -= to_enemy[0]
            ty -= to_enemy[1]
        return quantize_direction(tx, ty)


class JukeBot(StrafeBot):
    """Strafes, but flips direction on a rhythm."""

    style = "juke"

    def setup(self) -> None:
        super().setup()
        self.flip_every = self.rng.randint(20, 60)
        self._next_flip = self.flip_every

    def move(self, state, to_enemy, distance):
        if state.frame >= self._next_flip:
            self.spin = -self.spin
            self._next_flip = state.frame + self.flip_every
        return super().move(state, to_enemy, distance)


class ReverserBot(PlayerBot):
    """Always runs away, except that it doubles back briefly after every shot."""

    style = "reverser"

    def setup(self) -> None:
        self.double_back = self.rng.randint(10, 30)
        self._reverse_until = -1
        self._last_cooldown = 0

    def move(self, state, to_enemy, distance):
        if state.cooldown_left > self._last_cooldown:
            self._reverse_until = state.frame + self.double_back
        self._last_cooldown = state.cooldown_left
        if state.frame < self._reverse_until:
            return self.toward(to_enemy)
        return self.away(to_enemy)


class ChargerBot(PlayerBot):
    """Runs straight at the enemy while firing."""

    style = "charger"

    def move(self, state, to_enemy, distance):
        return self.toward(to_enemy)


class OrbitBot(PlayerBot):
    """Orbits the enemy at a preferred distance."""

    style = "orbit"

    def setup(self) -> None:
        self.spin = self.rng.choice((-1, 1))
        self.radius = self.rng.uniform(250.0, 550.0)

    def move(self, state, to_enemy, distance):
        radial = 2.0 * (distance - self.radius) / max(distance, 1.0)
        tx = -to_enemy[1] * self.spin + to_enemy[0] * radial
        ty = to_enemy[0] * self.spin + to_enemy[1] * radial
        return quantize_direction(tx, ty)


BOT_TYPES: Dict[str, Type[PlayerBot]] = {
    cls.style: cls
    for cls in (
        IdleBot,
        WanderBot,
        KiteBot,
        StrafeBot,
        JukeBot,
        ReverserBot,
        ChargerBot,
        OrbitBot,
    )
}
BOT_STYLES: Tuple[str, ...] = tuple(BOT_TYPES)


def make_bot(style: str, rng: random.Random) -> PlayerBot:
    if style not in BOT_TYPES:
        raise ValueError(f"unknown bot style {style!r}, expected one of {BOT_STYLES}")
    return BOT_TYPES[style](rng)
