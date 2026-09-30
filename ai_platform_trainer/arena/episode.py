"""One round of Play mode: from spawn until a catch, a missile hit or timeout.

The training environment and the benchmark both drive rounds through this
class, so reward, termination and timing rules cannot drift apart.
"""
import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from ai_platform_trainer.arena.config import ArenaConfig, ms_to_frames
from ai_platform_trainer.arena.player_bots import PlayerAction, PlayerBot
from ai_platform_trainer.arena.sim import ArenaSim
from ai_platform_trainer.arena.state import ArenaState

DEFAULT_MAX_FRAMES = 900  # 15 seconds at 60 FPS


@dataclass(frozen=True)
class StepResult:
    event: str  # '', 'caught' or 'hit'
    expired: int  # missiles that ran out of fuel this frame
    terminated: bool
    truncated: bool


class ArenaEpisode:
    """Plays the bot side automatically; the caller supplies enemy moves."""

    def __init__(
        self,
        config: ArenaConfig,
        bot: PlayerBot,
        rng: random.Random,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> None:
        self.sim = ArenaSim(config, rng)
        self.bot = bot
        self.rng = rng
        self.max_frames = max_frames
        self.missiles_fired = 0
        self.missiles_expired = 0
        self.last_player_action = PlayerAction()

    @property
    def state(self) -> ArenaState:
        return self.sim.state

    def reset(
        self,
        player_xy: Optional[Tuple[float, float]] = None,
        enemy_xy: Optional[Tuple[float, float]] = None,
        preload_missiles: int = 0,
    ) -> ArenaState:
        """Spawn, optionally put missiles in flight, then run the first player turn."""
        self.sim.reset(player_xy, enemy_xy)
        self.missiles_fired = 0
        self.missiles_expired = 0
        for _ in range(preload_missiles):
            self._preload_missile()
        self._player_turn()
        return self.sim.state

    def step(self, dx: float, dy: float, capped: bool = True) -> StepResult:
        """Finish the current frame with this enemy move and start the next."""
        sim = self.sim
        sim.enemy_phase(dx, dy, capped)
        event = sim.collision_phase()
        if event:
            return StepResult(event, 0, True, False)
        expired = sim.missile_phase()
        self.missiles_expired += expired
        self._player_turn()
        return StepResult("", expired, False, sim.state.frame >= self.max_frames)

    def _player_turn(self) -> None:
        action = self.bot.act(self.sim.state)
        if self.sim.player_phase(action.dx, action.dy, action.shoot):
            self.missiles_fired += 1
        self.last_player_action = action

    def _preload_missile(self) -> None:
        """A missile already homing in from 150-600 px away, roughly on target."""
        cfg = self.sim.config
        enemy = self.sim.state.enemy
        bearing = self.rng.uniform(-math.pi, math.pi)
        distance = self.rng.uniform(150.0, 600.0)
        x = enemy.x + math.cos(bearing) * distance
        y = enemy.y + math.sin(bearing) * distance
        heading = bearing + math.pi + self.rng.uniform(-0.5, 0.5)
        life = ms_to_frames(
            self.rng.randint(cfg.missile_life_ms[0], cfg.missile_life_ms[1])
        )
        self.sim.spawn_missile(x, y, heading, life, age_frames=self.rng.randint(0, 60))
