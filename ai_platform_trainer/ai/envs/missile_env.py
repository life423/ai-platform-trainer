"""Gymnasium environment that trains the SAC missile on the canonical arena rules.

One missile is fired the way PlayerPlay fires it, at an enemy that moves at up to
enemy_speed per frame and sometimes stands still. Each frame runs in the arena's
order (enemy move, collision check, guidance, missile move), with positions wrapping
at the edges. Sizes, speeds, lifetimes and the aim jitter come from ArenaConfig.

The action turns the missile by SAC_TURN_SCALE_DEG degrees per unit, through the same
_apply_turn and observation encoder as arena.missile_policies.SacGuidance, so a model
trained here steers the game's missiles and the arena's the same way.
"""
import math
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ai_platform_trainer.arena.config import ArenaConfig, ms_to_frames
from ai_platform_trainer.arena.geometry import (
    rects_overlap,
    wrap_coordinate,
    wrapped_move,
)
from ai_platform_trainer.arena.missile_policies import _apply_turn
from ai_platform_trainer.arena.observations import (
    SAC_TURN_SCALE_DEG,
    sac_missile_observation,
)
from ai_platform_trainer.arena.sim import random_spawn, spawn_away
from ai_platform_trainer.arena.state import MissileState

HIT_BONUS = 10.0
STEP_COST = 0.005
# Shaping: the reward for closing on the enemy is distance gained / POTENTIAL_SCALE
POTENTIAL_SCALE = 100.0


class MissileArenaEnv(gymnasium.Env):
    """One SAC missile against one enemy, on the canonical arena rules."""

    def __init__(
        self, config: Optional[ArenaConfig] = None, still_fraction: float = 0.3
    ) -> None:
        self.config = config if config is not None else ArenaConfig()
        self.still_fraction = still_fraction
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.rng = random.Random()
        self.missile: MissileState
        self.enemy: Tuple[float, float] = (0.0, 0.0)
        self._still = False
        self._move_left = 0
        self._enemy_step: Tuple[float, float] = (0.0, 0.0)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        cfg = self.config
        player = random_spawn(cfg, self.rng)  # the player's top-left corner
        self.enemy = spawn_away(cfg, self.rng, player)
        half = cfg.body_size // 2
        start_x, start_y = player[0] + half, player[1] + half
        angle = math.atan2(self.enemy[1] - start_y, self.enemy[0] - start_x)
        angle += self.rng.uniform(-cfg.missile_aim_jitter, cfg.missile_aim_jitter)
        self.missile = MissileState(
            x=float(start_x),
            y=float(start_y),
            vx=cfg.missile_speed * math.cos(angle),
            vy=cfg.missile_speed * math.sin(angle),
            life_frames=ms_to_frames(self.rng.randint(*cfg.missile_life_ms)),
            last_target_x=self.enemy[0],
            last_target_y=self.enemy[1],
        )
        self._still = self.rng.random() < self.still_fraction
        self._move_left = 0
        return self._observe(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cfg = self.config
        m = self.missile
        ex, ey = self.enemy

        # Guidance and motion for this frame, as arena.sim.missile_phase does
        before = math.hypot(m.x - ex, m.y - ey)
        turn = float(np.clip(action[0], -1.0, 1.0)) * SAC_TURN_SCALE_DEG
        _apply_turn(m, turn, cfg.missile_speed)
        m.last_target_x, m.last_target_y = ex, ey
        m.x = wrap_coordinate(m.x + m.vx, cfg.width, cfg.missile_size)
        m.y = wrap_coordinate(m.y + m.vy, cfg.height, cfg.missile_size)
        m.age_frames += 1
        after = math.hypot(m.x - ex, m.y - ey)
        reward = (before - after) / POTENTIAL_SCALE - STEP_COST

        if m.age_frames >= m.life_frames:
            return self._finish(reward, hit=False, truncated=True)

        # The next frame: the enemy moves, then the collision check, as in the arena
        self._move_enemy()
        if rects_overlap(
            m.x, m.y, cfg.missile_size, self.enemy[0], self.enemy[1], cfg.body_size
        ):
            return self._finish(reward + HIT_BONUS, hit=True, truncated=False)
        return self._observe(), float(reward), False, False, {"hit": False}

    def _finish(
        self, reward: float, hit: bool, truncated: bool
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs = np.zeros(11, dtype=np.float32)
        return obs, float(reward), hit, truncated, {"hit": hit}

    def _observe(self) -> np.ndarray:
        cfg = self.config
        m = self.missile
        ex, ey = self.enemy
        return sac_missile_observation(
            m.x,
            m.y,
            m.vx,
            m.vy,
            ex,
            ey,
            ex - m.last_target_x,
            ey - m.last_target_y,
            cfg.width,
            cfg.height,
        ).astype(np.float32)

    def _move_enemy(self) -> None:
        cfg = self.config
        if self._move_left <= 0:
            speed = 0.0 if self._still else self.rng.uniform(0.0, cfg.enemy_speed)
            heading = self.rng.uniform(-math.pi, math.pi)
            self._enemy_step = (speed * math.cos(heading), speed * math.sin(heading))
            self._move_left = self.rng.randint(15, 60)
        self._move_left -= 1
        dx, dy = self._enemy_step
        ex, ey = self.enemy
        nx, ny, _, _ = wrapped_move(
            ex, ey, dx, dy, cfg.width, cfg.height, cfg.body_size
        )
        self.enemy = (nx, ny)
