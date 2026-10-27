"""Gymnasium environment for the AI Agent (the enemy), on the canonical arena rules.

The enemy chooses a 2D move every ENEMY_DECISION_FRAMES frames and holds it in
between, as the game does. A round lasts ROUND_FRAMES frames and continues after a
catch or a hit, as in Play mode. Missiles are the SAC guidance the game uses.

Reward per decision:
- +CATCH_REWARD for each catch, hit_reward (negative) for each missile hit.
- Potential-based shaping, phi = -distance / DISTANCE_SCALE - threat_weight * threat,
  where threat is how fast the worst nearby missile is closing on the enemy. The
  shaping gives the approach signal without changing which policy is best.
There is no time cost: the objective is catches and hits. A catch or a hit moves
the player or hides the enemy, so the shaping baseline is reset then.

The deployed model (models/enemy/enemy.zip) was trained with hit_reward = -3.0 and
threat_weight = 1.5, which are the defaults below.
"""
import random
from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ai_platform_trainer.ai.model_paths import MISSILE_SAC_MODEL
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.episode import ArenaEpisode
from ai_platform_trainer.arena.geometry import torus_delta
from ai_platform_trainer.arena.missile_policies import SacGuidance
from ai_platform_trainer.arena.observations import (
    ENEMY_DECISION_FRAMES,
    ENEMY_OBS_SIZE,
    build_enemy_observation,
    enemy_action_to_displacement,
    enemy_player_distance,
)
from ai_platform_trainer.arena.player_bots import BOT_STYLES, make_bot
from ai_platform_trainer.arena.sim import CAUGHT, HIT
from ai_platform_trainer.arena.state import ArenaState

ROUND_FRAMES = 900  # 15 seconds at 60 FPS
CATCH_REWARD = 1.0
HIT_REWARD = -3.0
DISTANCE_SCALE = 200.0  # px: shaping for distance to the player
THREAT_RADIUS_PX = 250.0  # a missile this close and closing is a threat
THREAT_WEIGHT = 1.5
MAX_PRELOADED_MISSILES = 2  # rounds may start with missiles already in flight


def missile_threat(state: ArenaState) -> float:
    """How fast the worst nearby missile is closing on the enemy, from 0 to 1."""
    if not state.enemy_visible:
        return 0.0
    cfg = state.config
    enemy = state.enemy
    worst = 0.0
    for missile in state.missiles:
        if missile.exploded:
            continue
        dx = torus_delta(missile.x, enemy.x, cfg.period_x)
        dy = torus_delta(missile.y, enemy.y, cfg.period_y)
        distance = (dx * dx + dy * dy) ** 0.5
        if distance < 1e-6 or distance > THREAT_RADIUS_PX:
            continue
        closing = (missile.vx * dx + missile.vy * dy) / distance
        if closing <= 0.0:
            continue
        threat = closing / cfg.missile_speed * (1.0 - distance / THREAT_RADIUS_PX)
        worst = max(worst, threat)
    return min(1.0, worst)


def shaping_potential(state: ArenaState, threat_weight: float = THREAT_WEIGHT) -> float:
    """phi for potential-based shaping. Zero while the enemy is hidden."""
    if not state.enemy_visible:
        return 0.0
    return (
        -enemy_player_distance(state) / DISTANCE_SCALE
        - threat_weight * missile_threat(state)
    )


class ChaserEnv(gymnasium.Env):
    """The enemy against a random player bot, one decision per step."""

    def __init__(
        self,
        config: Optional[ArenaConfig] = None,
        guidance_model: str = MISSILE_SAC_MODEL,
        hit_reward: float = HIT_REWARD,
        threat_weight: float = THREAT_WEIGHT,
    ) -> None:
        self.config = config if config is not None else ArenaConfig()
        self.guidance = SacGuidance(guidance_model)
        self.hit_reward = hit_reward
        self.threat_weight = threat_weight
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(ENEMY_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.rng = random.Random()
        self.episode: Optional[ArenaEpisode] = None
        self._potential = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        style = self.rng.choice(BOT_STYLES)
        bot = make_bot(style, random.Random(self.rng.getrandbits(32)))
        self.episode = ArenaEpisode(
            self.config,
            bot,
            random.Random(self.rng.getrandbits(32)),
            ROUND_FRAMES,
            guidance=self.guidance,
        )
        state = self.episode.reset(
            preload_missiles=self.rng.randint(0, MAX_PRELOADED_MISSILES)
        )
        self._potential = shaping_potential(state, self.threat_weight)
        return self._observe(), {"bot": style}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.episode is None:
            raise RuntimeError("call reset() before step()")
        episode = self.episode
        dx, dy = enemy_action_to_displacement(action, self.config)
        teleports_before = episode.state.teleports

        catches = hits = 0
        truncated = False
        for _ in range(ENEMY_DECISION_FRAMES):
            result = episode.step(dx, dy)
            catches += result.event == CAUGHT
            hits += result.event == HIT
            if result.truncated:
                truncated = True
                break

        state = episode.state
        reward = CATCH_REWARD * catches + self.hit_reward * hits
        potential = shaping_potential(state, self.threat_weight)
        if state.teleports == teleports_before:
            reward += potential - self._potential
        self._potential = potential
        info = {"catches": catches, "hits": hits}
        return self._observe(), float(reward), False, truncated, info

    def _observe(self) -> np.ndarray:
        assert self.episode is not None
        return build_enemy_observation(self.episode.state).astype(np.float32)
