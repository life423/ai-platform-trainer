"""Gymnasium environment for training the enemy agent on the canonical arena.

The enemy picks a move every ENEMY_DECISION_FRAMES frames: a 2D action in
[-1, 1] x [-1, 1], its velocity as a fraction of its per-axis top speed.
The player is a scripted, human-like bot whose style is sampled every round.

Rules match the game: a catch respawns the player away from the enemy, and a
missile hit hides the enemy briefly and then respawns it. Nothing ends a round
early; a round is cut off after 15 s (truncated).

Reward per decision:
- +1 for each catch, -1 for each missile hit.
- A cost per frame: 15 s without a catch costs as much as a missile hit. The
  enemy is a hunter, so a round it fails to win is a loss, not a safe draw.
- Potential-based shaping, GAMMA * phi(next) - phi(now) with
  phi = -distance / 250, which speeds up learning without changing which
  policy is optimal (Ng, Harada and Russell, 1999).
"""
import math
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ai_platform_trainer.ai.model_paths import MISSILE_SAC_MODEL
from ai_platform_trainer.arena.config import DEFAULT_SCREEN, ArenaConfig
from ai_platform_trainer.arena.episode import DEFAULT_MAX_FRAMES, ArenaEpisode
from ai_platform_trainer.arena.geometry import torus_delta
from ai_platform_trainer.arena.missile_policies import SacGuidance
from ai_platform_trainer.arena.observations import (
    ENEMY_DECISION_FRAMES,
    ENEMY_OBS_SIZE,
    OBS_CLIP,
    build_enemy_observation,
    enemy_action_to_displacement,
    enemy_player_distance,
)
from ai_platform_trainer.arena.player_bots import BOT_STYLES, make_bot
from ai_platform_trainer.arena.sim import CAUGHT, HIT
from ai_platform_trainer.arena.state import ArenaState

PLANNING_HORIZON_FRAMES = 800  # about 13 s: how far ahead rewards still count
# Per decision, so the horizon in seconds stays fixed if the decision rate changes.
# Potential-based shaping and the PPO trainer must use the same value.
GAMMA = 1.0 - ENEMY_DECISION_FRAMES / PLANNING_HORIZON_FRAMES
CATCH_REWARD = 1.0
HIT_PENALTY = -1.0
TIME_COST_PER_FRAME = 1.0 / DEFAULT_MAX_FRAMES
POTENTIAL_SCALE = 250.0
# A missile closer than this, and closing on the enemy, is a threat to dodge.
THREAT_RADIUS_PX = 250.0
THREAT_WEIGHT = 0.5
REWARD_SPEC = (
    "+1 per catch, -1 per missile hit, -1 per 15 s, plus potential-based shaping "
    "for distance to the player (phi = -distance / 250) and for the closest "
    "closing missile (phi -= 0.5 * threat, threat in [0, 1])"
)


def round_return(catches: int, hits: int, frames: int) -> float:
    """Unshaped, undiscounted return of one round: the objective training maximizes."""
    return CATCH_REWARD * catches + HIT_PENALTY * hits - TIME_COST_PER_FRAME * frames


def missile_threat(state: ArenaState) -> float:
    """How urgently the worst missile is closing on the enemy, from 0 to 1."""
    if not state.enemy_visible:
        return 0.0
    cfg, enemy = state.config, state.enemy
    speed = cfg.missile_speed
    worst = 0.0
    for missile in state.missiles:
        if missile.exploded:
            continue
        rx = torus_delta(missile.x, enemy.x, cfg.width)
        ry = torus_delta(missile.y, enemy.y, cfg.height)
        distance = math.hypot(rx, ry)
        if distance < 1e-6 or distance > THREAT_RADIUS_PX:
            continue
        closing = (missile.vx * rx + missile.vy * ry) / distance
        threat = max(0.0, closing / speed) * (1.0 - distance / THREAT_RADIUS_PX)
        worst = max(worst, min(1.0, threat))
    return worst


def sample_training_config(rng: random.Random) -> ArenaConfig:
    """Canonical rules on the fixed world, with randomized player aim.

    The world is always 800x600, the size the enemy and the SAC missile see in
    the game. The missile turn and prediction settings only drive basic homing,
    which the game no longer uses, so they are not randomized.
    """
    return ArenaConfig(missile_aim_jitter=rng.uniform(0.0, 0.2))


class EnemyArenaEnv(gymnasium.Env):
    """One enemy against one scripted, human-like player bot."""

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        randomize: bool = True,
        screen: Tuple[int, int] = DEFAULT_SCREEN,
        bot_styles: Sequence[str] = BOT_STYLES,
        max_frames: int = DEFAULT_MAX_FRAMES,
        shaping: float = 1.0,
        preload_prob: float = 0.5,
        guidance_model: str = MISSILE_SAC_MODEL,
    ) -> None:
        super().__init__()
        self.randomize = randomize
        # The game's SAC missile steers the simulated missiles, so the enemy
        # learns against the same missiles the player faces.
        self.guidance = SacGuidance(guidance_model)
        self.screen = (int(screen[0]), int(screen[1]))
        self.bot_styles = tuple(bot_styles)
        self.max_frames = max_frames
        self.shaping = shaping
        self.preload_prob = preload_prob
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(ENEMY_OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.episode: Optional[ArenaEpisode] = None
        self._potential = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(0, 2**31 - 1)))
        if self.randomize:
            config = sample_training_config(rng)
        else:
            config = ArenaConfig(width=self.screen[0], height=self.screen[1])
        style = rng.choice(self.bot_styles)
        bot = make_bot(style, random.Random(rng.getrandbits(32)))
        self.episode = ArenaEpisode(
            config,
            bot,
            random.Random(rng.getrandbits(32)),
            self.max_frames,
            guidance=self.guidance,
        )
        preload = rng.randint(1, 2) if rng.random() < self.preload_prob else 0
        state = self.episode.reset(preload_missiles=preload)
        self._potential = self._phi(state)
        return build_enemy_observation(state), {"bot_style": style}

    def step(self, action):
        episode = self.episode
        assert episode is not None, "call reset() before step()"
        state = episode.state
        teleports_before = state.teleports
        dx, dy = enemy_action_to_displacement(action, state.config)
        reward, expired, catches, hits, last_event = 0.0, 0, 0, 0, ""
        for _ in range(ENEMY_DECISION_FRAMES):
            result = episode.step(dx, dy)
            reward -= TIME_COST_PER_FRAME
            expired += result.expired
            catches += result.event == CAUGHT
            hits += result.event == HIT
            last_event = result.event or last_event
            if result.truncated:
                break
        reward += CATCH_REWARD * catches + HIT_PENALTY * hits
        # A respawn moves a body without the enemy moving there, so the shaping
        # must not count that jump as progress or as a setback.
        if state.teleports != teleports_before:
            self._potential = self._phi(state)
        potential = self._phi(state)
        reward += self.shaping * (GAMMA * potential - self._potential)
        self._potential = potential
        info = {
            "event": last_event,
            "catches": catches,
            "hits": hits,
            "missiles_expired": expired,
            # A round cut off by the clock is not a death. Stable-Baselines3 only
            # bootstraps the future value of a truncated round when this key is set.
            "TimeLimit.truncated": bool(result.truncated),
        }
        return (
            build_enemy_observation(state),
            float(reward),
            False,
            result.truncated,
            info,
        )

    @staticmethod
    def _phi(state: ArenaState) -> float:
        return (
            -enemy_player_distance(state) / POTENTIAL_SCALE
            - THREAT_WEIGHT * missile_threat(state)
        )
