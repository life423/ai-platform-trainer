"""Gymnasium environment for training the enemy agent on the canonical arena.

The enemy picks a move every ENEMY_DECISION_FRAMES frames: a 2D action in
[-1, 1] x [-1, 1], its velocity as a fraction of the player per-axis speed.
The player is a scripted, human-like bot whose style is sampled every round.

Reward per decision:
- +1 for catching the player, -1 for being hit by a missile; both end the
  round. A round with neither is cut off after 15 s (truncated).
- A small cost per frame, so stalling is a failure rather than a safe choice:
  a full 15 s stalemate costs 0.5.
- Potential-based shaping, GAMMA * phi(next) - phi(now) with
  phi = -distance / 1000, which speeds up learning without changing which
  policy is optimal (Ng, Harada and Russell, 1999).
"""
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from ai_platform_trainer.arena.config import DEFAULT_SCREEN, ArenaConfig
from ai_platform_trainer.arena.episode import DEFAULT_MAX_FRAMES, ArenaEpisode
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

GAMMA = 0.99  # per decision; shaping and the PPO trainer must share it
CATCH_REWARD = 1.0
HIT_PENALTY = -1.0
TIME_COST_PER_FRAME = 0.5 / DEFAULT_MAX_FRAMES
POTENTIAL_SCALE = 1000.0
REWARD_SPEC = (
    "+1 catch, -1 missile hit, -0.5 per 15 s of play, "
    "plus potential-based distance shaping (phi = -distance / 1000)"
)


def sample_training_config(rng: random.Random) -> ArenaConfig:
    """Canonical rules on a random screen size, with randomized missile guidance."""
    return ArenaConfig(
        width=rng.randint(1100, 1920),
        height=rng.randint(650, 1200),
        missile_turn_deg=rng.uniform(8.0, 20.0),
        missile_prediction=rng.uniform(0.0, 1.0),
        missile_aim_jitter=rng.uniform(0.0, 0.2),
    )


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
    ) -> None:
        super().__init__()
        self.randomize = randomize
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
            config, bot, random.Random(rng.getrandbits(32)), self.max_frames
        )
        preload = rng.randint(1, 2) if rng.random() < self.preload_prob else 0
        state = self.episode.reset(preload_missiles=preload)
        self._potential = self._phi(state)
        return build_enemy_observation(state), {"bot_style": style}

    def step(self, action):
        episode = self.episode
        assert episode is not None, "call reset() before step()"
        state = episode.state
        dx, dy = enemy_action_to_displacement(action, state.config)
        reward, expired = 0.0, 0
        for _ in range(ENEMY_DECISION_FRAMES):
            result = episode.step(dx, dy)
            reward -= TIME_COST_PER_FRAME
            expired += result.expired
            if result.terminated or result.truncated:
                break
        if result.event == CAUGHT:
            reward += CATCH_REWARD
        elif result.event == HIT:
            reward += HIT_PENALTY
        potential = 0.0 if result.terminated else self._phi(state)
        reward += self.shaping * (GAMMA * potential - self._potential)
        self._potential = potential
        info = {"event": result.event, "missiles_expired": expired}
        return (
            build_enemy_observation(state),
            float(reward),
            result.terminated,
            result.truncated,
            info,
        )

    @staticmethod
    def _phi(state: ArenaState) -> float:
        return -enemy_player_distance(state) / POTENTIAL_SCALE
