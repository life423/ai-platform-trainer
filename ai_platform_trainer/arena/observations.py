"""Shared observation and action encoding for the enemy agent.

build_enemy_observation() is the only way an enemy observation is ever built.
The training environment calls it on sim state and the game calls it on a
snapshot of the live entities (see game_bridge). Normalization lives here
rather than in a training wrapper, so inference cannot forget to apply it.
"""
import math
from typing import List, Sequence, Tuple

import numpy as np

from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.geometry import torus_delta
from ai_platform_trainer.arena.state import ArenaState, MissileState

ENEMY_OBS_VERSION = 1
ENEMY_OBS_MISSILES = 3
MISSILE_FEATURES = 6
ENEMY_BASE_FEATURES = 11
ENEMY_OBS_SIZE = ENEMY_BASE_FEATURES + ENEMY_OBS_MISSILES * MISSILE_FEATURES
POSITION_SCALE = 500.0  # px; relative offsets are divided by this
ARENA_SCALE = 2000.0
OBS_CLIP = 5.0
# The enemy picks a new move every 4 frames (15 Hz) and holds it in between,
# so a catch a few seconds away still counts after discounting.
ENEMY_DECISION_FRAMES = 4

ENEMY_OBS_FEATURES: Tuple[str, ...] = (
    "player_dx",
    "player_dy",
    "player_distance",
    "player_vx",
    "player_vy",
    "enemy_vx",
    "enemy_vy",
    "arena_width",
    "arena_height",
    "player_cooldown",
    "player_missile_count",
) + tuple(
    f"missile{slot}_{name}"
    for slot in range(ENEMY_OBS_MISSILES)
    for name in ("dx", "dy", "vx", "vy", "life", "present")
)


def enemy_to_player(state: ArenaState) -> Tuple[float, float]:
    """Shortest offset from the enemy to the player (same size, so centers match)."""
    cfg = state.config
    return (
        torus_delta(state.enemy.x, state.player.x, cfg.period_x),
        torus_delta(state.enemy.y, state.player.y, cfg.period_y),
    )


def enemy_player_distance(state: ArenaState) -> float:
    dx, dy = enemy_to_player(state)
    return math.hypot(dx, dy)


def missile_threats(
    state: ArenaState,
) -> List[Tuple[float, float, float, MissileState]]:
    """Live missiles as (distance, dx, dy, missile) from the enemy center, nearest first."""
    cfg = state.config
    # enemy center to missile center: (m + missile/2) - (e + body/2)
    offset = (cfg.missile_size - cfg.body_size) / 2.0
    threats = []
    for missile in state.missiles:
        if missile.exploded:
            continue
        dx = torus_delta(state.enemy.x, missile.x + offset, cfg.period_x)
        dy = torus_delta(state.enemy.y, missile.y + offset, cfg.period_y)
        threats.append((math.hypot(dx, dy), dx, dy, missile))
    threats.sort(key=lambda item: (item[0], item[1], item[2]))
    return threats


def build_enemy_observation(state: ArenaState) -> np.ndarray:
    """Encode an ArenaState as the enemy policy input (see ENEMY_OBS_FEATURES)."""
    cfg = state.config
    dx, dy = enemy_to_player(state)
    features = [
        dx / POSITION_SCALE,
        dy / POSITION_SCALE,
        math.hypot(dx, dy) / POSITION_SCALE,
        state.player.vx / cfg.player_speed,
        state.player.vy / cfg.player_speed,
        state.enemy.vx / cfg.enemy_speed,
        state.enemy.vy / cfg.enemy_speed,
        cfg.width / ARENA_SCALE,
        cfg.height / ARENA_SCALE,
        state.cooldown_left / cfg.missile_cooldown_frames,
        len(state.missiles) / cfg.max_missiles,
    ]
    max_life = cfg.max_missile_life_frames
    threats = missile_threats(state)
    for slot in range(ENEMY_OBS_MISSILES):
        if slot < len(threats):
            _, mdx, mdy, missile = threats[slot]
            features.extend(
                [
                    mdx / POSITION_SCALE,
                    mdy / POSITION_SCALE,
                    missile.vx / cfg.missile_speed,
                    missile.vy / cfg.missile_speed,
                    missile.frames_left / max_life,
                    1.0,
                ]
            )
        else:
            features.extend([0.0] * MISSILE_FEATURES)
    obs = np.asarray(features, dtype=np.float32)
    np.clip(obs, -OBS_CLIP, OBS_CLIP, out=obs)
    return obs


def enemy_action_to_displacement(
    action: Sequence[float], config: ArenaConfig
) -> Tuple[float, float]:
    """Map a policy action in [-1, 1] x [-1, 1] to a per-frame displacement in px."""
    ax = max(-1.0, min(1.0, float(action[0])))
    ay = max(-1.0, min(1.0, float(action[1])))
    return ax * config.enemy_speed, ay * config.enemy_speed


# -- missile guidance observation (SAC) --------------------------------------

# The SAC missile model was trained with these scales (train_missile_sac.py)
SAC_MAX_MISSILE_SPEED = 10.0
SAC_TARGET_SPEED_SCALE = 5.0
# SmartMissile.RL_TURN_SCALES["sac"]: degrees per step at an action of 1.0
SAC_TURN_SCALE_DEG = 20.0
SAC_MISSILE_FEATURES = 11


def sac_missile_observation(
    x: float,
    y: float,
    vx: float,
    vy: float,
    target_x: float,
    target_y: float,
    target_vx: float,
    target_vy: float,
    width: float,
    height: float,
) -> np.ndarray:
    """The 11 inputs the SAC missile model was trained on.

    The game (SmartMissile) and the arena both build the missile's input here,
    with the same formula as train_missile_sac.py. Positions are scaled to
    [-1, 1] by the screen size, which is why the world is fixed at 800x600.
    """
    dx = target_x - x
    dy = target_y - y
    distance = math.hypot(dx, dy)
    angle_to_target = math.atan2(dy, dx)
    missile_angle = math.atan2(vy, vx)
    relative_angle = angle_to_target - missile_angle
    relative_angle_norm = (
        math.atan2(math.sin(relative_angle), math.cos(relative_angle)) / math.pi
    )
    return np.array(
        [
            (x / width) * 2 - 1,
            (y / height) * 2 - 1,
            vx / SAC_MAX_MISSILE_SPEED,
            vy / SAC_MAX_MISSILE_SPEED,
            (target_x / width) * 2 - 1,
            (target_y / height) * 2 - 1,
            target_vx / SAC_TARGET_SPEED_SCALE,
            target_vy / SAC_TARGET_SPEED_SCALE,
            distance / math.hypot(width, height),
            angle_to_target / math.pi,
            relative_angle_norm,
        ],
        dtype=np.float32,
    )
