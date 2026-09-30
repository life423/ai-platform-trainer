"""Headless, frame-accurate simulation of Play mode.

One frame runs the same phases, in the same order, as
PlayLearningMode.update():

1. player_phase: the player fires (if allowed), then moves and wraps
2. enemy_phase: the enemy moves and wraps
3. collision_phase: enemy touching the player, then missiles hitting the enemy
4. missile_phase: every missile steers and moves missile_substeps times,
   expires, wraps, and finished explosions are removed

Agents observe the state between phases 1 and 2, exactly where the game calls
the enemy update_movement().
"""
import math
import random
from typing import Callable, Optional, Tuple

from ai_platform_trainer.arena.config import ArenaConfig, ms_to_frames
from ai_platform_trainer.arena.geometry import (
    normalize_angle,
    rects_overlap,
    wrap_coordinate,
    wrapped_move,
)
from ai_platform_trainer.arena.state import ArenaState, Body, MissileState

CAUGHT = "caught"
HIT = "hit"

Guidance = Callable[[MissileState, float, float, ArenaConfig], None]


def basic_homing(
    missile: MissileState, target_x: float, target_y: float, config: ArenaConfig
) -> None:
    """One substep of SmartMissile basic homing: lead the target, then turn."""
    target_vx = target_x - missile.last_target_x
    target_vy = target_y - missile.last_target_y
    missile.last_target_x = target_x
    missile.last_target_y = target_y
    predicted_x = target_x + target_vx * config.missile_prediction
    predicted_y = target_y + target_vy * config.missile_prediction
    desired = math.atan2(predicted_y - missile.y, predicted_x - missile.x)
    current = math.atan2(missile.vy, missile.vx)
    turn = math.degrees(normalize_angle(desired - current))
    turn = max(-config.missile_turn_deg, min(config.missile_turn_deg, turn))
    new_angle = current + math.radians(turn)
    missile.vx = config.missile_speed * math.cos(new_angle)
    missile.vy = config.missile_speed * math.sin(new_angle)


class ArenaSim:
    """A pygame-free Play mode. Call the four phases in order once per frame."""

    def __init__(
        self,
        config: ArenaConfig,
        rng: Optional[random.Random] = None,
        guidance: Guidance = basic_homing,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else random.Random()
        self.guidance = guidance
        self.state = ArenaState(
            config=config, player=Body(0.0, 0.0), enemy=Body(0.0, 0.0)
        )

    # -- setup -------------------------------------------------------------

    def reset(
        self,
        player_xy: Optional[Tuple[float, float]] = None,
        enemy_xy: Optional[Tuple[float, float]] = None,
    ) -> ArenaState:
        if player_xy is None:
            player_xy = self.random_position()
        if enemy_xy is None:
            enemy_xy = self.spawn_position_away_from(player_xy)
        self.state = ArenaState(
            config=self.config,
            player=Body(float(player_xy[0]), float(player_xy[1])),
            enemy=Body(float(enemy_xy[0]), float(enemy_xy[1])),
        )
        return self.state

    def random_position(self) -> Tuple[float, float]:
        cfg = self.config
        return (
            float(self.rng.randint(0, cfg.width - cfg.body_size)),
            float(self.rng.randint(0, cfg.height - cfg.body_size)),
        )

    def spawn_position_away_from(
        self, other: Tuple[float, float], attempts: int = 10
    ) -> Tuple[float, float]:
        """PlayLearningMode._respawn_enemy: up to 10 draws, keep the last."""
        x, y = self.random_position()
        for _ in range(attempts - 1):
            if math.hypot(x - other[0], y - other[1]) > self.config.min_spawn_distance:
                break
            x, y = self.random_position()
        return x, y

    def spawn_missile(
        self, x: float, y: float, angle: float, life_frames: int, age_frames: int = 0
    ) -> MissileState:
        """Add a missile directly (used to start episodes with threats in flight)."""
        cfg = self.config
        missile = MissileState(
            x=float(x),
            y=float(y),
            vx=cfg.missile_speed * math.cos(angle),
            vy=cfg.missile_speed * math.sin(angle),
            life_frames=int(life_frames),
            age_frames=int(age_frames),
            last_target_x=self.state.enemy.x,
            last_target_y=self.state.enemy.y,
        )
        self.state.missiles.append(missile)
        return missile

    # -- phase 1: player -----------------------------------------------------

    def can_fire(self) -> bool:
        return (
            self.state.cooldown_left <= 0
            and len(self.state.missiles) < self.config.max_missiles
        )

    def try_fire(self) -> bool:
        """PlayerPlay.shoot_missile, aimed at the enemy top-left corner."""
        if not self.can_fire():
            return False
        cfg = self.config
        player, enemy = self.state.player, self.state.enemy
        half = cfg.body_size // 2
        start_x = player.x + half
        start_y = player.y + half
        life_ms = self.rng.randint(cfg.missile_life_ms[0], cfg.missile_life_ms[1])
        angle = math.atan2(enemy.y - start_y, enemy.x - start_x)
        angle += self.rng.uniform(-cfg.missile_aim_jitter, cfg.missile_aim_jitter)
        self.spawn_missile(start_x, start_y, angle, ms_to_frames(life_ms))
        self.state.cooldown_left = cfg.missile_cooldown_frames
        return True

    def player_phase(self, dx: int, dy: int, shoot: bool) -> bool:
        """Fire from the pre-move position, then move and wrap. True if fired."""
        fired = self.try_fire() if shoot else False
        cfg, player = self.config, self.state.player
        player.x, player.y, player.vx, player.vy = wrapped_move(
            player.x,
            player.y,
            dx * cfg.player_speed,
            dy * cfg.player_speed,
            cfg.width,
            cfg.height,
            cfg.body_size,
        )
        return fired

    # -- phase 2: enemy ------------------------------------------------------

    def enemy_phase(self, dx: float, dy: float, capped: bool = True) -> None:
        cfg, enemy = self.config, self.state.enemy
        if not self.state.enemy_visible:
            enemy.vx = enemy.vy = 0.0
            return
        if capped:
            dx = max(-cfg.enemy_speed, min(cfg.enemy_speed, dx))
            dy = max(-cfg.enemy_speed, min(cfg.enemy_speed, dy))
        enemy.x, enemy.y, enemy.vx, enemy.vy = wrapped_move(
            enemy.x, enemy.y, dx, dy, cfg.width, cfg.height, cfg.body_size
        )

    # -- phase 3: collisions -------------------------------------------------

    def collision_phase(self) -> str:
        """PlayLearningMode._check_collisions: a catch wins over a missile hit."""
        state, cfg = self.state, self.config
        if not state.enemy_visible:
            return ""
        enemy, player = state.enemy, state.player
        if rects_overlap(
            player.x, player.y, cfg.body_size, enemy.x, enemy.y, cfg.body_size
        ):
            return CAUGHT
        for missile in state.missiles:
            if missile.exploded:
                continue
            if rects_overlap(
                missile.x, missile.y, cfg.missile_size, enemy.x, enemy.y, cfg.body_size
            ):
                state.missiles.remove(missile)
                return HIT
        return ""

    # -- phase 4: missiles and end of frame ----------------------------------

    def missile_phase(self) -> int:
        """Steer, move, expire and wrap missiles. Returns how many expired."""
        state, cfg = self.state, self.config
        enemy = state.enemy
        expired = 0
        survivors = []
        for missile in state.missiles:
            if missile.exploded:
                missile.explosion_age += 1
                if missile.explosion_age < cfg.explosion_frames:
                    survivors.append(missile)
                continue
            for _ in range(cfg.missile_substeps):
                if state.enemy_visible:
                    self.guidance(missile, enemy.x, enemy.y, cfg)
                missile.x += missile.vx
                missile.y += missile.vy
            if missile.age_frames >= missile.life_frames:
                missile.exploded = True
                missile.vx = missile.vy = 0.0
                expired += 1
            missile.age_frames += 1
            missile.x = wrap_coordinate(missile.x, cfg.width, cfg.missile_size)
            missile.y = wrap_coordinate(missile.y, cfg.height, cfg.missile_size)
            survivors.append(missile)
        state.missiles = survivors
        if state.cooldown_left > 0:
            state.cooldown_left -= 1
        state.frame += 1
        return expired
