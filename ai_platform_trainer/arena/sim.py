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
    rects_overlap,
    wrap_coordinate,
    wrapped_move,
)
from ai_platform_trainer.arena.state import ArenaState, Body, MissileState

CAUGHT = "caught"
HIT = "hit"

Guidance = Callable[[MissileState, float, float, ArenaConfig], None]


def random_spawn(config: ArenaConfig, rng: random.Random) -> Tuple[float, float]:
    """A uniformly random top-left corner for a body that fits in the world."""
    return (
        float(rng.randint(0, config.width - config.body_size)),
        float(rng.randint(0, config.height - config.body_size)),
    )


def spawn_away(
    config: ArenaConfig,
    rng: random.Random,
    other: Tuple[float, float],
    attempts: int = 10,
) -> Tuple[float, float]:
    """The respawn rule for both the player and the enemy.

    Up to `attempts` draws, keeping the last one if none is far enough from `other`.
    """
    x, y = random_spawn(config, rng)
    for _ in range(attempts - 1):
        if math.hypot(x - other[0], y - other[1]) > config.min_spawn_distance:
            break
        x, y = random_spawn(config, rng)
    return x, y


class ArenaSim:
    """A pygame-free Play mode. Call the four phases in order once per frame."""

    def __init__(
        self,
        config: ArenaConfig,
        rng: Optional[random.Random] = None,
        *,
        guidance: Guidance,
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
        return random_spawn(self.config, self.rng)

    def spawn_position_away_from(
        self, other: Tuple[float, float], attempts: int = 10
    ) -> Tuple[float, float]:
        return spawn_away(self.config, self.rng, other, attempts)

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
            self._count_down_respawn()
            return
        if capped:
            dx = max(-cfg.enemy_speed, min(cfg.enemy_speed, dx))
            dy = max(-cfg.enemy_speed, min(cfg.enemy_speed, dy))
        enemy.x, enemy.y, enemy.vx, enemy.vy = wrapped_move(
            enemy.x, enemy.y, dx, dy, cfg.width, cfg.height, cfg.body_size
        )

    # -- phase 3: collisions -------------------------------------------------

    def collision_phase(self) -> str:
        """PlayLearningMode._check_collisions. The game goes on after either event.

        A catch respawns the player away from the enemy; the enemies stay put.
        A missile hit removes the missile and hides the enemy until it respawns.
        A catch wins over a missile hit in the same frame.
        """
        state, cfg = self.state, self.config
        if not state.enemy_visible:
            return ""
        enemy, player = state.enemy, state.player
        if rects_overlap(
            player.x, player.y, cfg.body_size, enemy.x, enemy.y, cfg.body_size
        ):
            self._respawn_player()
            return CAUGHT
        for missile in state.missiles:
            if missile.exploded:
                continue
            if rects_overlap(
                missile.x, missile.y, cfg.missile_size, enemy.x, enemy.y, cfg.body_size
            ):
                state.missiles.remove(missile)
                self._hide_enemy()
                return HIT
        return ""

    def _respawn_player(self) -> None:
        x, y = self.spawn_position_away_from((self.state.enemy.x, self.state.enemy.y))
        player = self.state.player
        player.x, player.y, player.vx, player.vy = float(x), float(y), 0.0, 0.0
        self.state.teleports += 1

    def _hide_enemy(self) -> None:
        self.state.enemy_visible = False
        self.state.enemy_respawn_frames = self.config.respawn_frames

    def _count_down_respawn(self) -> None:
        """A hidden enemy reappears away from the player when its time is up."""
        state = self.state
        state.enemy_respawn_frames -= 1
        if state.enemy_respawn_frames <= 0:
            x, y = self.spawn_position_away_from((state.player.x, state.player.y))
            state.enemy.x, state.enemy.y = float(x), float(y)
            state.enemy_visible = True
            state.teleports += 1

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
