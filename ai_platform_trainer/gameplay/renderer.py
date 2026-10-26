"""
Draws the Play world: a dark arena with a starfield, the two ships, missiles
with curved trails and glows, impact effects and the optional AI overlay.

The world is drawn in layers. The arena background and the stars go straight
onto the world surface. Everything else is drawn onto one transparent overlay
the size of the world. The overlay is cleared every frame and blitted onto the
world once, so alpha is never lost. The HUD is drawn on top afterwards by the
game mode.
"""

import logging
import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import pygame

from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.gameplay.effects import Effects, radial_glow
from ai_platform_trainer.utils.sprite_manager import SpriteManager

Point = Tuple[float, float]

# Background
SKY_TOP = (14, 20, 56)  # Deep navy
SKY_BOTTOM = (3, 4, 12)  # Near black
GRID_COLOUR = (120, 150, 255, 26)
GRID_SPACING = 50
STAR_COUNT = 120
STAR_SEED = 1977

# Missiles
# The 32x48 sprite scaled to about the length of the old triangle
MISSILE_SPRITE_SIZE = (14, 21)
TRAIL_LENGTH = 24
TRAIL_COLOUR = (255, 226, 150)
MISSILE_GLOW_COLOUR = (255, 180, 60)
MISSILE_EXPLOSION_COLOUR = (255, 150, 40)
# A missile that moves this far in one frame has wrapped round the screen
WRAP_JUMP_PX = 40

# Ships
ENGINE_GLOW_COLOUR = (90, 200, 255)
ENEMY_GLOW_COLOUR = (255, 70, 70)
# A player that moves this far in one frame has respawned or wrapped
MAX_PLAYER_STEP_PX = 20

# AI overlay
AI_LINE_COLOUR = (140, 255, 170, 255)
AI_LINE_DIM_COLOUR = (140, 255, 170, 110)
AI_ENEMY_COLOUR = (255, 170, 120, 255)
AI_LABEL_COLOUR = (210, 255, 220)
MISSILE_LABEL = "SAC"
ENEMY_LABEL = "AI Agent (PPO)"

# The arena's missile counts its explosion in frames, not milliseconds
ARENA_EXPLOSION_FRAMES = ArenaConfig().explosion_frames


def facing_for_heading(vx: float, vy: float) -> float:
    """
    Degrees to rotate an upward-pointing sprite so its nose follows (vx, vy).

    This is the same convention as PlayerPlay.facing_angle: 0 points up, and
    positive angles turn counter-clockwise. A zero heading gives 0.
    """
    if vx == 0 and vy == 0:
        return 0.0
    return -math.degrees(math.atan2(vx, -vy))


def build_arena_background(width: int, height: int) -> pygame.Surface:
    """A navy-to-black gradient, a faint grid and a soft vignette, built once."""
    surface = pygame.Surface((width, height))

    top, bottom = SKY_TOP, SKY_BOTTOM
    for y in range(height):
        t = y / max(1, height - 1)
        colour = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        pygame.draw.line(surface, colour, (0, y), (width - 1, y))

    grid = pygame.Surface((width, height), pygame.SRCALPHA)
    for x in range(0, width, GRID_SPACING):
        pygame.draw.line(grid, GRID_COLOUR, (x, 0), (x, height - 1))
    for y in range(0, height, GRID_SPACING):
        pygame.draw.line(grid, GRID_COLOUR, (0, y), (width - 1, y))
    surface.blit(grid, (0, 0))

    surface.blit(build_vignette(width, height), (0, 0))
    return surface


def build_vignette(width: int, height: int) -> pygame.Surface:
    """A black layer that is clear in the middle and dark at the corners."""
    # Worked out at an eighth of the size and scaled up smoothly: it is cheap
    # and the result is soft, with no visible steps.
    small_w, small_h = max(2, width // 8), max(2, height // 8)
    small = pygame.Surface((small_w, small_h), pygame.SRCALPHA)
    for x in range(small_w):
        nx = (x + 0.5) / small_w * 2.0 - 1.0
        for y in range(small_h):
            ny = (y + 0.5) / small_h * 2.0 - 1.0
            distance = math.sqrt((nx * nx + ny * ny) / 2.0)  # 0 centre, 1 corner
            strength = max(0.0, min(1.0, (distance - 0.55) / 0.45))
            small.set_at((x, y), (0, 0, 0, int(150 * strength**1.6)))
    return pygame.transform.smoothscale(small, (width, height))


def make_starfield(
    width: int, height: int, count: int = STAR_COUNT
) -> List[Tuple[int, int, int, float, float]]:
    """Fixed star positions, brightnesses and twinkle phases, from a fixed seed."""
    rng = random.Random(STAR_SEED)
    stars = []
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        brightness = rng.randint(90, 255)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.02, 0.06)  # radians per frame: a slow twinkle
        stars.append((x, y, brightness, phase, speed))
    return stars


def _centre(pos: Dict[str, float], size: float) -> Point:
    return (pos["x"] + size / 2.0, pos["y"] + size / 2.0)


def _explosion_frame_index(missile: Any) -> int:
    """Which explosion frame to show, read from the missile's own timer."""
    frames = Missile.EXPLOSION_FRAMES
    start = getattr(missile, "explosion_start_time", None)
    if start is not None:
        # The game's Missile records its explosion start in milliseconds
        progress = (pygame.time.get_ticks() - start) / Missile.EXPLOSION_DURATION_MS
    else:
        # The arena's missile counts its explosion in frames
        progress = getattr(missile, "explosion_age", 0) / ARENA_EXPLOSION_FRAMES
    return min(frames - 1, max(0, int(progress * frames)))


def _draw_arrow(
    surface: pygame.Surface,
    start: Point,
    heading: Point,
    length: float,
    colour: Sequence[int],
    width: int,
) -> None:
    """A solid arrow from start along heading, with a two-stroke head."""
    hx, hy = heading
    magnitude = math.hypot(hx, hy)
    if magnitude == 0:
        return
    ux, uy = hx / magnitude, hy / magnitude
    tip = (start[0] + ux * length, start[1] + uy * length)
    pygame.draw.line(surface, colour, start, tip, width)
    # The two head strokes sit 30 degrees either side of the reverse heading
    for angle in (math.radians(150), math.radians(-150)):
        bx, by = -ux, -uy
        rx = bx * math.cos(angle) - by * math.sin(angle)
        ry = bx * math.sin(angle) + by * math.cos(angle)
        head = (tip[0] + rx * 6, tip[1] + ry * 6)
        pygame.draw.line(surface, colour, tip, head, width)


def _draw_dashed_line(
    surface: pygame.Surface,
    start: Point,
    end: Point,
    colour: Sequence[int],
    dash: float = 6.0,
    gap: float = 5.0,
) -> None:
    """A dashed line from start to end."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    along = 0.0
    while along < length:
        stop = min(along + dash, length)
        p0 = (start[0] + ux * along, start[1] + uy * along)
        p1 = (start[0] + ux * stop, start[1] + uy * stop)
        pygame.draw.line(surface, colour, p0, p1, 1)
        along += dash + gap


class _MissileTrail:
    """What the renderer remembers about one missile between frames."""

    __slots__ = ("missile", "points", "burst_shown")

    def __init__(self, missile: Any) -> None:
        # Holding the missile itself means its id() cannot be reused while tracked
        self.missile = missile
        self.points: Deque[Point] = deque(maxlen=TRAIL_LENGTH)
        self.burst_shown = False


class Renderer:
    def __init__(self, world: pygame.Surface, display) -> None:
        """
        Initialize the Renderer.

        Args:
            world: The fixed-size surface the game world is drawn on
            display: DisplayManager, used to scale the world onto the window
        """
        self.screen = world
        self.display = display
        width, height = world.get_size()

        self.sprite_manager = SpriteManager()
        self.effects = Effects()

        # Optional effects, and the AI overlay that Play mode toggles with V
        self.enable_effects = True
        self.show_ai_overlay = False
        self.frame_count = 0

        # Built once: the arena and the fixed starfield
        self._background = build_arena_background(width, height)
        self._stars = make_starfield(width, height)

        # Everything above the background is drawn here, then composited once
        self.overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        self._label_font = pygame.font.Font(None, 18)

        self._missile_trails: Dict[int, _MissileTrail] = {}
        self._last_player_position: Optional[Point] = None
        self._enemy_facing = 0.0  # Holds the last heading while the enemy stands still

    def render(
        self,
        player,
        enemy,
        learning_mode_manager=None,
    ) -> None:
        """
        Render the game elements on the screen.

        Args:
            player: Player instance
            enemy: Enemy instance
            learning_mode_manager: Learning mode manager for UI rendering
        """
        try:
            # Update frame counter for animations
            self.frame_count += 1

            # The game world is drawn at its fixed size, then scaled to the window
            self.screen.blit(self._background, (0, 0))
            self._draw_stars()

            self.overlay.fill((0, 0, 0, 0))
            self._render_game(player, enemy)
            self.screen.blit(self.overlay, (0, 0))

            # Render mode-specific UI overlay, if this mode has one
            if learning_mode_manager:
                learning_mode_manager.draw_mode_info(self.screen)

            logging.debug("Game elements rendered.")
            self.display.present(self.screen)
            logging.debug("Frame updated on display.")

        except Exception as e:
            logging.error(f"Error during rendering: {e}")

    def _draw_stars(self) -> None:
        """Stars on the world, each twinkling slowly on the frame counter."""
        t = self.frame_count
        for x, y, brightness, phase, speed in self._stars:
            level = 0.6 + 0.4 * math.sin(t * speed + phase)
            value = int(brightness * level)
            self.screen.set_at((x, y), (value, value, min(255, value + 20)))

    def _render_game(self, player, enemy) -> None:
        """
        Draw the game elements onto the overlay, back to front.

        Args:
            player: Player instance
            enemy: Enemy instance
        """
        self.effects.update()

        has_player = (
            player is not None
            and hasattr(player, "position")
            and hasattr(player, "size")
        )
        has_enemy = (
            enemy is not None
            and hasattr(enemy, "pos")
            and hasattr(enemy, "size")
            and enemy.visible
        )
        missiles = list(getattr(player, "missiles", [])) if has_player else []

        if has_player:
            self._render_engine_glow(player)
            self._render_player(player)
        if has_enemy:
            self._render_enemy(enemy)

        self._render_missiles(missiles)

        # Render particle effects if enabled
        if self.enable_effects:
            self.effects.draw(self.overlay)

        if self.show_ai_overlay:
            self._render_ai_overlay(missiles, enemy if has_enemy else None)

    def _render_player(self, player) -> None:
        """
        Render the player entity with sprites.

        Args:
            player: Player instance
        """
        size = int(player.size)
        center = _centre(player.position, size)
        self.sprite_manager.render_centered(
            self.overlay,
            "player",
            center,
            (size, size),
            rotation=getattr(player, "facing_angle", 0),
        )

    def _render_engine_glow(self, player) -> None:
        """A small pulsing glow behind the player while it is moving."""
        x, y = float(player.position["x"]), float(player.position["y"])
        last = self._last_player_position
        self._last_player_position = (x, y)
        if last is None or not self.enable_effects:
            return

        dx, dy = x - last[0], y - last[1]
        step = math.hypot(dx, dy)
        if step < 0.5 or step > MAX_PLAYER_STEP_PX:
            return  # Standing still, or a respawn or screen wrap

        size = int(player.size)
        cx, cy = _centre(player.position, size)
        # Behind the ship, against the direction of travel
        offset = size / 2.0 + 2  # Just past the tail of the sprite
        gx = cx - dx / step * offset
        gy = cy - dy / step * offset
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.5)
        glow = radial_glow(10 + 3 * round(pulse), ENGINE_GLOW_COLOUR, 230)
        self.overlay.blit(
            glow,
            glow.get_rect(center=(int(round(gx)), int(round(gy)))),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    def _render_enemy(self, enemy) -> None:
        """
        Render the enemy entity with sprites, turned to face where it last moved.

        Args:
            enemy: Enemy instance
        """
        vx, vy = getattr(enemy, "_last_move", (0.0, 0.0))
        if vx or vy:
            self._enemy_facing = facing_for_heading(vx, vy)

        size = int(enemy.size)
        center = _centre(enemy.pos, size)

        # The enemy fades in after it respawns
        alpha = 255
        if getattr(enemy, "fading_in", False):
            alpha = int(getattr(enemy, "fade_alpha", 255))
        if alpha >= 255 and self.enable_effects:
            self._draw_enemy_glow(center, size)

        self.sprite_manager.render_centered(
            self.overlay,
            "enemy",
            center,
            (size, size),
            rotation=self._enemy_facing,
            alpha=alpha,
        )

    def _draw_enemy_glow(self, center: Point, size: int) -> None:
        """A subtle red halo that breathes slowly around the enemy."""
        pulse = 0.5 + 0.5 * math.sin(self.frame_count * 0.06)
        level = int(round(pulse * 4))  # Five steps, so only five glows are cached
        glow = radial_glow(
            size / 2.0 + 8 + 2 * level, ENEMY_GLOW_COLOUR, 40 + 15 * level
        )
        self.overlay.blit(
            glow,
            glow.get_rect(center=(int(round(center[0])), int(round(center[1])))),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    def _render_missiles(self, missiles: List[Any]) -> None:
        """Trails, glows, sprites and explosions for the player's missiles."""
        # Forget the trail of any missile that is gone
        live = {id(missile): missile for missile in missiles}
        for key in list(self._missile_trails):
            if live.get(key) is not self._missile_trails[key].missile:
                del self._missile_trails[key]

        for missile in missiles:
            trail = self._missile_trails.get(id(missile))
            if trail is None:
                trail = _MissileTrail(missile)
                self._missile_trails[id(missile)] = trail

            center = _centre(missile.pos, missile.size)
            if missile.exploded:
                self._draw_explosion(missile, trail, center)
                continue

            self._record_trail_point(trail, center)
            if self.enable_effects:
                self._draw_trail(trail.points)
                self._draw_missile_glow(center)
            self.sprite_manager.render_centered(
                self.overlay,
                "missile",
                center,
                MISSILE_SPRITE_SIZE,
                rotation=facing_for_heading(missile.vx, missile.vy),
            )

    def _record_trail_point(self, trail: _MissileTrail, center: Point) -> None:
        if trail.points:
            last_x, last_y = trail.points[-1]
            if math.hypot(center[0] - last_x, center[1] - last_y) > WRAP_JUMP_PX:
                trail.points.clear()  # The missile wrapped to the other side
        trail.points.append(center)

    def _draw_trail(self, points: Deque[Point]) -> None:
        """A polyline behind the missile that fades and thins toward the tail."""
        count = len(points)
        if count < 2:
            return
        pts = [(int(round(x)), int(round(y))) for x, y in points]
        for i in range(1, count):
            fade = i / (count - 1)  # 0 at the tail, 1 at the head
            colour = (*TRAIL_COLOUR, int(235 * fade**1.2))
            width = 1 + int(round(3 * fade))
            pygame.draw.line(self.overlay, colour, pts[i - 1], pts[i], width)

    def _draw_missile_glow(self, center: Point) -> None:
        glow = radial_glow(10, MISSILE_GLOW_COLOUR, 150)
        self.overlay.blit(
            glow,
            glow.get_rect(center=(int(round(center[0])), int(round(center[1])))),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    def _draw_explosion(
        self, missile: Any, trail: _MissileTrail, center: Point
    ) -> None:
        """The explosion frames for a missile that ran out, plus a short burst."""
        if self.enable_effects and not trail.burst_shown:
            trail.burst_shown = True
            self.effects.burst(
                center[0], center[1], MISSILE_EXPLOSION_COLOUR, 14, 2.5, 20
            )

        size = missile.size * 4
        frames = self.sprite_manager.load_animation(
            "explosion", (size, size), frames=Missile.EXPLOSION_FRAMES
        )
        frame = frames[_explosion_frame_index(missile)]
        self.overlay.blit(
            frame, frame.get_rect(center=(int(round(center[0])), int(round(center[1]))))
        )

    def _render_ai_overlay(self, missiles: List[Any], enemy) -> None:
        """Headings, lines of sight and labels for the AI, drawn over the game."""
        target = _centre(enemy.pos, enemy.size) if enemy is not None else None

        for missile in missiles:
            if missile.exploded:
                continue
            center = _centre(missile.pos, missile.size)
            _draw_arrow(
                self.overlay,
                center,
                (missile.vx, missile.vy),
                26,
                AI_LINE_COLOUR,
                2,
            )
            if target is not None:
                _draw_dashed_line(self.overlay, center, target, AI_LINE_DIM_COLOUR)
            self._draw_label(MISSILE_LABEL, center[0] + 10, center[1] - 24)

        if enemy is not None:
            center = _centre(enemy.pos, enemy.size)
            vx, vy = getattr(enemy, "_last_move", (0.0, 0.0))
            if vx or vy:
                _draw_arrow(self.overlay, center, (vx, vy), 40, AI_ENEMY_COLOUR, 3)
            self._draw_label(ENEMY_LABEL, enemy.pos["x"], enemy.pos["y"] - 20)

    def _draw_label(self, text: str, x: float, y: float) -> None:
        """Small text on the overlay, kept inside the world."""
        label = self._label_font.render(text, True, AI_LABEL_COLOUR)
        width, height = self.screen.get_size()
        left = int(min(max(2, x), width - label.get_width() - 2))
        top = int(min(max(2, y), height - label.get_height() - 2))
        self.overlay.blit(label, (left, top))
