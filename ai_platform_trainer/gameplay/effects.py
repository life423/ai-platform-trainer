"""
Impact effects for Play mode: particle bursts, expanding shockwave rings and
soft radial glows.

Effects are advanced once per frame with update() and drawn onto a transparent
overlay with draw(). Timers count frames at the game's fixed 60 FPS. Nothing
here touches game state; it only draws.
"""

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import pygame

Colour = Tuple[int, int, int]

# Each frame a particle keeps this share of its speed, so bursts slow down
PARTICLE_DRAG = 0.92


def _rgb(colour: Sequence[int]) -> Colour:
    return (int(colour[0]), int(colour[1]), int(colour[2]))


@lru_cache(maxsize=256)
def _cached_glow(radius: int, colour: Colour, strength: int) -> pygame.Surface:
    size = 2 * radius + 1
    glow = pygame.Surface((size, size), pygame.SRCALPHA)
    centre = (radius, radius)
    r, g, b = colour
    # The largest, faintest circle is drawn first and the smaller, brighter
    # ones overwrite it, which gives a radial gradient without per-pixel maths.
    for i in range(radius, 0, -1):
        falloff = 1.0 - i / radius  # 0 at the edge, near 1 at the centre
        alpha = int(strength * falloff * falloff)
        pygame.draw.circle(glow, (r, g, b, alpha), centre, i)
    return glow


def radial_glow(
    radius: float, colour: Sequence[int], strength: int = 255
) -> pygame.Surface:
    """
    A cached radial-gradient sprite in one colour: strongest at the centre and
    fully transparent at its edge. Blit it with BLEND_RGBA_ADD to add light.

    The same arguments return the same surface, so callers must not change it.
    """
    return _cached_glow(
        max(2, int(round(radius))), _rgb(colour), max(0, min(255, int(strength)))
    )


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    colour: Colour
    life: int
    max_life: int
    size: float


@dataclass
class Ring:
    x: float
    y: float
    colour: Colour
    max_radius: float
    life: int
    age: int = 0

    @property
    def progress(self) -> float:
        """0 when the ring starts, 1 when it has finished."""
        return min(1.0, self.age / self.life)

    @property
    def radius(self) -> float:
        """Grows quickly at first, then slows as it fades."""
        return self.max_radius * (1.0 - (1.0 - self.progress) ** 2)


class Effects:
    """Particle bursts and shockwave rings, advanced per frame and drawn on an overlay."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._particles: List[Particle] = []
        self._rings: List[Ring] = []

    @property
    def particles(self) -> List[Particle]:
        """The live particles (a copy of the list)."""
        return list(self._particles)

    @property
    def rings(self) -> List[Ring]:
        """The live rings (a copy of the list)."""
        return list(self._rings)

    def burst(
        self,
        x: float,
        y: float,
        colour: Sequence[int],
        count: int,
        speed: float,
        life: int,
    ) -> None:
        """
        Throw count particles out of (x, y) in random directions.

        Each particle starts at up to speed pixels per frame, slows with drag,
        and lives for life frames.
        """
        colour = _rgb(colour)
        life = max(1, int(life))
        for _ in range(max(0, int(count))):
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            magnitude = speed * self._rng.uniform(0.4, 1.0)
            self._particles.append(
                Particle(
                    x=float(x),
                    y=float(y),
                    vx=magnitude * math.cos(angle),
                    vy=magnitude * math.sin(angle),
                    colour=colour,
                    life=life,
                    max_life=life,
                    size=self._rng.uniform(1.5, 3.0),
                )
            )

    def ring(
        self,
        x: float,
        y: float,
        colour: Sequence[int],
        max_radius: float,
        life: int,
    ) -> None:
        """A shockwave that grows from (x, y) to max_radius over life frames."""
        self._rings.append(
            Ring(
                x=float(x),
                y=float(y),
                colour=_rgb(colour),
                max_radius=float(max_radius),
                life=max(1, int(life)),
            )
        )

    def update(self) -> None:
        """Advance every effect by one frame and drop the finished ones."""
        for p in self._particles:
            p.x += p.vx
            p.y += p.vy
            p.vx *= PARTICLE_DRAG
            p.vy *= PARTICLE_DRAG
            p.life -= 1
        self._particles = [p for p in self._particles if p.life > 0]

        for ring in self._rings:
            ring.age += 1
        self._rings = [ring for ring in self._rings if ring.age < ring.life]

    def clear(self) -> None:
        """Remove every effect at once."""
        self._particles = []
        self._rings = []

    def draw(self, overlay: pygame.Surface) -> None:
        """
        Draw the live effects onto overlay, an SRCALPHA surface. Alpha is
        written as-is, so the overlay must be composited onto the world.
        """
        for p in self._particles:
            # Each particle is a soft glow, dimming as it ages. Its brightness is
            # quantised to eight steps so the glow sprites can be cached.
            level = max(1, int(round(8 * p.life / p.max_life)))
            glow = radial_glow(p.size * 2, p.colour, 255 * level // 8)
            overlay.blit(
                glow,
                glow.get_rect(center=(int(round(p.x)), int(round(p.y)))),
                special_flags=pygame.BLEND_RGBA_ADD,
            )

        for ring in self._rings:
            radius = int(round(ring.radius))
            if radius < 1:
                continue
            fade = 1.0 - ring.progress
            width = min(radius, 1 + int(round(2 * fade)))
            r, g, b = ring.colour
            pygame.draw.circle(
                overlay,
                (r, g, b, int(255 * fade)),
                (int(round(ring.x)), int(round(ring.y))),
                radius,
                width,
            )
