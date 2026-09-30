"""Arena geometry must match pygame and the Play mode wrap rule exactly."""
import math
import random

import pygame

from ai_platform_trainer.arena.geometry import (
    OCTANT_DIRECTIONS,
    normalize_angle,
    quantize_direction,
    rects_overlap,
    torus_delta,
    wrap_coordinate,
)


def test_rects_overlap_matches_pygame_colliderect():
    rng = random.Random(0)
    for _ in range(20000):
        size = rng.choice((10, 50))
        ax, ay = rng.uniform(-70.0, 130.0), rng.uniform(-70.0, 130.0)
        bx, by = rng.uniform(-70.0, 130.0), rng.uniform(-70.0, 130.0)
        expected = pygame.Rect(ax, ay, size, size).colliderect(
            pygame.Rect(bx, by, 50, 50)
        )
        assert rects_overlap(ax, ay, size, bx, by, 50) == expected


def test_touching_edges_do_not_collide():
    assert not rects_overlap(0, 0, 50, 50, 0, 50)
    assert not rects_overlap(0, 0, 50, 0, 50, 50)
    assert rects_overlap(0, 0, 50, 49.9, 0, 50)


def test_wrap_coordinate_matches_play_mode_rule():
    assert wrap_coordinate(-50.0, 1470, 50) == -50.0
    assert wrap_coordinate(-50.5, 1470, 50) == 1470.0
    assert wrap_coordinate(1470.0, 1470, 50) == 1470.0
    assert wrap_coordinate(1470.5, 1470, 50) == -50.0


def test_torus_delta_takes_the_short_way_round():
    period = 1520.0
    assert torus_delta(10.0, 30.0, period) == 20.0
    assert torus_delta(1500.0, 10.0, period) == 30.0
    assert torus_delta(10.0, 1500.0, period) == -30.0
    rng = random.Random(1)
    for _ in range(2000):
        a, b = rng.uniform(-50.0, 1470.0), rng.uniform(-50.0, 1470.0)
        delta = torus_delta(a, b, period)
        assert abs(delta) <= period / 2.0 + 1e-9
        remainder = (a + delta - b) % period
        assert min(remainder, period - remainder) < 1e-6


def test_quantize_direction_keeps_keyboard_directions():
    for dx, dy in OCTANT_DIRECTIONS:
        assert quantize_direction(dx * 3.0, dy * 3.0) == (dx, dy)
    assert quantize_direction(0.0, 0.0) == (0, 0)
    assert quantize_direction(10.0, 1.0) == (1, 0)
    assert quantize_direction(-1.0, -10.0) == (0, -1)


def test_normalize_angle_range():
    for angle in (-10.0, -math.pi, 0.0, 3.5, 12.0):
        assert -math.pi <= normalize_angle(angle) <= math.pi
