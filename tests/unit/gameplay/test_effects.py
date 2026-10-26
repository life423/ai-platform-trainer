"""Impact effects: particle bursts, shockwave rings, glows and compositing."""

import pygame

from ai_platform_trainer.gameplay.effects import Effects, radial_glow


def test_burst_spawns_the_requested_count_with_the_requested_life():
    effects = Effects(seed=1)

    effects.burst(100, 120, (255, 0, 0), count=25, speed=3.0, life=12)

    particles = effects.particles
    assert len(particles) == 25
    assert all(p.life == 12 and p.max_life == 12 for p in particles)
    assert all((p.x, p.y) == (100.0, 120.0) for p in particles)


def test_burst_particles_expire_after_their_life():
    effects = Effects(seed=1)
    effects.burst(0, 0, (255, 255, 255), count=10, speed=2.0, life=5)

    for _ in range(4):
        effects.update()
    assert len(effects.particles) == 10

    effects.update()
    assert effects.particles == []


def test_ring_radius_grows_and_the_ring_then_expires():
    effects = Effects(seed=1)
    effects.ring(50, 50, (0, 255, 0), max_radius=40, life=10)

    radii = []
    for _ in range(9):
        effects.update()
        radii.append(effects.rings[0].radius)

    assert all(later > earlier for earlier, later in zip(radii, radii[1:]))
    assert radii[-1] < 40

    effects.update()
    assert effects.rings == []


def test_overlay_composites_with_alpha_and_leaves_clear_pixels_alone():
    world = pygame.Surface((32, 32))
    world.fill((0, 0, 255))
    overlay = pygame.Surface((32, 32), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))

    # Half-transparent red over blue: the result is halfway between the two
    overlay.set_at((10, 10), (255, 0, 0, 128))
    world.blit(overlay, (0, 0))

    blended = world.get_at((10, 10))
    assert abs(blended.r - 128) <= 2
    assert blended.g == 0
    assert abs(blended.b - 127) <= 2
    # A clear overlay pixel does not change the world at all
    assert world.get_at((0, 0))[:3] == (0, 0, 255)


def test_effects_drawn_on_an_overlay_show_through_it_with_alpha():
    overlay = pygame.Surface((64, 64), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 0))
    effects = Effects(seed=1)
    effects.ring(32, 32, (255, 0, 0), max_radius=20, life=10)
    effects.update()

    effects.draw(overlay)

    bounds = overlay.get_bounding_rect()
    assert bounds.width > 0 and bounds.height > 0
    # Some drawn pixels are partly transparent, not solid
    alphas = {overlay.get_at((x, y)).a for x in range(64) for y in range(64)}
    assert any(0 < a < 255 for a in alphas)


def test_radial_glow_is_brightest_at_the_centre_and_clear_at_the_edge():
    glow = radial_glow(8, (255, 200, 0), strength=255)
    w, h = glow.get_size()

    centre = glow.get_at((w // 2, h // 2))
    assert centre.a > 150
    assert centre[:3] == (255, 200, 0)
    assert glow.get_at((0, 0)).a == 0


def test_radial_glow_is_cached_by_its_arguments():
    assert radial_glow(9, (90, 200, 255), 200) is radial_glow(9, (90, 200, 255), 200)
    assert radial_glow(9, (90, 200, 255), 200) is not radial_glow(
        10, (90, 200, 255), 200
    )
