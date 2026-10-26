"""
Play mode: the player against the AI Agent, with a card showing what drives
the enemy.
"""
import logging
import random

import pygame

from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.sim import spawn_away
from ai_platform_trainer.entities.enemy_agent import create_trained_enemy
from ai_platform_trainer.entities.enemy_play import EnemyPlay

# Color constants
COLOR_TEXT_PRIMARY = (255, 255, 255)  # White
COLOR_SELECTED = (255, 215, 0)  # Gold

# The HUD: rounded, translucent navy cards
CARD_FILL = (20, 30, 70, 215)
CARD_BORDER = (200, 220, 255)
CARD_RADIUS = 10
CARD_PADDING = 14
CARD_MARGIN = 12
CARD_MIN_WIDTH = 240
CARD_MAX_WIDTH = 330

CONTROLS_TEXT = (
    "Move: WASD or Arrows     Fire: Space     V: AI overlay     "
    "Menu: M     Fullscreen: F     Quit: Esc"
)

# Impact effects: particles and a shockwave, drawn by the renderer's Effects
CATCH_COLOUR = (255, 90, 70)  # The enemy caught the player
HIT_COLOUR = (255, 215, 0)  # The player's missile hit the enemy


def draw_card(screen: pygame.Surface, rect) -> None:
    """Draw a rounded, translucent card at rect (x, y, width, height)."""
    card = pygame.Surface(rect[2:], pygame.SRCALPHA)
    card_rect = card.get_rect()
    pygame.draw.rect(card, CARD_FILL, card_rect, border_radius=CARD_RADIUS)
    pygame.draw.rect(
        card, CARD_BORDER, card_rect, width=2, border_radius=CARD_RADIUS
    )
    screen.blit(card, rect[:2])


def _centre_of(entity) -> tuple:
    """The centre of an entity, from its top-left corner (pos, or position)."""
    origin = entity.pos if hasattr(entity, "pos") else entity.position
    return (origin["x"] + entity.size / 2, origin["y"] + entity.size / 2)


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list:
    """Split text into lines no wider than max_width. A long word stays whole."""
    lines = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current or not lines:
        lines.append(current)
    return lines


class PlayLearningMode:
    """Play mode against the selected enemy."""

    def __init__(self, game):
        """
        Initialize Play mode against the AI Agent.

        Raises:
            MissingModelError: if the AI Agent has not been trained yet.
        """
        self.game = game
        self.space_pressed_last_frame = False
        self.v_pressed_last_frame = False

        # Spawns and respawns follow the arena's rule, the one the models trained on
        self.arena_config = ArenaConfig(
            width=self.game.screen_width, height=self.game.screen_height
        )
        self._rng = random.Random()

        self.learning_enemy: EnemyPlay = create_trained_enemy(
            self.game.screen_width,
            self.game.screen_height,
            player_provider=lambda: self.game.player,
        )

        # Replace the regular enemy with our learning AI
        self.game.enemy = self.learning_enemy

        # Fonts for the HUD: pygame's default font, which has no emoji
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        self.text_primary = COLOR_TEXT_PRIMARY
        self.highlight_color = COLOR_SELECTED

        logging.info("Play mode initialized")

    def update(self, current_time: int) -> None:
        """Update learning play mode."""
        # Handle space bar for shooting
        keys = pygame.key.get_pressed()
        if (
            keys[pygame.K_SPACE]
            and not self.space_pressed_last_frame
            and self.game.player
        ):
            self.game.player.shoot_missile(self.learning_enemy.pos)
            logging.debug("Player shot missile at learning AI")
        self.space_pressed_last_frame = keys[pygame.K_SPACE]

        self._toggle_ai_overlay(keys)

        # Update player
        if self.game.player:
            self.game.player.handle_input()

        # Update learning enemy with missile information
        if self.learning_enemy:
            missiles = self.game.player.missiles if self.game.player else []
            self.learning_enemy.update_movement(
                self.game.player.position["x"] if self.game.player else 0,
                self.game.player.position["y"] if self.game.player else 0,
                self.game.player.step if self.game.player else 0,
                current_time,
                missiles,
            )

        # Check collisions
        self._check_collisions(current_time)

        # Update missiles
        if self.game.player:
            # Pass enemy position for smart missile tracking
            enemy_pos = (
                self.learning_enemy.pos
                if self.learning_enemy and self.learning_enemy.visible
                else None
            )
            self.game.player.update_missiles(enemy_pos)

        # Each missile is steered and moved exactly once per frame, inside
        # update_missiles() above, as in arena.sim.missile_phase.

        # Handle respawning
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self._respawn_enemy()

        # Handle enemy fade-in
        if (
            self.learning_enemy
            and hasattr(self.learning_enemy, "fading_in")
            and self.learning_enemy.fading_in
        ):
            self.learning_enemy.update_fade_in(current_time)

    def _check_collisions(self, current_time: int):
        """Check for collisions and update learning AI."""
        if not self.game.player or not self.learning_enemy:
            return

        # Player-Enemy collision. A catch respawns the player away from the enemy
        # and the enemy stays put. As in arena.sim.collision_phase, a catch ends
        # the frame's collision checks, so a missile hit does not count too.
        if self._entities_collide(self.game.player, self.learning_enemy):
            logging.info("Learning AI caught the player! The player respawns.")
            self.learning_enemy.on_hit_player()  # Notify AI of success
            self._show_impact(
                _centre_of(self.game.player),
                CATCH_COLOUR,
                burst_count=40,
                ring_radius=90,
            )
            self._respawn_player()
            return

        # Missile-Enemy collisions
        for missile in self.game.player.missiles[:]:
            # An already-expired missile is just playing out its explosion
            # animation in place - it shouldn't still be able to score a hit.
            if missile.exploded:
                continue
            if self._missile_enemy_collide(missile, self.learning_enemy):
                logging.info("Player missile hit learning AI!")
                self.learning_enemy.on_hit_by_missile()  # Notify AI of failure
                self.game.player.missiles.remove(missile)
                self.learning_enemy.hide()
                self._show_impact(
                    _centre_of(self.learning_enemy),
                    HIT_COLOUR,
                    burst_count=32,
                    ring_radius=80,
                )
                self.game.is_respawning = True
                self.game.respawn_timer = current_time + self.game.respawn_delay
                break

    def _renderer(self):
        """The game's renderer, or None when the game has none (e.g. in tests)."""
        return getattr(self.game, "renderer", None)

    def _toggle_ai_overlay(self, keys) -> None:
        """V shows or hides the AI overlay drawn by the renderer, once per press."""
        v_held = bool(keys[pygame.K_v])
        if v_held and not self.v_pressed_last_frame:
            renderer = self._renderer()
            if renderer is not None:
                renderer.show_ai_overlay = not renderer.show_ai_overlay
        self.v_pressed_last_frame = v_held

    def _show_impact(
        self, centre, colour, burst_count: int, ring_radius: float
    ) -> None:
        """Visual only: a burst of particles and a shockwave at centre."""
        renderer = self._renderer()
        effects = getattr(renderer, "effects", None)
        if effects is None:
            return
        effects.burst(centre[0], centre[1], colour, burst_count, 4.0, 36)
        effects.ring(centre[0], centre[1], colour, ring_radius, 30)

    def _respawn_player(self) -> None:
        """Move the player to a spawn point away from the enemy, by the arena's rule."""
        player = self.game.player
        enemy = self.learning_enemy
        x, y = spawn_away(
            self.arena_config, self._rng, (enemy.pos["x"], enemy.pos["y"])
        )
        player.position["x"], player.position["y"] = x, y
        enemy.on_player_teleport()

    def _entities_collide(self, entity1, entity2) -> bool:
        """Check collision between two entities."""
        if not entity2.visible:
            return False

        rect1 = pygame.Rect(
            entity1.position["x"], entity1.position["y"], entity1.size, entity1.size
        )
        rect2 = pygame.Rect(
            entity2.pos["x"], entity2.pos["y"], entity2.size, entity2.size
        )
        return rect1.colliderect(rect2)

    def _missile_enemy_collide(self, missile, enemy) -> bool:
        """Check collision between missile and enemy."""
        if not enemy.visible:
            return False

        missile_rect = pygame.Rect(
            missile.pos["x"], missile.pos["y"], missile.size, missile.size
        )
        enemy_rect = pygame.Rect(enemy.pos["x"], enemy.pos["y"], enemy.size, enemy.size)
        return missile_rect.colliderect(enemy_rect)

    def _respawn_enemy(self):
        """Respawn the learning enemy away from the player, by the arena's rule."""
        if not self.game.player:
            return

        player = self.game.player
        x, y = spawn_away(
            self.arena_config, self._rng, (player.position["x"], player.position["y"])
        )
        self.learning_enemy.set_position(x, y)
        self.learning_enemy.show(pygame.time.get_ticks())
        self.game.is_respawning = False

        logging.debug(f"Learning AI respawned at ({x}, {y})")

    def render_enemy_panel(self, screen: pygame.Surface) -> None:
        """Top-right card: what drives the enemy, and the score so far."""
        if not self.learning_enemy:
            return
        stats = self.learning_enemy.panel_stats()
        lines = list(stats["summary"]) + [
            f"Caught you: {stats['hits']}",
            f"Your missiles hit it: {stats['deaths']}",
        ]

        # Size the card to its text, then wrap the text to fit the card
        title = self.font_medium.render(stats["title"], True, self.highlight_color)
        widest = max(
            [title.get_width()] + [self.font_small.size(line)[0] for line in lines]
        )
        card_width = min(
            CARD_MAX_WIDTH, max(CARD_MIN_WIDTH, widest + 2 * CARD_PADDING)
        )
        text_width = card_width - 2 * CARD_PADDING
        body = [
            part
            for line in lines
            for part in wrap_text(line, self.font_small, text_width)
        ]

        line_height = self.font_small.get_linesize()
        body_top = CARD_PADDING + title.get_height() + 8
        card_height = body_top + line_height * len(body) + CARD_PADDING
        x = screen.get_width() - card_width - CARD_MARGIN
        y = CARD_MARGIN
        draw_card(screen, (x, y, card_width, card_height))

        screen.blit(title, (x + CARD_PADDING, y + CARD_PADDING))
        for i, line in enumerate(body):
            text = self.font_small.render(line, True, self.text_primary)
            screen.blit(text, (x + CARD_PADDING, y + body_top + i * line_height))

    def draw_mode_info(self, screen: pygame.Surface):
        """The Play HUD: a mode chip, a controls bar, and the enemy card."""
        self._draw_mode_chip(screen)
        self._draw_controls_bar(screen)
        self.render_enemy_panel(screen)

    def _draw_mode_chip(self, screen: pygame.Surface) -> None:
        label = self.font_medium.render("PLAY", True, self.highlight_color)
        chip = (
            CARD_MARGIN,
            CARD_MARGIN,
            label.get_width() + 2 * CARD_PADDING,
            label.get_height() + 10,
        )
        draw_card(screen, chip)
        screen.blit(label, (chip[0] + CARD_PADDING, chip[1] + 5))

    def _draw_controls_bar(self, screen: pygame.Surface) -> None:
        text = self.font_small.render(CONTROLS_TEXT, True, self.text_primary)
        bar_height = text.get_height() + 16
        bar = pygame.Surface((screen.get_width(), bar_height), pygame.SRCALPHA)
        bar.fill(CARD_FILL)
        y = screen.get_height() - bar_height
        screen.blit(bar, (0, y))
        centre = (screen.get_width() // 2, y + bar_height // 2)
        screen.blit(text, text.get_rect(center=centre))
