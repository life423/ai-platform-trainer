import pygame
import logging

# Import sprite manager for entity rendering
from ai_platform_trainer.utils.sprite_manager import SpriteManager


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        """
        Initialize the Renderer.

        Args:
            screen: Pygame display surface
        """
        self.screen = screen
        self.BACKGROUND_COLOR = (135, 206, 235)  # Light blue

        # Initialize sprite manager
        self.sprite_manager = SpriteManager()

        # Optional effects
        self.enable_effects = True
        self.frame_count = 0
        self.particle_effects = []

    def render(self, menu, player, enemy, menu_active: bool, game_mode: str = None, learning_mode_manager=None) -> None:
        """
        Render the game elements on the screen.

        Args:
            menu: Menu instance
            player: Player instance
            enemy: Enemy instance
            menu_active: Boolean indicating if the menu is active
            game_mode: Current game mode (for special rendering)
            learning_mode_manager: Learning mode manager for UI rendering
        """
        try:
            # Clear screen with background color
            self.screen.fill(self.BACKGROUND_COLOR)

            # Update frame counter for animations
            self.frame_count += 1

            if menu_active:
                # Draw menu
                menu.draw(self.screen)
                logging.debug("Menu rendered.")
            else:
                # Render game elements
                self._render_game(player, enemy)
                
                # Render learning mode UI if in learning mode
                if game_mode == "play_learning" and learning_mode_manager:
                    learning_mode_manager.draw_mode_info(self.screen)
                
                logging.debug("Game elements rendered.")

            # Update display
            pygame.display.flip()
            logging.debug("Frame updated on display.")

        except Exception as e:
            logging.error(f"Error during rendering: {e}")

    def _render_game(self, player, enemy) -> None:
        """
        Render the game elements during gameplay.

        Args:
            player: Player instance
            enemy: Enemy instance
        """
        # Draw player with sprite
        if hasattr(player, 'position') and hasattr(player, 'size'):
            self._render_player(player)

            # Render player missiles
            if hasattr(player, 'missiles'):
                for missile in player.missiles:
                    self._render_missile(missile)

        # Draw enemy with sprite
        if hasattr(enemy, 'pos') and hasattr(enemy, 'size') and enemy.visible:
            self._render_enemy(enemy)

        # Render particle effects if enabled
        if self.enable_effects:
            self._update_and_render_effects()

    def _render_player(self, player) -> None:
        """
        Render the player entity with sprites.

        Args:
            player: Player instance
        """
        # Determine sprite size
        size = (player.size, player.size)

        # Render the player sprite
        self.sprite_manager.render(
            screen=self.screen,
            entity_type="player",
            position=player.position,
            size=size
        )

    def _render_enemy(self, enemy) -> None:
        """
        Render the enemy entity with sprites.

        Args:
            enemy: Enemy instance
        """
        # Determine sprite size
        size = (enemy.size, enemy.size)

        # Check if the enemy is fading in
        alpha = 255
        if hasattr(enemy, 'fading_in') and enemy.fading_in:
            alpha = enemy.fade_alpha

        # Render the enemy sprite
        sprite = self.sprite_manager.load_sprite("enemy", size)
        sprite.set_alpha(alpha)
        self.screen.blit(sprite, (enemy.pos["x"], enemy.pos["y"]))

    def _render_missile(self, missile) -> None:
        """
        Render a missile entity using its own draw method.

        Args:
            missile: Missile instance
        """
        # Missiles have their own draw method that renders them as triangles
        # pointing in their movement direction
        missile.draw(self.screen)

        # Add a trail effect if effects are enabled
        if self.enable_effects and self.frame_count % 2 == 0:
            self._add_missile_trail(missile)

    def _add_missile_trail(self, missile) -> None:
        """
        Add a particle effect trail behind a missile.

        Args:
            missile: Missile instance
        """
        if not hasattr(missile, 'pos'):
            return

        # Create a small particle effect behind the missile
        x = missile.pos["x"] + missile.size // 2
        y = missile.pos["y"] + missile.size // 2

        # Trail particles
        import random
        for _ in range(2):
            # Random offset
            offset_x = random.randint(-3, 3)
            offset_y = random.randint(-3, 3)

            # Random size
            size = random.randint(2, 5)

            # Random lifetime
            lifetime = random.randint(5, 15)

            # Create particle
            particle = {
                'x': x + offset_x,
                'y': y + offset_y,
                'size': size,
                'color': (255, 255, 200, 200),  # Yellowish with alpha
                'lifetime': lifetime,
                'max_lifetime': lifetime
            }

            self.particle_effects.append(particle)

    def _update_and_render_effects(self) -> None:
        """Update and render all particle effects."""
        # Update particles
        updated_particles = []
        for particle in self.particle_effects:
            # Decrease lifetime
            particle['lifetime'] -= 1

            # Skip dead particles
            if particle['lifetime'] <= 0:
                continue

            # Calculate alpha based on remaining lifetime
            alpha = int(255 * (particle['lifetime'] / particle['max_lifetime']))
            color = list(particle['color'])
            if len(color) > 3:
                color[3] = min(color[3], alpha)
            else:
                color.append(alpha)

            # Draw particle
            pygame.draw.circle(
                self.screen,
                color,
                (int(particle['x']), int(particle['y'])),
                particle['size']
            )

            # Keep particle for next frame
            updated_particles.append(particle)

        # Replace particle list with updated one
        self.particle_effects = updated_particles
