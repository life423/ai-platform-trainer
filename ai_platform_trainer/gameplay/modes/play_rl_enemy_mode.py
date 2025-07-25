"""
RL Enemy Play Mode

This mode features a true reinforcement learning-trained enemy that has learned
optimal evasion and survival strategies through experience.
"""
import logging
import pygame
import os

from ai_platform_trainer.entities.enemy_rl import EnemyRL
from ai_platform_trainer.entities.player_play import PlayerPlay
from ai_platform_trainer.ai.inference.missile_controller import update_missile_ai


class PlayRLEnemyMode:
    """
    Play mode against a true RL-trained enemy.
    
    This mode showcases genuine machine learning where the enemy has learned
    through trial and error to survive against missile-equipped players.
    """
    
    def __init__(self, game, rl_model_path: str = None):
        """
        Initialize RL enemy play mode.
        
        Args:
            game: Reference to main game instance
            rl_model_path: Path to trained RL enemy model
        """
        self.game = game
        self.space_pressed_last_frame = False
        
        # Default RL model path
        if rl_model_path is None:
            rl_model_path = "models/enemy_rl_sac_final.zip"
        
        # Create RL-trained enemy
        self.rl_enemy = EnemyRL(
            self.game.screen_width, 
            self.game.screen_height,
            rl_model_path=rl_model_path if os.path.exists(rl_model_path) else None,
            fallback_to_scripted=True
        )
        
        # Replace the regular enemy with our RL enemy
        self.game.enemy = self.rl_enemy
        
        # UI elements for RL feedback
        self.font_large = pygame.font.Font(None, 42)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 22)
        
        # Colors for UI
        self.ui_bg_color = (25, 25, 112, 180)  # Semi-transparent navy
        self.text_primary = (255, 255, 255)    # White
        self.text_secondary = (200, 255, 200)  # Light green
        self.rl_indicator_color = (0, 255, 255)  # Cyan for RL
        self.warning_color = (255, 100, 100)   # Red for danger
        
        logging.info("🤖 RL Enemy play mode initialized - prepare for intelligent opposition!")
    
    def update(self, current_time: int) -> None:
        """Update RL enemy play mode."""
        # Handle space bar for shooting
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not self.space_pressed_last_frame and self.game.player:
            self.game.player.shoot_missile(self.rl_enemy.pos)
            logging.debug("Player shot missile at RL enemy")
        self.space_pressed_last_frame = keys[pygame.K_SPACE]

        # Update player
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit.")
            self.game.running = False
            return

        # Update RL enemy with missile information
        if self.rl_enemy:
            missiles = self.game.player.missiles if self.game.player else []
            self.rl_enemy.update_movement(
                self.game.player.position["x"] if self.game.player else 0,
                self.game.player.position["y"] if self.game.player else 0,
                self.game.player.step if self.game.player else 0,
                current_time,
                missiles
            )

        # Check collisions
        self._check_collisions(current_time)
        
        # Update missiles
        if self.game.player:
            # Pass enemy position for smart missile tracking
            enemy_pos = self.rl_enemy.pos if self.rl_enemy and self.rl_enemy.visible else None
            self.game.player.update_missiles(enemy_pos)
            
        # Update smart missile AI
        if self.game.player and self.game.player.missiles and self.rl_enemy:
            for missile in self.game.player.missiles:
                if hasattr(missile, 'update_with_ai'):
                    missile.update_with_ai(
                        self.game.player.position,
                        self.rl_enemy.pos,
                        getattr(self.game, '_missile_input', None)
                    )

        # Handle respawning
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self._respawn_enemy()
        
        # Update fade-in animation
        if self.rl_enemy and hasattr(self.rl_enemy, 'fading_in') and self.rl_enemy.fading_in:
            self.rl_enemy.update_fade_in(current_time)
    
    def _check_collisions(self, current_time: int):
        """Check for collisions between player/missiles and RL enemy."""
        if not self.rl_enemy or not self.rl_enemy.visible:
            return
        
        # Player-enemy collision
        if self.game.player:
            player_rect = pygame.Rect(
                self.game.player.position["x"] - self.game.player.size // 2,
                self.game.player.position["y"] - self.game.player.size // 2,
                self.game.player.size,
                self.game.player.size
            )
            enemy_rect = self.rl_enemy.get_rect()
            
            if player_rect.colliderect(enemy_rect):
                logging.info("RL enemy caught the player!")
                self._handle_player_caught(current_time)
        
        # Missile-enemy collisions
        if self.game.player and self.game.player.missiles:
            missiles_to_remove = []
            
            for missile in self.game.player.missiles:
                missile_rect = missile.get_rect()
                enemy_rect = self.rl_enemy.get_rect()
                
                if missile_rect.colliderect(enemy_rect):
                    logging.info("Missile hit the RL enemy!")
                    missiles_to_remove.append(missile)
                    self._handle_enemy_hit(current_time)
                    break  # Only one missile can hit per frame
            
            # Remove hit missiles
            for missile in missiles_to_remove:
                self.game.player.missiles.remove(missile)
    
    def _handle_player_caught(self, current_time: int):
        """Handle when RL enemy catches the player."""
        # Player loses, enemy wins
        if self.game.player:
            # Reset player position
            self.game.player.position = {
                "x": self.game.screen_width // 4,
                "y": self.game.screen_height // 4
            }
            self.game.player.missiles.clear()
        
        # Enemy gets a brief victory moment
        logging.info("🤖 RL Enemy Victory! Advanced AI tactics succeeded.")
    
    def _handle_enemy_hit(self, current_time: int):
        """Handle when RL enemy gets hit by missile."""
        self.rl_enemy.hide()
        self.game.is_respawning = True
        self.game.respawn_timer = current_time + self.game.respawn_delay
        logging.info("RL enemy hit - will respawn with learned experience")
    
    def _respawn_enemy(self):
        """Respawn the RL enemy."""
        if not self.rl_enemy:
            return
        
        # Choose respawn position away from player
        import random
        if self.game.player:
            # Find position far from player
            attempts = 0
            while attempts < 10:
                x = random.randint(50, self.game.screen_width - 50)
                y = random.randint(50, self.game.screen_height - 50)
                
                dx = x - self.game.player.position["x"]
                dy = y - self.game.player.position["y"]
                distance = (dx*dx + dy*dy) ** 0.5
                
                if distance > 200:  # Far enough
                    break
                attempts += 1
        else:
            x = self.game.screen_width // 2
            y = self.game.screen_height // 2
        
        self.rl_enemy.set_position(x, y)
        self.rl_enemy.start_fade_in(pygame.time.get_ticks())
        self.game.is_respawning = False
        
        logging.info(f"RL enemy respawned at ({x}, {y}) - ready to apply learned strategies")
    
    def draw_ui(self, screen: pygame.Surface):
        """Draw RL-specific UI elements."""
        if not self.rl_enemy:
            return
        
        # Get performance stats
        stats = self.rl_enemy.get_performance_stats()
        
        # Create semi-transparent background for UI
        ui_width = 300
        ui_height = 150
        ui_x = screen.get_width() - ui_width - 10
        ui_y = 10
        
        ui_surface = pygame.Surface((ui_width, ui_height), pygame.SRCALPHA)
        ui_surface.fill(self.ui_bg_color)
        screen.blit(ui_surface, (ui_x, ui_y))
        
        # Draw RL enemy status
        y_offset = ui_y + 10
        
        # Title
        title_text = self.font_large.render("🤖 RL ENEMY", True, self.rl_indicator_color)
        screen.blit(title_text, (ui_x + 10, y_offset))
        y_offset += 35
        
        # AI status
        ai_info = self.rl_enemy.get_ai_info()
        ai_text = self.font_small.render(ai_info[:35], True, self.text_secondary)
        screen.blit(ai_text, (ui_x + 10, y_offset))
        y_offset += 20
        
        # Performance stats
        survival_text = self.font_small.render(f"Survival Time: {stats['survival_time']}", True, self.text_primary)
        screen.blit(survival_text, (ui_x + 10, y_offset))
        y_offset += 18
        
        dodged_text = self.font_small.render(f"Missiles Dodged: {stats['missiles_dodged']}", True, self.text_primary)
        screen.blit(dodged_text, (ui_x + 10, y_offset))
        y_offset += 18
        
        # Energy bar
        energy_text = self.font_small.render(f"Energy: {stats['energy']:.0f}%", True, self.text_primary)
        screen.blit(energy_text, (ui_x + 10, y_offset))
        
        # Energy bar visual
        bar_width = 100
        bar_height = 8
        bar_x = ui_x + 120
        bar_y = y_offset + 5
        
        # Background bar
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
        
        # Energy fill
        energy_fill_width = int((stats['energy'] / 100.0) * bar_width)
        energy_color = self.text_secondary if stats['energy'] > 30 else self.warning_color
        pygame.draw.rect(screen, energy_color, (bar_x, bar_y, energy_fill_width, bar_height))
        
        # Draw RL status indicator
        if stats['rl_enabled']:
            indicator_text = self.font_small.render("TRUE RL ACTIVE", True, self.rl_indicator_color)
            screen.blit(indicator_text, (ui_x + 10, ui_y + ui_height - 25))
        else:
            indicator_text = self.font_small.render("FALLBACK MODE", True, self.warning_color)
            screen.blit(indicator_text, (ui_x + 10, ui_y + ui_height - 25))
        
        # Draw velocity vector (for debugging/visualization)
        if abs(stats['velocity']['x']) > 0.1 or abs(stats['velocity']['y']) > 0.1:
            start_pos = (int(self.rl_enemy.pos['x']), int(self.rl_enemy.pos['y']))
            end_pos = (
                int(self.rl_enemy.pos['x'] + stats['velocity']['x'] * 10),
                int(self.rl_enemy.pos['y'] + stats['velocity']['y'] * 10)
            )
            pygame.draw.line(screen, self.rl_indicator_color, start_pos, end_pos, 2)
    
    def get_mode_info(self) -> str:
        """Get information about this game mode."""
        if self.rl_enemy:
            stats = self.rl_enemy.get_performance_stats()
            if stats['rl_enabled']:
                return f"RL Enemy Mode - True AI (Survived: {stats['survival_time']}, Dodged: {stats['missiles_dodged']})"
            else:
                return f"RL Enemy Mode - Fallback (Survived: {stats['survival_time']})"
        return "RL Enemy Mode - Initializing"
    
    def reset(self):
        """Reset the RL enemy mode."""
        if self.rl_enemy:
            # Reset enemy to starting position
            self.rl_enemy.set_position(
                self.game.screen_width // 2,
                self.game.screen_height // 2
            )
            self.rl_enemy.show()
            self.rl_enemy.survival_time = 0
            self.rl_enemy.missiles_dodged = 0
            self.rl_enemy.energy = 100.0
        
        logging.info("RL enemy mode reset - AI ready for new challenge")