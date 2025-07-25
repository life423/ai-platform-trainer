"""
Learning AI Play Mode

This module provides a play mode where the player fights against a real-time learning AI.
The AI starts weak and evolves during gameplay with visual feedback showing its progress.
"""
import pygame
import random
import logging
import math

from ai_platform_trainer.entities.enemy_learning import LearningEnemyAI
# Color constants
COLOR_TEXT_PRIMARY = (255, 255, 255)  # White  
COLOR_TEXT_SECONDARY = (240, 248, 255)  # Alice blue
COLOR_SELECTED = (255, 215, 0)  # Gold


class PlayLearningMode:
    """Play mode against real-time learning AI with visual feedback."""
    
    def __init__(self, game):
        """Initialize learning play mode."""
        self.game = game
        self.space_pressed_last_frame = False
        
        # Create learning AI enemy
        self.learning_enemy = LearningEnemyAI(
            self.game.screen_width, 
            self.game.screen_height
        )
        
        # Replace the regular enemy with our learning AI
        self.game.enemy = self.learning_enemy
        
        # UI elements for learning feedback
        self.font_large = pygame.font.Font(None, 42)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 22)
        
        # Colors for UI - using accessible colors from settings
        self.ui_bg_color = (25, 25, 112, 180)  # Semi-transparent midnight blue for contrast
        self.progress_bar_bg = (60, 60, 60)    # Dark gray
        self.progress_bar_fill = (255, 140, 0)  # Dark orange (good contrast)
        self.text_primary = COLOR_TEXT_PRIMARY      # White
        self.text_secondary = COLOR_TEXT_SECONDARY  # Alice blue
        self.highlight_color = COLOR_SELECTED       # Gold
        self.warning_color = (255, 69, 0)  # Red-orange for danger
        
        logging.info("Learning AI play mode initialized")
    
    def update(self, current_time: int) -> None:
        """Update learning play mode."""
        # Handle space bar for shooting
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and not self.space_pressed_last_frame and self.game.player:
            self.game.player.shoot_missile(self.learning_enemy.pos)
            logging.debug("Player shot missile at learning AI")
        self.space_pressed_last_frame = keys[pygame.K_SPACE]

        # Update player
        if self.game.player and not self.game.player.handle_input():
            logging.info("Player requested to quit.")
            self.game.running = False
            return

        # Update learning enemy with missile information
        if self.learning_enemy:
            missiles = self.game.player.missiles if self.game.player else []
            self.learning_enemy.update_movement(
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
            self.game.player.update_missiles()
            
        # Update smart missile AI - missiles will automatically home in on the learning enemy
        if self.game.player and self.game.player.missiles and self.learning_enemy:
            for missile in self.game.player.missiles:
                # Update smart missiles with AI guidance
                if hasattr(missile, 'update_with_ai'):
                    missile.update_with_ai(
                        self.game.player.position,
                        self.learning_enemy.pos,
                        getattr(self.game, '_missile_input', None)
                    )

        # Handle respawning
        if self.game.is_respawning and current_time >= self.game.respawn_timer:
            self._respawn_enemy()

        # Handle enemy fade-in
        if self.learning_enemy and hasattr(self.learning_enemy, 'fading_in') and self.learning_enemy.fading_in:
            self.learning_enemy.update_fade_in(current_time)
    
    def _check_collisions(self, current_time: int):
        """Check for collisions and update learning AI."""
        if not self.game.player or not self.learning_enemy:
            return
            
        # Player-Enemy collision
        if self._entities_collide(self.game.player, self.learning_enemy):
            logging.info("Learning AI caught the player!")
            self.learning_enemy.on_hit_player()  # Notify AI of success
            self.learning_enemy.hide()
            self.game.is_respawning = True
            self.game.respawn_timer = current_time + self.game.respawn_delay

        # Missile-Enemy collisions
        for missile in self.game.player.missiles[:]:
            if self._missile_enemy_collide(missile, self.learning_enemy):
                logging.info("Player missile hit learning AI!")
                self.learning_enemy.on_hit_by_missile()  # Notify AI of failure
                self.game.player.missiles.remove(missile)
                self.learning_enemy.hide()
                self.game.is_respawning = True
                self.game.respawn_timer = current_time + self.game.respawn_delay
                break
    
    def _entities_collide(self, entity1, entity2) -> bool:
        """Check collision between two entities."""
        if not entity2.visible:
            return False
        
        rect1 = pygame.Rect(entity1.position["x"], entity1.position["y"], 
                           entity1.size, entity1.size)
        rect2 = pygame.Rect(entity2.pos["x"], entity2.pos["y"], 
                           entity2.size, entity2.size)
        return rect1.colliderect(rect2)
    
    def _missile_enemy_collide(self, missile, enemy) -> bool:
        """Check collision between missile and enemy."""
        if not enemy.visible:
            return False
        
        missile_rect = pygame.Rect(missile.pos["x"], missile.pos["y"], 
                                  missile.size, missile.size)
        enemy_rect = pygame.Rect(enemy.pos["x"], enemy.pos["y"], 
                                enemy.size, enemy.size)
        return missile_rect.colliderect(enemy_rect)
    
    def _respawn_enemy(self):
        """Respawn the learning enemy at a new location."""
        if not self.game.player:
            return
            
        # Find spawn position away from player
        attempts = 0
        while attempts < 10:
            x = random.randint(0, self.game.screen_width - self.learning_enemy.size)
            y = random.randint(0, self.game.screen_height - self.learning_enemy.size)
            
            # Check distance from player
            dx = x - self.game.player.position["x"]
            dy = y - self.game.player.position["y"]
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance > 200:  # Minimum spawn distance
                break
            attempts += 1
        
        self.learning_enemy.set_position(x, y)
        self.learning_enemy.show(pygame.time.get_ticks())
        self.game.is_respawning = False
        
        logging.debug(f"Learning AI respawned at ({x}, {y})")
    
    def render_learning_ui(self, screen: pygame.Surface):
        """Render the learning AI feedback UI with proper spacing and contrast."""
        if not self.learning_enemy:
            return
            
        stats = self.learning_enemy.get_learning_stats()
        difficulty = self.learning_enemy.get_difficulty_level()
        
        # UI panel dimensions - improved layout
        panel_width = 320
        panel_height = 180
        panel_x = screen.get_width() - panel_width - 15
        panel_y = 15
        
        # Draw semi-transparent background panel with border
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill(self.ui_bg_color)
        screen.blit(panel_surface, (panel_x, panel_y))
        
        # Draw border for better definition
        pygame.draw.rect(screen, self.text_secondary, (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Title with better spacing
        title_text = self.font_medium.render("AI Learning Progress", True, self.highlight_color)
        screen.blit(title_text, (panel_x + 15, panel_y + 12))
        
        # AI Stage with proper spacing
        stage_text = self.font_small.render(f"Stage: {stats['stage']}", True, self.text_primary)
        screen.blit(stage_text, (panel_x + 15, panel_y + 45))
        
        # Difficulty bar with improved design
        bar_x = panel_x + 15
        bar_y = panel_y + 68
        bar_width = panel_width - 30
        bar_height = 16
        
        # Background bar with border
        pygame.draw.rect(screen, self.progress_bar_bg, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(screen, self.text_secondary, (bar_x, bar_y, bar_width, bar_height), 1)
        
        # Progress fill with smooth gradient effect
        fill_width = int(bar_width * difficulty)
        if fill_width > 0:
            # Choose color based on difficulty level
            if difficulty < 0.3:
                fill_color = (76, 175, 80)   # Green - easy
            elif difficulty < 0.6:
                fill_color = (255, 193, 7)   # Yellow - medium
            elif difficulty < 0.8:
                fill_color = (255, 152, 0)   # Orange - hard
            else:
                fill_color = (244, 67, 54)   # Red - expert
                
            pygame.draw.rect(screen, fill_color, (bar_x + 1, bar_y + 1, fill_width - 2, bar_height - 2))
        
        # Difficulty percentage with better positioning
        difficulty_text = self.font_small.render(f"Difficulty: {difficulty:.1%}", True, self.text_primary)
        screen.blit(difficulty_text, (panel_x + 15, bar_y + 22))
        
        # Stats with better spacing and organization
        stats_y_start = panel_y + 105
        line_height = 16
        
        # Split stats into two columns for better layout
        left_stats = [
            f"Frames: {stats['frames']:,}",
            f"Speed: {stats['speed']:.1f}"
        ]
        right_stats = [
            f"Hits: {stats['hits']}",
            f"Deaths: {stats['deaths']}"
        ]
        
        # Left column
        for i, line in enumerate(left_stats):
            stats_text = self.font_small.render(line, True, self.text_secondary)
            screen.blit(stats_text, (panel_x + 15, stats_y_start + i * line_height))
        
        # Right column
        for i, line in enumerate(right_stats):
            stats_text = self.font_small.render(line, True, self.text_secondary)
            screen.blit(stats_text, (panel_x + 180, stats_y_start + i * line_height))
        
        # Learning status indicator at bottom of screen with better visibility
        if stats['frames'] < 60:
            status = "🤖 AI is learning basic behavior..."
            status_color = self.text_primary
        elif stats['frames'] < 180:
            status = "🎯 AI is actively hunting you!"
            status_color = (255, 193, 7)  # Yellow
        elif stats['frames'] < 300:
            status = "🧠 AI has become a smart predator!"
            status_color = (255, 152, 0)  # Orange
        else:
            status = "💀 NIGHTMARE MODE - AI is extremely dangerous!"
            status_color = self.warning_color
        
        # Background for status text
        status_text = self.font_small.render(status, True, status_color)
        status_rect = status_text.get_rect()
        status_bg = pygame.Surface((status_rect.width + 20, status_rect.height + 8), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 160))
        
        status_x = 15
        status_y = screen.get_height() - 40
        screen.blit(status_bg, (status_x - 10, status_y - 4))
        screen.blit(status_text, (status_x, status_y))
    
    def draw_mode_info(self, screen: pygame.Surface):
        """Draw mode information with improved styling."""
        # Mode title with background for better visibility
        title_text = self.font_large.render("LEARNING AI MODE", True, self.highlight_color)
        title_rect = title_text.get_rect()
        title_bg = pygame.Surface((title_rect.width + 20, title_rect.height + 8), pygame.SRCALPHA)
        title_bg.fill((0, 0, 0, 140))
        
        screen.blit(title_bg, (10, 10))
        screen.blit(title_text, (20, 14))
        
        # Instructions with better spacing and colors
        instructions = [
            "🎮 WASD/Arrows: Move  |  Space: Shoot  |  ESC: Menu",
            "🤖 Watch the AI evolve from basic to expert!",
            "📊 Progress shown in top-right panel"
        ]
        
        instruction_y_start = title_rect.height + 25
        for i, instruction in enumerate(instructions):
            instruction_text = self.font_small.render(instruction, True, self.text_primary)
            instruction_rect = instruction_text.get_rect()
            
            # Background for each instruction
            inst_bg = pygame.Surface((instruction_rect.width + 16, instruction_rect.height + 4), pygame.SRCALPHA)
            inst_bg.fill((0, 0, 0, 120))
            
            y_pos = instruction_y_start + i * 22
            screen.blit(inst_bg, (10, y_pos))
            screen.blit(instruction_text, (18, y_pos + 2))
        
        # Render learning progress UI
        self.render_learning_ui(screen)