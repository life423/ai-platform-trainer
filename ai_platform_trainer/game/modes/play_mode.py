"""
Play mode for AI Platform Trainer.

Human player vs trained AI enemy.
"""
import pygame
import logging
import os
from ai_platform_trainer.game.entities.player import Player
from ai_platform_trainer.game.entities.enemy import Enemy
from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel


class PlayMode:
    """Play mode - human vs trained AI."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create entities
        self.player = Player(screen_width, screen_height, mode="play")
        self.enemy = Enemy(screen_width, screen_height, mode="play")
        
        # Load AI model
        self._load_enemy_ai()
        
        # Game state
        self.space_pressed_last_frame = False
        self.respawn_timer = 0
        self.respawn_delay = 1000
        self.is_respawning = False
    
    def _load_enemy_ai(self):
        """Load the best available AI model for the enemy."""
        # Try RL model first
        rl_model_path = "models/enemy_rl/final_model.zip"
        if os.path.exists(rl_model_path):
            try:
                from stable_baselines3 import PPO
                model = PPO.load(rl_model_path)
                self.enemy.set_rl_model(model)
                logging.info("Loaded RL model for enemy")
                return
            except Exception as e:
                logging.warning(f"Failed to load RL model: {e}")
        
        # Fallback to neural network
        nn_model_path = "models/enemy_ai_model.pth"
        if os.path.exists(nn_model_path):
            try:
                import torch
                model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
                model.load_state_dict(torch.load(nn_model_path, map_location="cpu"))
                model.eval()
                self.enemy.set_neural_network_model(model)
                logging.info("Loaded neural network model for enemy")
                return
            except Exception as e:
                logging.warning(f"Failed to load neural network model: {e}")
        
        # Use random AI as fallback
        logging.info("Using random AI for enemy")
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.shoot_missile(self.enemy.pos)
    
    def update(self):
        """Update game state."""
        current_time = pygame.time.get_ticks()
        
        # Update player
        self.player.handle_input()
        self.player.update_missiles()
        
        # Update enemy
        self.enemy.update_movement(
            self.player.position["x"],
            self.player.position["y"],
            self.player.step,
            current_time
        )
        
        # Check collisions
        self._check_collisions()
        
        # Handle respawning
        if self.is_respawning and current_time >= self.respawn_timer:
            self._respawn_enemy()
    
    def _check_collisions(self):
        """Check for collisions between entities."""
        # Player-Enemy collision
        if self._entities_collide(self.player, self.enemy):
            logging.info("Player-Enemy collision!")
            self.enemy.hide()
            self.is_respawning = True
            self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
        
        # Missile-Enemy collisions
        for missile in self.player.missiles[:]:
            if self._missile_enemy_collide(missile, self.enemy):
                logging.info("Missile hit enemy!")
                self.player.missiles.remove(missile)
                self.enemy.hide()
                self.is_respawning = True
                self.respawn_timer = pygame.time.get_ticks() + self.respawn_delay
    
    def _entities_collide(self, entity1, entity2) -> bool:
        """Check collision between two entities."""
        if not entity2.visible:
            return False
        
        rect1 = pygame.Rect(entity1.position["x"], entity1.position["y"], entity1.size, entity1.size)
        rect2 = pygame.Rect(entity2.pos["x"], entity2.pos["y"], entity2.size, entity2.size)
        return rect1.colliderect(rect2)
    
    def _missile_enemy_collide(self, missile, enemy) -> bool:
        """Check collision between missile and enemy."""
        if not enemy.visible:
            return False
        
        missile_rect = pygame.Rect(missile.pos["x"], missile.pos["y"], missile.size, missile.size)
        enemy_rect = pygame.Rect(enemy.pos["x"], enemy.pos["y"], enemy.size, enemy.size)
        return missile_rect.colliderect(enemy_rect)
    
    def _respawn_enemy(self):
        """Respawn the enemy at a random location."""
        import random
        
        # Find a position away from player
        while True:
            x = random.randint(0, self.screen_width - self.enemy.size)
            y = random.randint(0, self.screen_height - self.enemy.size)
            
            # Check distance from player
            dx = x - self.player.position["x"]
            dy = y - self.player.position["y"]
            distance = (dx*dx + dy*dy) ** 0.5
            
            if distance > 200:  # Minimum spawn distance
                break
        
        self.enemy.set_position(x, y)
        self.enemy.show(pygame.time.get_ticks())
        self.is_respawning = False
    
    def render(self, screen: pygame.Surface):
        """Render the play mode."""
        self.player.draw(screen)
        self.enemy.draw(screen)
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        text = font.render("PLAY MODE - ESC to menu", True, (255, 255, 255))
        screen.blit(text, (10, 10))