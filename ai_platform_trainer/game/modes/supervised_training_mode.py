"""
Supervised Training Mode.

Collect gameplay data and train models from human demonstrations.
"""
import pygame
import logging
import json
import os
from typing import List, Dict, Any

from ai_platform_trainer.game.entities.player import Player
from ai_platform_trainer.game.entities.enemy import Enemy


class SupervisedTrainingMode:
    """Supervised learning training mode - collect data and train models."""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Create entities
        self.player = Player(screen_width, screen_height, mode="training")
        self.enemy = Enemy(screen_width, screen_height, mode="supervised_training")
        
        # Data collection
        self.training_data: List[Dict[str, Any]] = []
        self.data_file_path = "data/raw/training_data.json"
        self.collecting_data = True
        self.auto_save_interval = 30000  # 30 seconds
        self.last_save_time = pygame.time.get_ticks()
        
        # Training metrics
        self.data_points_collected = 0
        self.sessions_completed = 0
        self.current_session_data = 0
        
        # Load existing data if available
        self._load_existing_data()
        
        logging.info("Supervised training mode initialized")
    
    def _load_existing_data(self):
        """Load existing training data if available."""
        if os.path.exists(self.data_file_path):
            try:
                with open(self.data_file_path, 'r') as f:
                    existing_data = json.load(f)
                    if isinstance(existing_data, list):
                        self.training_data = existing_data
                        self.data_points_collected = len(self.training_data)
                        logging.info(f"Loaded {self.data_points_collected} existing data points")
            except Exception as e:
                logging.error(f"Failed to load existing data: {e}")
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.player.shoot_missile(self.enemy.pos)
            elif event.key == pygame.K_s:
                self._save_data()
            elif event.key == pygame.K_c:
                self.collecting_data = not self.collecting_data
                status = "enabled" if self.collecting_data else "disabled"
                logging.info(f"Data collection {status}")
            elif event.key == pygame.K_t:
                self._train_models()
            elif event.key == pygame.K_r:
                self._reset_session()
    
    def update(self):
        """Update supervised training mode."""
        current_time = pygame.time.get_ticks()
        
        # Update player
        self.player.handle_input()
        self.player.update_missiles()
        
        # Update enemy (simple AI for demonstration)
        self.enemy.update_movement(
            self.player.position["x"],
            self.player.position["y"],
            self.player.step,
            current_time
        )
        
        # Collect training data
        if self.collecting_data:
            self._collect_data_point(current_time)
        
        # Auto-save periodically
        if current_time - self.last_save_time > self.auto_save_interval:
            self._save_data()
            self.last_save_time = current_time
    
    def _collect_data_point(self, current_time: int):
        """Collect a training data point."""
        # Collect comprehensive game state
        data_point = {
            "timestamp": current_time,
            "player": {
                "position": self.player.position.copy(),
                "velocity": self.player.step,
                "missiles": [
                    {
                        "position": missile.pos.copy(),
                        "velocity": missile.velocity.copy(),
                        "age": current_time - missile.birth_time
                    }
                    for missile in self.player.missiles
                ]
            },
            "enemy": {
                "position": self.enemy.pos.copy(),
                "visible": self.enemy.visible,
                "ai_mode": self.enemy.ai_mode
            },
            "game_state": {
                "screen_width": self.screen_width,
                "screen_height": self.screen_height,
                "session_id": self.sessions_completed
            }
        }
        
        self.training_data.append(data_point)
        self.data_points_collected += 1
        self.current_session_data += 1
    
    def _save_data(self):
        """Save collected training data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.data_file_path), exist_ok=True)
            
            # Save data
            with open(self.data_file_path, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            
            logging.info(f"Saved {len(self.training_data)} data points to {self.data_file_path}")
            
        except Exception as e:
            logging.error(f"Failed to save training data: {e}")
    
    def _train_models(self):
        """Train AI models from collected data."""
        if len(self.training_data) < 100:
            logging.warning("Not enough data points for training (need at least 100)")
            return
        
        logging.info("Starting model training from collected data...")
        
        try:
            # Train enemy movement model
            self._train_enemy_model()
            
            # Train missile guidance model
            self._train_missile_model()
            
            logging.info("Model training completed!")
            
        except Exception as e:
            logging.error(f"Model training failed: {e}")
    
    def _train_enemy_model(self):
        """Train enemy movement model from collected data."""
        import torch
        import torch.nn as nn
        from ai_platform_trainer.ai.models.enemy_movement_model import EnemyMovementModel
        
        # Prepare training data
        inputs = []
        targets = []
        
        for i in range(len(self.training_data) - 1):
            current = self.training_data[i]
            next_frame = self.training_data[i + 1]
            
            # Input: normalized positions and distance
            player_pos = current["player"]["position"]
            enemy_pos = current["enemy"]["position"]
            
            dx = player_pos["x"] - enemy_pos["x"]
            dy = player_pos["y"] - enemy_pos["y"]
            distance = (dx*dx + dy*dy) ** 0.5
            
            input_vec = [
                player_pos["x"] / self.screen_width,
                player_pos["y"] / self.screen_height,
                enemy_pos["x"] / self.screen_width,
                enemy_pos["y"] / self.screen_height,
                distance / max(self.screen_width, self.screen_height)
            ]
            
            # Target: movement direction
            next_enemy_pos = next_frame["enemy"]["position"]
            move_x = next_enemy_pos["x"] - enemy_pos["x"]
            move_y = next_enemy_pos["y"] - enemy_pos["y"]
            
            # Normalize movement
            move_length = (move_x*move_x + move_y*move_y) ** 0.5
            if move_length > 0:
                move_x /= move_length
                move_y /= move_length
            
            inputs.append(input_vec)
            targets.append([move_x, move_y])
        
        # Convert to tensors
        X = torch.tensor(inputs, dtype=torch.float32)
        y = torch.tensor(targets, dtype=torch.float32)
        
        # Create and train model
        model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logging.info(f"Enemy model training epoch {epoch}, loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/enemy_ai_model.pth")
        logging.info("Enemy model saved to models/enemy_ai_model.pth")
    
    def _train_missile_model(self):
        """Train missile guidance model from collected data."""
        # Similar to enemy model but for missile trajectories
        logging.info("Missile model training not implemented yet")
    
    def _reset_session(self):
        """Reset current training session."""
        self.sessions_completed += 1
        self.current_session_data = 0
        self.player.reset()
        
        # Reset enemy to random position
        import random
        x = random.randint(0, self.screen_width - self.enemy.size)
        y = random.randint(0, self.screen_height - self.enemy.size)
        self.enemy.set_position(x, y)
        
        logging.info(f"Started new training session #{self.sessions_completed}")
    
    def render(self, screen: pygame.Surface):
        """Render the supervised training mode."""
        # Draw entities
        self.player.draw(screen)
        self.enemy.draw(screen)
        
        # Draw training UI
        font = pygame.font.Font(None, 24)
        y_offset = 10
        
        # Title
        title = pygame.font.Font(None, 36).render("SUPERVISED TRAINING", True, (255, 255, 255))
        screen.blit(title, (10, y_offset))
        y_offset += 40
        
        # Data collection status
        status_color = (0, 255, 0) if self.collecting_data else (255, 0, 0)
        status_text = "COLLECTING" if self.collecting_data else "PAUSED"
        status = font.render(f"Data Collection: {status_text}", True, status_color)
        screen.blit(status, (10, y_offset))
        y_offset += 25
        
        # Statistics
        stats_text = [
            f"Total Data Points: {self.data_points_collected}",
            f"Current Session: {self.current_session_data}",
            f"Sessions Completed: {self.sessions_completed}",
            f"Data File: {self.data_file_path}"
        ]
        
        for text in stats_text:
            rendered = font.render(text, True, (255, 255, 255))
            screen.blit(rendered, (10, y_offset))
            y_offset += 20
        
        # Controls
        y_offset += 10
        controls = [
            "SPACE - Shoot missile",
            "S - Save data now",
            "C - Toggle data collection",
            "T - Train models from data",
            "R - Reset session",
            "ESC - Return to menu"
        ]
        
        for control in controls:
            rendered = font.render(control, True, (200, 200, 200))
            screen.blit(rendered, (10, y_offset))
            y_offset += 18