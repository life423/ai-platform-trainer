"""
Clean Supervised Training

This module provides simplified supervised learning training.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import List, Dict, Any
import os

from config.paths import paths
from ..models.enemy_model import EnemyModel
from ..models.missile_model import MissileModel


class SupervisedTrainer:
    """Handles supervised learning training for AI models."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        """Load training data from the JSON file."""
        if not os.path.exists(paths.TRAINING_DATA_FILE):
            logging.error(f"Training data file not found: {paths.TRAINING_DATA_FILE}")
            return []
        
        try:
            with open(paths.TRAINING_DATA_FILE, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logging.error("Training data is not in the expected list format")
                return []
            
            logging.info(f"Loaded {len(data)} training samples")
            return data
            
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return []
    
    def prepare_enemy_data(self, data: List[Dict[str, Any]]) -> tuple:
        """
        Prepare data for enemy model training.
        
        Returns:
            Tuple of (inputs, targets) as tensors
        """
        inputs = []
        targets = []
        
        for i, sample in enumerate(data):
            if i == 0:
                continue  # Skip first sample as we need previous position
            
            prev_sample = data[i-1]
            
            # Input features: normalized positions and distance
            player_x = sample.get('player_x', 0) / 1280.0
            player_y = sample.get('player_y', 0) / 720.0
            enemy_x = sample.get('enemy_x', 0) / 1280.0
            enemy_y = sample.get('enemy_y', 0) / 720.0
            distance = sample.get('distance_to_enemy', 0) / 1000.0
            
            inputs.append([player_x, player_y, enemy_x, enemy_y, distance])
            
            # Target: movement direction (normalized)
            prev_enemy_x = prev_sample.get('enemy_x', enemy_x * 1280.0)
            prev_enemy_y = prev_sample.get('enemy_y', enemy_y * 720.0)
            
            move_x = (sample.get('enemy_x', 0) - prev_enemy_x) / 10.0  # Normalize movement
            move_y = (sample.get('enemy_y', 0) - prev_enemy_y) / 10.0
            
            # Clamp to [-1, 1]
            move_x = max(-1, min(1, move_x))
            move_y = max(-1, min(1, move_y))
            
            targets.append([move_x, move_y])
        
        if not inputs:
            logging.warning("No valid enemy training data prepared")
            return torch.tensor([]), torch.tensor([])
        
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def train_enemy_model(self, data: List[Dict[str, Any]], epochs: int = 100, lr: float = 0.001):
        """Train the enemy AI model."""
        logging.info("Starting enemy model training")
        
        # Prepare data
        inputs, targets = self.prepare_enemy_data(data)
        
        if len(inputs) == 0:
            logging.error("No training data available for enemy model")
            return False
        
        # Create model
        model = EnemyModel().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(paths.ENEMY_AI_MODEL), exist_ok=True)
        torch.save(model.state_dict(), paths.ENEMY_AI_MODEL)
        logging.info(f"Enemy model saved to {paths.ENEMY_AI_MODEL}")
        
        return True
    
    def prepare_missile_data(self, data: List[Dict[str, Any]]) -> tuple:
        """
        Prepare data for missile model training.
        
        Returns:
            Tuple of (inputs, targets) as tensors
        """
        inputs = []
        targets = []
        
        for i, sample in enumerate(data):
            if not sample.get('has_missile', False) or i == 0:
                continue
            
            prev_sample = data[i-1]
            if not prev_sample.get('has_missile', False):
                continue
            
            # Input features
            missile_x = sample.get('missile_x', 0) / 1280.0
            missile_y = sample.get('missile_y', 0) / 720.0
            missile_vx = sample.get('missile_vx', 0) / 10.0
            missile_vy = sample.get('missile_vy', 0) / 10.0
            enemy_x = sample.get('enemy_x', 0) / 1280.0
            enemy_y = sample.get('enemy_y', 0) / 720.0
            player_x = sample.get('player_x', 0) / 1280.0
            player_y = sample.get('player_y', 0) / 720.0
            step = sample.get('player_step', 0) / 1000.0
            
            inputs.append([missile_x, missile_y, missile_vx, missile_vy, 
                          enemy_x, enemy_y, player_x, player_y, step])
            
            # Target: velocity change
            prev_vx = prev_sample.get('missile_vx', missile_vx * 10.0)
            prev_vy = prev_sample.get('missile_vy', missile_vy * 10.0)
            
            delta_vx = (sample.get('missile_vx', 0) - prev_vx) / 10.0
            delta_vy = (sample.get('missile_vy', 0) - prev_vy) / 10.0
            
            # Clamp to [-1, 1]
            delta_vx = max(-1, min(1, delta_vx))
            delta_vy = max(-1, min(1, delta_vy))
            
            targets.append([delta_vx, delta_vy])
        
        if not inputs:
            logging.warning("No valid missile training data prepared")
            return torch.tensor([]), torch.tensor([])
        
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def train_missile_model(self, data: List[Dict[str, Any]], epochs: int = 100, lr: float = 0.001):
        """Train the missile AI model."""
        logging.info("Starting missile model training")
        
        # Prepare data
        inputs, targets = self.prepare_missile_data(data)
        
        if len(inputs) == 0:
            logging.error("No training data available for missile model")
            return False
        
        # Create model
        model = MissileModel().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Move data to device
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(paths.MISSILE_MODEL), exist_ok=True)
        torch.save(model.state_dict(), paths.MISSILE_MODEL)
        logging.info(f"Missile model saved to {paths.MISSILE_MODEL}")
        
        return True
    
    def train_all_models(self):
        """Train all AI models from collected data."""
        logging.info("Starting training of all AI models")
        
        # Load training data
        data = self.load_training_data()
        
        if not data:
            logging.error("No training data available")
            return False
        
        # Train enemy model
        enemy_success = self.train_enemy_model(data)
        
        # Train missile model
        missile_success = self.train_missile_model(data)
        
        if enemy_success and missile_success:
            logging.info("All models trained successfully")
            return True
        else:
            logging.error("Some models failed to train")
            return False