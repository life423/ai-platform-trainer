"""
Clean Enemy AI Model

This module provides a simplified enemy AI model.
"""
import torch
import torch.nn as nn
import os
import logging

from config.paths import paths


class EnemyModel(nn.Module):
    """Neural network model for enemy movement."""
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 2):
        super(EnemyModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))
        return x


class EnemyAI:
    """AI controller for enemy behavior."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the trained enemy model."""
        # Try new path first, then legacy paths
        model_paths = [
            paths.ENEMY_AI_MODEL,
            paths.LEGACY_ENEMY_MODEL
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = EnemyModel()
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    logging.info(f"Loaded enemy AI model from {model_path}")
                    return
                except Exception as e:
                    logging.warning(f"Failed to load model from {model_path}: {e}")
        
        logging.warning("No enemy AI model found, using random behavior")
    
    def predict_movement(self, player_x: float, player_y: float, enemy_x: float, 
                        enemy_y: float, distance: float) -> tuple:
        """
        Predict enemy movement based on game state.
        
        Args:
            player_x: Player X position
            player_y: Player Y position
            enemy_x: Enemy X position
            enemy_y: Enemy Y position
            distance: Distance between player and enemy
            
        Returns:
            Tuple of (move_x, move_y) values
        """
        if self.model is None:
            # Random movement fallback
            import random
            return (random.uniform(-1, 1), random.uniform(-1, 1))
        
        try:
            # Prepare input tensor
            input_data = torch.tensor([
                player_x / 1280.0,  # Normalize to screen width
                player_y / 720.0,   # Normalize to screen height
                enemy_x / 1280.0,
                enemy_y / 720.0,
                distance / 1000.0   # Normalize distance
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_data)
                move_x, move_y = output[0].cpu().numpy()
            
            return float(move_x), float(move_y)
            
        except Exception as e:
            logging.error(f"Error in AI prediction: {e}")
            return (0.0, 0.0)