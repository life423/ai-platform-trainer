"""
Clean Missile AI Model

This module provides a simplified missile AI model.
"""
import torch
import torch.nn as nn
import os
import logging

from config.paths import paths


class MissileModel(nn.Module):
    """Neural network model for missile guidance."""
    
    def __init__(self, input_size: int = 9, hidden_size: int = 64, output_size: int = 2):
        super(MissileModel, self).__init__()
        
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


class MissileAI:
    """AI controller for missile guidance."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
    
    def _load_model(self):
        """Load the trained missile model."""
        # Try new path first, then legacy paths
        model_paths = [
            paths.MISSILE_MODEL,
            paths.LEGACY_MISSILE_MODEL
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.model = MissileModel()
                    self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model.eval()
                    logging.info(f"Loaded missile AI model from {model_path}")
                    return
                except Exception as e:
                    logging.warning(f"Failed to load model from {model_path}: {e}")
        
        logging.warning("No missile AI model found, using simple guidance")
    
    def predict_guidance(self, missile_x: float, missile_y: float, missile_vx: float, 
                        missile_vy: float, target_x: float, target_y: float,
                        player_x: float, player_y: float, step: int) -> tuple:
        """
        Predict missile guidance adjustment.
        
        Args:
            missile_x: Missile X position
            missile_y: Missile Y position
            missile_vx: Missile X velocity
            missile_vy: Missile Y velocity
            target_x: Target X position
            target_y: Target Y position
            player_x: Player X position
            player_y: Player Y position
            step: Current game step
            
        Returns:
            Tuple of (adjust_x, adjust_y) values
        """
        if self.model is None:
            # Simple guidance fallback
            dx = target_x - missile_x
            dy = target_y - missile_y
            distance = (dx*dx + dy*dy) ** 0.5
            
            if distance > 1:
                return (dx/distance * 0.1, dy/distance * 0.1)
            return (0.0, 0.0)
        
        try:
            # Prepare input tensor
            input_data = torch.tensor([
                missile_x / 1280.0,    # Normalize positions
                missile_y / 720.0,
                missile_vx / 10.0,     # Normalize velocities
                missile_vy / 10.0,
                target_x / 1280.0,
                target_y / 720.0,
                player_x / 1280.0,
                player_y / 720.0,
                step / 1000.0          # Normalize step
            ], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_data)
                adjust_x, adjust_y = output[0].cpu().numpy()
            
            return float(adjust_x), float(adjust_y)
            
        except Exception as e:
            logging.error(f"Error in missile AI prediction: {e}")
            return (0.0, 0.0)