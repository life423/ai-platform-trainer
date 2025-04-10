"""
Neural network model for enemy movement prediction.

This module defines a multi-layer perceptron used to predict enemy movement
based on the current game state, including player and enemy positions.
"""
import torch
import torch.nn as nn


class EnemyMovementModel(nn.Module):
    """
    A neural network model for predicting enemy movement.
    
    This model consists of multiple fully connected layers with 
    batch normalization, dropout for regularization, and leaky ReLU
    activations to handle enemy movement AI.
    """
    
    def __init__(
        self, 
        input_size: int = 5, 
        hidden_size: int = 128, 
        output_size: int = 2, 
        dropout_prob: float = 0.3
    ) -> None:
        """
        Initialize the enemy movement model.
        
        Args:
            input_size: Number of input features (player pos, enemy pos, etc.)
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output values (dx, dy movement)
            dropout_prob: Dropout probability for regularization
        """
        super(EnemyMovementModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor containing model features
               Expected shape: (batch_size, input_size)
        
        Returns:
            Tensor containing movement predictions (dx, dy)
        
        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(
                f"Expected input to have shape (batch_size, {self.fc1.in_features})."
            )

        # First hidden layer with batch norm and leaky ReLU
        x = nn.functional.leaky_relu(
            self.bn1(self.fc1(x)), negative_slope=0.01
        )
        x = self.dropout(x)
        
        # Second hidden layer
        x = nn.functional.leaky_relu(
            self.bn2(self.fc2(x)), negative_slope=0.01
        )
        x = self.dropout(x)
        
        # Third hidden layer
        x = nn.functional.leaky_relu(
            self.bn3(self.fc3(x)), negative_slope=0.01
        )
        x = self.dropout(x)
        
        # Output layer
        x = self.fc4(x)
        return x
