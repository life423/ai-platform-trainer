"""
SimpleMissileModel: A neural network model for missile trajectory prediction.

This module defines a feedforward neural network used for predicting
missile trajectories based on game state inputs, including player position,
enemy position, missile position and angle, and collision information.
"""
import torch
from torch import nn
from torch import Tensor


class SimpleMissileModel(nn.Module):
    """
    Feedforward PyTorch model for missile trajectory predictions.

    This model consists of multiple fully connected layers with ReLU activations
    on the hidden layers, with optional dropout and batch normalization for
    improved training stability and generalization.
    """

    def __init__(
        self, 
        input_size: int = 9, 
        hidden_size: int = 64, 
        output_size: int = 1,
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ) -> None:
        """
        Initialize the model with configurable layer sizes and regularization.

        Args:
            input_size: Number of input features (default: 9)
                - player_x, player_y: Player position
                - enemy_x, enemy_y: Enemy position
                - missile_x, missile_y: Missile position
                - missile_angle: Current missile angle
                - distance: Distance between missile and enemy
                - collision: Whether the missile has collided with anything
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output values (usually 1 for steering angle)
            dropout_rate: Probability of dropping units during training
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.use_batch_norm = use_batch_norm
        
        # First hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity()
        
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size) if use_batch_norm else nn.Identity()
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor containing model features
                Shape: (batch_size, input_size)

        Returns:
            Tensor containing prediction outputs
                Shape: (batch_size, output_size)
        """
        # First hidden layer
        x = self.fc1(x)
        if self.use_batch_norm and x.shape[0] > 1:  # Check batch size for BatchNorm
            x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        if self.use_batch_norm and x.shape[0] > 1:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Output layer (no activation - raw logits)
        x = self.fc3(x)
        
        return x

    def predict(self, state: Tensor) -> float:
        """
        Make a prediction for a single game state.

        This is a convenience method for inference that handles
        the conversion to evaluation mode and detachment of gradients.

        Args:
            state: A single state tensor
                Shape: (input_size,) or (1, input_size)

        Returns:
            The predicted action value as a float
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Ensure proper batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # Make prediction
            prediction = self(state)
            
            # Extract the single value
            result = prediction.item()
        
        return result
