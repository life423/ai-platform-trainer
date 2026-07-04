"""
EnemyMovementModel: A neural network model for enemy movement prediction.

This module defines a neural network used for predicting enemy movement
based on game state inputs.
"""
import torch.nn as nn
from torch import Tensor


class EnemyMovementModel(nn.Module):
    """
    Neural network model for enemy movement prediction.

    This model consists of four fully connected layers with LeakyReLU activations,
    batch normalization, and dropout for regularization.
    """

    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 128,
        output_size: int = 2,
        dropout_prob: float = 0.3,
    ) -> None:
        """
        Initialize the model with configurable layer sizes.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output values (typically 2 for x,y movement)
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

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor containing model features

        Returns:
            Tensor containing prediction outputs (typically x,y movement)

        Raises:
            ValueError: If input tensor has incorrect shape
        """
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(
                f"Expected input to have shape (batch_size, {self.fc1.in_features})."
            )

        x = nn.functional.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
