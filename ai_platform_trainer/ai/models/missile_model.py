"""
MissileModel: A neural network model for missile trajectory prediction.

This module defines a simple feedforward neural network used for predicting
missile trajectories based on game state inputs.
"""
from typing import Any  # noqa: F401

import torch
from torch import Tensor, nn


class MissileModel(nn.Module):
    """
    Simple feedforward PyTorch model for missile trajectory predictions.

    This model consists of three fully connected layers with ReLU activations
    on the hidden layers.
    """

    def __init__(
        self, input_size: int = 9, hidden_size: int = 64, output_size: int = 1
    ) -> None:
        """
        Initialize the model with configurable layer sizes.

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layers
            output_size: Number of output values
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor containing model features

        Returns:
            Tensor containing prediction outputs
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# For backward compatibility
SimpleMissileModel = MissileModel
