"""
Neural network policy for reinforcement learning.

This module defines the neural network architecture used for the enemy AI policy
in reinforcement learning training.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """Neural network for the enemy AI policy."""

    def __init__(self, input_size=7, hidden_size=64, output_size=2):
        """
        Initialize the policy network.

        Args:
            input_size: Size of observation vector
            hidden_size: Size of hidden layers
            output_size: Size of action vector
        """
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor with shape [batch_size, input_size]

        Returns:
            Output tensor with shape [batch_size, output_size]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output in range [-1, 1]
        return x

    def save(self, path):
        """
        Save the model weights.

        Args:
            path: Path to save the model weights
        """
        try:
            torch.save(self.state_dict(), path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    def load(self, path):
        """
        Load the model weights.

        Args:
            path: Path to load the model weights from

        Returns:
            True if successful, False otherwise
        """
        try:
            self.load_state_dict(torch.load(path))
            self.eval()
            logging.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
