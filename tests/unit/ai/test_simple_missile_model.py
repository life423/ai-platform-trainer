"""
Unit tests for SimpleMissileModel.

Tests the initialization, structure, and forward pass functionality
of the SimpleMissileModel neural network.
"""
import pytest  # noqa: F401 - Used for the fixture decorator
import torch

from ai_platform_trainer.ai.models.simple_missile_model import SimpleMissileModel


class TestSimpleMissileModel:
    """Test suite for SimpleMissileModel class."""

    def test_model_initialization(self):
        """Test that model initializes with expected structure."""
        # Create a model with default parameters
        model = SimpleMissileModel()

        # Test model structure
        assert isinstance(model.fc1, torch.nn.Linear)
        assert isinstance(model.fc2, torch.nn.Linear)
        assert isinstance(model.fc3, torch.nn.Linear)

        # Test model layer dimensions
        assert model.fc1.in_features == 9
        assert model.fc1.out_features == 64
        assert model.fc2.in_features == 64
        assert model.fc2.out_features == 64
        assert model.fc3.in_features == 64
        assert model.fc3.out_features == 1

    def test_model_initialization_custom_sizes(self):
        """Test that model initializes correctly with custom parameters."""
        # Create a model with custom parameters
        model = SimpleMissileModel(input_size=12, hidden_size=32, output_size=2)

        # Test model layer dimensions match custom parameters
        assert model.fc1.in_features == 12
        assert model.fc1.out_features == 32
        assert model.fc2.in_features == 32
        assert model.fc2.out_features == 32
        assert model.fc3.in_features == 32
        assert model.fc3.out_features == 2

    def test_model_forward_pass(self):
        """Test model forward pass produces expected output shape."""
        # Use fixture for deterministic model
        model = SimpleMissileModel()

        # Create batch of input data (2 samples, 9 features)
        x = torch.rand(2, 9)

        # Perform forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (2, 1)

        # Check output is of expected type
        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32

    def test_model_forward_pass_single_sample(self):
        """Test model forward pass with a single sample."""
        # Use fixture for deterministic model
        model = SimpleMissileModel()

        # Create single input sample
        x = torch.rand(1, 9)

        # Perform forward pass
        output = model(x)

        # Check output shape
        assert output.shape == (1, 1)

        # Ensure output is a valid floating point number (not NaN or Inf)
        assert torch.isfinite(output).all()

    def test_model_with_zeros(self, simple_missile_model):
        """Test model produces consistent output with zero inputs using fixture."""
        # Create zero input
        x = torch.zeros(1, 9)

        # Get output from model
        output = simple_missile_model(x)

        # With fixed weights (0.01) and ReLU activation, the output should be predictable
        # The actual value is ~0.0205 due to bias terms
        expected = torch.tensor([[0.0205]])  # approximate expected value
        assert torch.allclose(output, expected, rtol=1e-2)
