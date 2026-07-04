"""
Integration tests for the AI training pipeline.

Tests the complete flow from data loading to model training and saving,
ensuring all components work together correctly.
"""
import os
import tempfile
from pathlib import Path  # noqa: F401 - May be used in future tests

import pytest
import torch

from ai_platform_trainer.ai.models.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.ai.training.missile_dataset import MissileDataset
from ai_platform_trainer.ai.training.train_missile_model import MissileTrainer


@pytest.mark.integration
class TestTrainingPipeline:
    """Test the complete training pipeline from data to model."""

    def test_end_to_end_training(self, sample_training_json_path):
        """
        Test the entire training pipeline end to end.

        This test verifies that:
        1. Data can be loaded correctly
        2. Model can be instantiated
        3. Training can be performed
        4. Model can be saved
        5. Saved model can be loaded and used for inference
        """
        # Create temporary directory for saving model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")

            # Create and run trainer
            trainer = MissileTrainer(
                filename=sample_training_json_path,
                epochs=2,  # Use small number of epochs for testing
                batch_size=1,
                lr=0.01,
                model_save_path=model_path,
            )

            # Run training
            trainer.run_training()

            # Verify model file exists
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0

            # Load the saved model and verify it works
            model = SimpleMissileModel()
            model.load_state_dict(torch.load(model_path))

            # Test inference
            x = torch.rand(1, 9)  # Random input
            with torch.no_grad():
                output = model(x)

            # Check output shape and type
            assert output.shape == (1, 1)
            assert isinstance(output, torch.Tensor)
            assert output.dtype == torch.float32

    def test_dataset_to_dataloader_integration(self, sample_training_json_path):
        """
        Test integration between MissileDataset and DataLoader.

        Verifies that the dataset can be used with PyTorch's DataLoader
        for batch processing during training.
        """
        # Create dataset
        dataset = MissileDataset(json_file=sample_training_json_path)

        # Create data loader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

        # Iterate through all batches
        batch_count = 0
        for states, actions, weights in dataloader:
            # Check shapes
            assert states.dim() == 2  # [batch_size, features]
            assert states.shape[1] == 9  # 9 features
            assert actions.shape[1] == 1  # 1 output value
            assert weights.shape[1] == 1  # 1 weight value

            batch_count += 1

        # Verify we got the expected number of batches
        # With a dataset of 2 items and batch_size=2, expect 1 batch
        assert batch_count == 1

    def test_trainer_model_integration(self, sample_training_json_path):
        """
        Test integration between Trainer and Model components.

        Verifies that the trainer can properly initialize and train the model.
        """
        # Create temporary directory for saving model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")

            # Create trainer
            trainer = MissileTrainer(
                filename=sample_training_json_path,
                epochs=1,
                batch_size=1,
                lr=0.01,
                model_save_path=model_path,
            )

            # Verify model was initialized correctly
            assert isinstance(trainer.model, SimpleMissileModel)
            assert trainer.model.fc1.in_features == 9
            assert trainer.model.fc3.out_features == 1

            # Verify optimizer was initialized with the model's parameters
            for param_group in trainer.optimizer.param_groups:
                for param in param_group["params"]:
                    # Check parameter is part of the model
                    model_params = {id(p) for p in trainer.model.parameters()}
                    assert id(param) in model_params
