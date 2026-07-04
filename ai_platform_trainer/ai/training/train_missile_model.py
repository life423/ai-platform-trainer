"""
Train a missile trajectory prediction model using the SimpleMissileModel architecture.

This module provides a training pipeline for the missile trajectory prediction model,
including data loading, training loop, optimization, and model saving.
"""
import os
from typing import Dict, List, Optional

import torch
from torch import optim
from torch.utils.data import DataLoader

from ai_platform_trainer.ai.models.simple_missile_model import SimpleMissileModel
from ai_platform_trainer.ai.training.missile_dataset import MissileDataset


class MissileTrainer:
    """
    Trainer for the missile trajectory prediction model.

    Encapsulates the training logic for SimpleMissileModel, including dataset handling,
    optimization, and training loop execution. The class structure allows for easy extension
    with features like custom callbacks, advanced logging, or checkpointing.
    """

    def __init__(
        self,
        filename: str = "data/raw/training_data.json",
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
        model_save_path: str = "models/missile_model.pth",
    ) -> None:
        """
        Initialize the trainer with configurable training parameters.

        Args:
            filename: Path to the JSON dataset file
            epochs: Number of training epochs to run
            batch_size: Number of samples per training batch
            lr: Learning rate for the Adam optimizer
            model_save_path: File path where trained model will be saved
        """
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path

        # Initialize dataset and loader
        self.dataset = MissileDataset(json_file=self.filename)

        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # Initialize model and optimizer
        self.model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def run_training(self) -> Optional[Dict[str, List[float]]]:
        """
        Execute the main training loop and save the final model.

        Performs the complete training process, including:
        - Batch iteration through the dataset
        - Forward and backward passes
        - Optimization steps
        - Loss tracking
        - Model saving

        Returns:
            Optional dictionary containing training metrics (None for now,
            but could be extended to return loss history, etc.)
        """
        for epoch in range(self.epochs):
            running_loss = 0.0
            total_batches = 0

            for states, actions, weights in self.dataloader:
                self.optimizer.zero_grad()

                preds = self.model(states).view(-1)
                actions = actions.view(-1)
                weights = weights.view(-1)

                # Weighted MSE
                loss_per_sample = (preds - actions) ** 2 * weights
                loss = loss_per_sample.mean()

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_batches += 1

            avg_loss = running_loss / total_batches if total_batches > 0 else 0
            print(f"Epoch {epoch}/{self.epochs - 1}, Avg Loss: {avg_loss:.4f}")

        # Optionally remove old file
        if os.path.exists(self.model_save_path):
            os.remove(self.model_save_path)
            print(f"Removed old file at '{self.model_save_path}'.")

        # Save the model
        torch.save(self.model.state_dict(), self.model_save_path)
        print(f"Saved new model to '{self.model_save_path}'.")

        return None


if __name__ == "__main__":
    trainer = MissileTrainer(
        filename="data/raw/training_data.json",
        epochs=20,
        batch_size=32,
        lr=0.001,
        model_save_path="models/missile_model.pth",
    )
    trainer.run_training()
