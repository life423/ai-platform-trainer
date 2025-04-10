"""
Train a missile trajectory prediction model using the SimpleMissileModel architecture.

This module provides a training pipeline for the missile trajectory prediction model,
including data loading, training loop, optimization, and model saving.
"""
import os
import logging
import argparse
from typing import Optional, List, Dict

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from ai_platform_trainer.ml.training.missile_dataset import MissileDataset
from ai_platform_trainer.ml.models.missile_model import SimpleMissileModel
from ai_platform_trainer.utils.environment import get_device


class MissileTrainer:
    """
    Trainer for the missile trajectory prediction model.

    Encapsulates the training logic for SimpleMissileModel, including dataset handling,
    optimization, training loop execution, and validation. The class implements early
    stopping, device-aware training (CPU/GPU), and comprehensive logging.
    """

    def __init__(
        self,
        filename: str = "data/raw/training_data.json",
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        model_save_path: str = "models/missile_model.pth",
        patience: int = 10,
        validation_split: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize the trainer with configurable training parameters.

        Args:
            filename: Path to the JSON dataset file
            epochs: Maximum number of training epochs to run
            batch_size: Number of samples per training batch
            lr: Learning rate for the Adam optimizer
            model_save_path: File path where trained model will be saved
            patience: Number of epochs to wait for validation improvement before early stopping
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
            device: Device to use for training (None for auto-detection)
        """
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path
        self.patience = patience
        self.validation_split = validation_split
        
        # Auto-detect device if not provided
        self.device = device if device is not None else get_device()
        logging.info(f"Using device: {self.device}")

        # Initialize dataset
        self.dataset = MissileDataset(json_file=self.filename)
        
        # Split dataset for training and validation
        dataset_size = len(self.dataset)
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = random_split(
            self.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0  # Adjust based on system capabilities
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )

        # Initialize model and move to appropriate device
        self.model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Track best model and metrics
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, actions, weights in self.train_loader:
            # Move data to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            weights = weights.to(self.device)
            
            # Reset gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            preds = self.model(states).view(-1)
            actions = actions.view(-1)
            weights = weights.view(-1)
            
            # Weighted MSE loss
            loss_per_sample = (preds - actions)**2 * weights
            loss = loss_per_sample.mean()
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def validate(self) -> float:
        """
        Evaluate the model on the validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, actions, weights in self.val_loader:
                # Move data to device
                states = states.to(self.device)
                actions = actions.to(self.device)
                weights = weights.to(self.device)
                
                # Forward pass
                preds = self.model(states).view(-1)
                actions = actions.view(-1)
                weights = weights.view(-1)
                
                # Weighted MSE loss
                loss_per_sample = (preds - actions)**2 * weights
                loss = loss_per_sample.mean()
                
                # Track metrics
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def run_training(self) -> Dict[str, List[float]]:
        """
        Execute the main training loop with early stopping and model saving.

        Performs the complete training process, including:
        - Batch iteration through the dataset
        - Forward and backward passes
        - Optimization steps
        - Validation
        - Early stopping
        - Model saving

        Returns:
            Dictionary containing training metrics (train_loss, val_loss)
        """
        logging.info(
            f"Starting training for {self.epochs} epochs "
            f"(early stopping patience: {self.patience})"
        )
        logging.info(
            f"Training on {len(self.train_loader.dataset)} samples, "
            f"validating on {len(self.val_loader.dataset)} samples"
        )
        
        for epoch in range(self.epochs):
            # Train one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Track metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            # Log progress
            logging.info(
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                logging.info(f"New best model (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1
                logging.info(f"No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping check
                if self.epochs_without_improvement >= self.patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logging.info("Restored best model for saving")
        
        # Save the model
        self._save_model()
        
        return self.training_history

    def _save_model(self) -> None:
        """
        Save the trained model to disk.
        
        Creates necessary directories if they don't exist and
        handles overwriting of existing model files.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Check for existing file
        if os.path.exists(self.model_save_path):
            logging.info(f"Removing existing model file at '{self.model_save_path}'")
            os.remove(self.model_save_path)
        
        # Save model
        torch.save(self.model.state_dict(), self.model_save_path)
        logging.info(f"Saved model to '{self.model_save_path}'")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the training script.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Train missile trajectory prediction model")
    
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/raw/training_data.json",
        help="Path to training data JSON file"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Maximum number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="models/missile_model.pth",
        help="Path to save trained model"
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=10,
        help="Early stopping patience (epochs)"
    )
    
    parser.add_argument(
        "--validation-split", 
        type=float, 
        default=0.2,
        help="Fraction of data for validation (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="Force using CPU even if GPU is available"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/training/missile_model.log', mode='w')
        ]
    )

    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cpu") if args.cpu else get_device()
    
    # Initialize and run trainer
    trainer = MissileTrainer(
        filename=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_save_path=args.output,
        patience=args.patience,
        validation_split=args.validation_split,
        device=device
    )
    
    # Run training
    training_history = trainer.run_training()
    
    logging.info("Training completed successfully")
