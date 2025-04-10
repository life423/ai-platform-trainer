import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ai_platform_trainer.ai_model.missile_dataset import MissileDataset
from ai_platform_trainer.ai_model.simple_missile_model import SimpleMissileModel


class MissileTrainer:
    """
    A class encapsulating the training logic for SimpleMissileModel.
    Useful when you want to preserve OOP style, store training state,
    or easily extend functionality (e.g. custom callbacks, advanced
    logging, checkpointing, etc.).
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
        Initialize the dataset, model, and other resources for training.
        
        :param filename: Path to the JSON dataset.
        :param epochs: Number of training epochs.
        :param batch_size: Training batch size.
        :param lr: Learning rate for the optimizer.
        :param model_save_path: Where to save the trained model weights.
        """
        self.filename = filename
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path

        # Initialize dataset and loader
        self.dataset = MissileDataset(json_file=self.filename)

        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model and optimizer
        self.model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def run_training(self) -> None:
        """
        Execute the main training loop, saving the model when complete.
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
                loss_per_sample = (preds - actions)**2 * weights
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


if __name__ == "__main__":
    trainer = MissileTrainer(
        filename="data/raw/training_data.json",
        epochs=20,
        batch_size=32,
        lr=0.001,
        model_save_path="models/missile_model.pth"
    )
    trainer.run_training()
