import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class MissileDataset(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = json.load(f)

        self.samples = []
        self.weights = []  # store a weight per sample

        for entry in self.data:
            # We check for new keys:
            #   "player_x", "player_y", "enemy_x", "enemy_y",
            #   "missile_x", "missile_y", "missile_angle", "dist",
            #   "missile_collision", "missile_action"
            needed_keys = [
                "player_x",
                "player_y",
                "enemy_x",
                "enemy_y",
                "missile_x",
                "missile_y",
                "missile_angle",
                "dist",
                "missile_collision",
                "missile_action",
            ]
            if all(k in entry for k in needed_keys):
                # Convert boolean collision => float
                collision_val = 1.0 if entry["missile_collision"] else 0.0
                # Build a 9-element state
                state = [
                    entry["player_x"],
                    entry["player_y"],
                    entry["enemy_x"],
                    entry["enemy_y"],
                    entry["missile_x"],
                    entry["missile_y"],
                    entry["missile_angle"],
                    entry["dist"],
                    collision_val,
                ]
                # The network will learn to match this single float action
                action = [entry["missile_action"]]

                # weight = 2.0 for collision frames, 1.0 for non-collision
                wt = 2.0 if collision_val == 1.0 else 1.0

                self.samples.append((state, action))
                self.weights.append(wt)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action = self.samples[idx]
        weight = self.weights[idx]
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )


class SimpleMissileModel(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, output_size=1):
        super(SimpleMissileModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(filename):
    # 1) Load dataset
    dataset = MissileDataset(filename)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 2) Initialize model and optimizer
    model = SimpleMissileModel(input_size=9, hidden_size=64, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3) Training loop
    for epoch in range(20):
        running_loss = 0.0
        total_batches = 0
        for states, actions, weights in dataloader:
            optimizer.zero_grad()

            # Forward pass
            preds = model(states).view(-1)  # shape: (batch_size,)
            actions = actions.view(-1)  # shape: (batch_size,)
            weights = weights.view(-1)  # shape: (batch_size,)

            # Weighted MSE
            loss_per_sample = (preds - actions) ** 2 * weights
            loss = torch.mean(loss_per_sample)

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        # Print average loss per epoch
        avg_loss = running_loss / total_batches if total_batches > 0 else 0
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    # 4) Remove old model file if it exists (optional but helps ensure a fresh file)
    model_path = "models/missile_model.pth"
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Removed old file at '{model_path}'.")

    # 5) Save the newly trained model
    torch.save(model.state_dict(), "models/missile_model.pth")
    print(f"Saved new model to '{model_path}'.")


if __name__ == "__main__":
    train_model("data/raw/training_data.json")
