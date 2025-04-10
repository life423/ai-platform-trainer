import torch
import torch.nn as nn


class EnemyMovementModel(nn.Module):
    def __init__(
        self, input_size=5, hidden_size=128, output_size=2, dropout_prob=0.3
    ):  # Adjusted dropout_prob
        super(EnemyMovementModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Now using the potentially adjusted value
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.fc1.in_features:
            raise ValueError(
                f"Expected input to have shape (batch_size, {self.fc1.in_features})."
            )

        x = nn.functional.leaky_relu(
            self.bn1(self.fc1(x)), negative_slope=0.01
        )  # Leaky ReLU
        x = self.dropout(x)
        x = nn.functional.leaky_relu(
            self.bn2(self.fc2(x)), negative_slope=0.01
        )  # Leaky ReLU
        x = self.dropout(x)
        x = nn.functional.leaky_relu(
            self.bn3(self.fc3(x)), negative_slope=0.01
        )  # Leaky ReLU
        x = self.dropout(x)
        x = self.fc4(x)
        return x
