import json
import torch
from torch.utils.data import Dataset
import math

def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

class MissileDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        entry = self.data[idx]
        collision_val = 1.0 if entry["missile_collision"] else 0.0

        # Calculate distance on the fly
        dist = calculate_distance(
            entry["missile_x"], entry["missile_y"], entry["enemy_x"], entry["enemy_y"]
        )

        state = [
            entry["player_x"],
            entry["player_y"],
            entry["enemy_x"],
            entry["enemy_y"],
            entry["missile_x"],
            entry["missile_y"],
            entry["missile_angle"],
            dist,
            collision_val,
        ]
        action = [entry["missile_action"]]
        weight = 2.0 if collision_val == 1.0 else 1.0 #Corrected line: weight is now defined
        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )
