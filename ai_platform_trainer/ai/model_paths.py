"""Where trained agents live, shared by the trainers, the CLI and the game."""
import json
import os
from typing import Any, Dict, Optional

ENEMY_PPO_DIR = "models/enemy_ppo"
ENEMY_PPO_MODEL = "models/enemy_ppo/enemy_ppo.zip"
ENEMY_PPO_LOG_DIR = "logs/enemy_ppo"
BENCHMARK_DIR = "reports/benchmarks"


def model_card_path(model_path: str) -> str:
    """The JSON model card that lives next to a saved model."""
    return os.path.splitext(model_path)[0] + ".json"


def load_model_card(model_path: str) -> Optional[Dict[str, Any]]:
    """The model card for a saved model, or None if it has none."""
    path = model_card_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
