"""Where trained agents live, shared by the trainers, the CLI and the game."""
import json
import os
from typing import Any, Dict, Optional

ENEMY_PPO_DIR = "models/enemy_ppo"
ENEMY_PPO_MODEL = "models/enemy_ppo/enemy_ppo.zip"
ENEMY_PPO_LOG_DIR = "logs/enemy_ppo"
MISSILE_SAC_MODEL = "models/missile_sac_model_final.zip"
BENCHMARK_DIR = "reports/benchmarks"


class MissingModelError(RuntimeError):
    """A trained model the game needs is not available.

    Raised instead of falling back to a scripted or basic behavior, so a
    missing model is always visible.
    """


MODEL_NAMES = {
    "enemy": "AI Agent",
    "sac": "SAC",
}

# The command that produces each model, shown when one is missing.
TRAINING_COMMANDS = {
    "enemy": "python -m ai_platform_trainer train-enemy",
    "sac": "python -m ai_platform_trainer.ai.training.train_missile_sac",
}


def missing_model_message(key: str) -> str:
    """Explain which model is missing and the command that trains it."""
    return (
        f"{MODEL_NAMES[key]} is not trained yet. "
        f"Train it with: {TRAINING_COMMANDS[key]}"
    )


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
