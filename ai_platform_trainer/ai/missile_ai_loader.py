"""
Missile AI Model Loader

Loads the trained missile guidance models and creates the missiles that use
them. Nothing here falls back to a scripted or basic behavior: a missile
whose model is missing or failed to load raises MissingModelError, and the
menu reports it.
"""
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import torch

try:
    from stable_baselines3 import PPO, SAC

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL missile AI disabled.")

from ai_platform_trainer.ai.model_paths import MissingModelError, missing_model_message
from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.entities.smart_missile import SmartMissile

# Display names for each selectable guidance model, keyed the same way as
# model_choice throughout this module and the menu.
MODEL_CHOICES = {
    "sac": "SAC",
    "ppo": "PPO",
    "supervised": "Supervised NN",
}


class MissileAIManager:
    """
    Loads every missile guidance model independently, so the menu can offer
    each one and report exactly which ones are missing or failed to load.
    """

    def __init__(self):
        self.sac_model: Optional[Any] = None
        self.ppo_model: Optional[Any] = None
        self.neural_network_model: Optional[MissileModel] = None
        # Why a model that is on disk could not be loaded, keyed by model_choice
        self.load_errors: Dict[str, str] = {}

        self._load_models()

    def _candidate_paths(self, filename: str) -> List[str]:
        """Standard/relative/PyInstaller-bundle locations to check for a model file."""
        return [
            f"models/{filename}",
            os.path.join(os.path.dirname(__file__), "..", "..", "models", filename),
            os.path.join(getattr(sys, "_MEIPASS", os.getcwd()), "models", filename),
        ]

    def _find(self, filename: str) -> Optional[str]:
        for path in self._candidate_paths(filename):
            if os.path.exists(path):
                return path
        return None

    def _load_rl_model(
        self, loader: Callable[[str], Any], filename: str, key: str
    ) -> Optional[Any]:
        """Load an RL model (SAC or PPO). A file that exists but won't load is recorded."""
        path = self._find(filename)
        if path is None:
            return None
        try:
            model = loader(path)
        except Exception as e:
            self.load_errors[key] = f"{MODEL_CHOICES[key]} model at {path} failed to load: {e}"
            logging.error(self.load_errors[key])
            return None
        logging.info(f"Loaded {MODEL_CHOICES[key]} missile AI model from {path}")
        return model

    def _load_nn_model(self) -> Optional[MissileModel]:
        """Load the supervised neural network model."""
        path = self._find("missile_model.pth")
        if path is None:
            return None
        try:
            model = MissileModel()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
        except Exception as e:
            self.load_errors["supervised"] = (
                f"Supervised NN model at {path} failed to load: {e}"
            )
            logging.error(self.load_errors["supervised"])
            return None
        logging.info(f"Loaded supervised neural network missile model from {path}")
        return model

    def _load_models(self) -> None:
        """Load every available missile AI model independently."""
        if STABLE_BASELINES_AVAILABLE:
            self.sac_model = self._load_rl_model(
                SAC.load, "missile_sac_model_final.zip", "sac"
            )
            self.ppo_model = self._load_rl_model(
                PPO.load, "missile_rl_model_final.zip", "ppo"
            )
        else:
            self.load_errors["sac"] = "stable-baselines3 is not installed"
            self.load_errors["ppo"] = "stable-baselines3 is not installed"

        self.neural_network_model = self._load_nn_model()

        available = [key for key in MODEL_CHOICES if self.is_model_available(key)]
        logging.info(
            "Missile AI models available: "
            + (", ".join(MODEL_CHOICES[key] for key in available) or "none")
        )

    @staticmethod
    def _attr_for(model_choice: str) -> str:
        return {
            "sac": "sac_model",
            "ppo": "ppo_model",
            "supervised": "neural_network_model",
        }[model_choice]

    def is_model_available(self, model_choice: str) -> bool:
        """Whether the given model_choice ("sac"/"ppo"/"supervised") is loaded."""
        return getattr(self, self._attr_for(model_choice), None) is not None

    def problem_with(self, model_choice: str) -> Optional[str]:
        """Why this model cannot be used right now, or None if it can."""
        if self.is_model_available(model_choice):
            return None
        if model_choice in self.load_errors:
            return self.load_errors[model_choice]
        return missing_model_message(model_choice)

    def create_smart_missile(
        self,
        x: int,
        y: int,
        target_x: float = 0.0,
        target_y: float = 0.0,
        speed: float = 8.0,
        vx: float = 8.0,
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 20000,
        model_choice: str = "sac",
    ) -> SmartMissile:
        """
        Create a smart missile guided by exactly the requested model.

        Raises:
            MissingModelError: if that model is not loaded. There is no
                fallback to another model or to basic homing.
        """
        if model_choice not in MODEL_CHOICES:
            raise ValueError(f"Unknown missile model '{model_choice}'")
        problem = self.problem_with(model_choice)
        if problem is not None:
            raise MissingModelError(problem)

        if model_choice == "supervised":
            return SmartMissile(
                x=x,
                y=y,
                target_x=target_x,
                target_y=target_y,
                speed=speed,
                vx=vx,
                vy=vy,
                birth_time=birth_time,
                lifespan=lifespan,
                ai_model=self.neural_network_model,
            )

        rl_model = self.sac_model if model_choice == "sac" else self.ppo_model
        return SmartMissile(
            x=x,
            y=y,
            target_x=target_x,
            target_y=target_y,
            speed=speed,
            vx=vx,
            vy=vy,
            birth_time=birth_time,
            lifespan=lifespan,
            rl_model=rl_model,
            rl_algorithm=model_choice,
        )


# Global missile AI manager instance
missile_ai_manager = MissileAIManager()


def create_smart_missile(
    x: int,
    y: int,
    target_x: float = 0.0,
    target_y: float = 0.0,
    speed: float = 8.0,
    vx: float = 8.0,
    vy: float = 0.0,
    birth_time: int = 0,
    lifespan: int = 20000,
    model_choice: str = "sac",
) -> SmartMissile:
    """
    Convenience function to create a smart missile.

    This function uses the global missile AI manager to create
    missiles with the requested (model_choice) AI guidance.
    """
    return missile_ai_manager.create_smart_missile(
        x, y, target_x, target_y, speed, vx, vy, birth_time, lifespan, model_choice
    )
