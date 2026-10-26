"""
Missile AI model loader.

Loads the trained SAC missile model and creates the missiles that use it.
Nothing here falls back to a scripted or basic behavior: a missile whose
model is missing or failed to load raises MissingModelError, and the menu
reports it.
"""
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

try:
    from stable_baselines3 import SAC

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL missile AI disabled.")

from ai_platform_trainer.ai.model_paths import (
    MODEL_NAMES,
    MissingModelError,
    missing_model_message,
)
from ai_platform_trainer.entities.smart_missile import SmartMissile

SAC_KEY = "sac"
SAC_FILENAME = "missile_sac_model_final.zip"


class MissileAIManager:
    """Loads the SAC missile model, so the menu can report whether it is missing."""

    def __init__(self) -> None:
        self.sac_model: Optional[Any] = None
        # Why a model that is on disk could not be loaded, keyed by "sac"
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
        """Load an RL model. A file that exists but won't load is recorded."""
        path = self._find(filename)
        if path is None:
            return None
        try:
            model = loader(path)
        except Exception as e:
            self.load_errors[key] = (
                f"{MODEL_NAMES[key]} model at {path} failed to load: {e}"
            )
            logging.error(self.load_errors[key])
            return None
        logging.info(f"Loaded {MODEL_NAMES[key]} missile AI model from {path}")
        return model

    def _load_models(self) -> None:
        if STABLE_BASELINES_AVAILABLE:
            self.sac_model = self._load_rl_model(SAC.load, SAC_FILENAME, SAC_KEY)
        else:
            self.load_errors[SAC_KEY] = "stable-baselines3 is not installed"

        logging.info(
            "Missile AI model available: "
            + ("SAC" if self.is_model_available() else "none")
        )

    def is_model_available(self) -> bool:
        """Whether the SAC missile model is loaded."""
        return self.sac_model is not None

    def problem_with(self) -> Optional[str]:
        """Why the SAC missile cannot be used right now, or None if it can."""
        if self.is_model_available():
            return None
        if SAC_KEY in self.load_errors:
            return self.load_errors[SAC_KEY]
        return missing_model_message(SAC_KEY)

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
    ) -> SmartMissile:
        """
        Create a smart missile guided by the SAC model.

        Raises:
            MissingModelError: if the SAC model is not loaded. There is no
                fallback to basic homing.
        """
        problem = self.problem_with()
        if problem is not None:
            raise MissingModelError(problem)

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
            rl_model=self.sac_model,
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
) -> SmartMissile:
    """Create a smart missile guided by the SAC model, through the global manager."""
    return missile_ai_manager.create_smart_missile(
        x, y, target_x, target_y, speed, vx, vy, birth_time, lifespan
    )
