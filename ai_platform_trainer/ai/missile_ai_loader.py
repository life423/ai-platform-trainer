"""
Missile AI Model Loader

This module handles loading pre-trained missile AI models and provides
a unified interface for creating intelligent homing missiles.
"""
import logging
import os
import sys
from typing import Any, Callable, List, Optional

import torch

try:
    from stable_baselines3 import PPO, SAC

    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL missile AI disabled.")

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
    Manages missile AI models and provides smart missiles to the game.

    Loads every available guidance model independently (rather than only
    the first one found) so the menu can let the player choose which one
    to play against, instead of always getting whichever model happened
    to win an automatic priority order.
    """

    def __init__(self):
        self.sac_model: Optional[Any] = None
        self.ppo_model: Optional[Any] = None
        self.neural_network_model: Optional[MissileModel] = None
        self.models_loaded = False

        self._load_models()

    def _candidate_paths(self, filename: str) -> List[str]:
        """Standard/relative/PyInstaller-bundle locations to check for a model file."""
        return [
            f"models/{filename}",
            os.path.join(os.path.dirname(__file__), "..", "..", "models", filename),
            os.path.join(getattr(sys, "_MEIPASS", os.getcwd()), "models", filename),
        ]

    def _load_rl_model(
        self, loader: Callable[[str], Any], filename: str, label: str
    ) -> Optional[Any]:
        """Try loading an RL model (SAC or PPO) from its candidate paths."""
        for path in self._candidate_paths(filename):
            if os.path.exists(path):
                try:
                    model = loader(path)
                    logging.info(f"✅ Loaded {label} missile AI model from {path}")
                    return model
                except Exception as e:
                    logging.warning(
                        f"Failed to load {label} missile model from {path}: {e}"
                    )
        return None

    def _load_nn_model(self) -> Optional[MissileModel]:
        """Try loading the supervised neural network model."""
        for path in self._candidate_paths("missile_model.pth"):
            if os.path.exists(path):
                try:
                    model = MissileModel()
                    model.load_state_dict(torch.load(path, map_location="cpu"))
                    model.eval()
                    logging.info(
                        f"✅ Loaded supervised neural network missile model from {path}"
                    )
                    return model
                except Exception as e:
                    logging.warning(
                        f"Failed to load neural network missile model from {path}: {e}"
                    )
        return None

    def _load_models(self) -> None:
        """Load every available missile AI model independently."""
        if STABLE_BASELINES_AVAILABLE:
            self.sac_model = self._load_rl_model(
                SAC.load, "missile_sac_model_final.zip", "SAC"
            )
            self.ppo_model = self._load_rl_model(
                PPO.load, "missile_rl_model_final.zip", "PPO"
            )
        else:
            logging.info("stable_baselines3 unavailable - skipping SAC/PPO loading")

        self.neural_network_model = self._load_nn_model()

        self.models_loaded = bool(
            self.sac_model or self.ppo_model or self.neural_network_model
        )
        if self.models_loaded:
            available = [
                label
                for key, label in MODEL_CHOICES.items()
                if getattr(self, self._attr_for(key)) is not None
            ]
            logging.info(f"🎯 Missile AI models available: {', '.join(available)}")
        else:
            logging.warning(
                "⚠️  No missile AI models found - missiles will use basic homing"
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
        Create a smart missile guided by the requested model.

        Args:
            model_choice: "sac", "ppo", or "supervised". Falls back through
                sac -> ppo -> supervised -> basic homing if the requested
                model isn't actually loaded.

        Returns:
            SmartMissile configured with the requested (or best available
            fallback) AI model loaded.
        """
        rl_model = None
        rl_algorithm = None
        ai_model = None

        if not self.is_model_available(model_choice):
            logging.warning(
                f"Requested missile model '{model_choice}' unavailable, "
                "falling back to the best one that is."
            )
            for fallback in ("sac", "ppo", "supervised"):
                if self.is_model_available(fallback):
                    model_choice = fallback
                    break

        if model_choice == "sac" and self.sac_model:
            rl_model, rl_algorithm = self.sac_model, "sac"
        elif model_choice == "ppo" and self.ppo_model:
            rl_model, rl_algorithm = self.ppo_model, "ppo"
        elif model_choice == "supervised" and self.neural_network_model:
            ai_model = self.neural_network_model

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
            ai_model=ai_model,
            rl_model=rl_model,
            rl_algorithm=rl_algorithm,
            use_rl=rl_model is not None,
        )

    def get_ai_info(self, model_choice: str = "sac") -> str:
        """Get a short status string for the given model choice."""
        label = MODEL_CHOICES.get(model_choice, model_choice)
        if self.is_model_available(model_choice):
            return f"{label} (loaded)"
        return f"{label} (not available - will fall back)"

    def is_ai_available(self) -> bool:
        """Check if any AI models are available."""
        return self.models_loaded


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


def get_missile_ai_status(model_choice: str = "sac") -> str:
    """Get current missile AI status for display."""
    return missile_ai_manager.get_ai_info(model_choice)


def check_and_train_missile_ai():
    """Check if missile AI models exist, and train if needed with loading screen."""
    global missile_ai_manager
    import threading
    import time

    import numpy as np
    import pygame

    # Check if we're running from a PyInstaller bundle (executable)
    if hasattr(sys, "_MEIPASS"):
        logging.info("🎮 Running from executable - AI models should be pre-bundled")
        # In bundled executable, models should already be included
        # Just verify they loaded properly in the manager
        if missile_ai_manager.models_loaded:
            logging.info("✅ Bundled AI models loaded successfully")
        else:
            logging.warning(
                "⚠️  Bundled AI models not found - falling back to basic homing"
            )
        return

    # Check if RL model already exists (priority model) - for development/source runs
    possible_rl_paths = [
        "models/missile_rl_model_final.zip",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "missile_rl_model_final.zip",
        ),
    ]

    rl_model_exists = any(os.path.exists(path) for path in possible_rl_paths)

    if rl_model_exists:
        logging.info("Advanced RL missile AI model found - skipping training")
        return

    # RL model doesn't exist - need to train the advanced AI
    logging.info(
        "No advanced RL missile AI found - training intelligent missile system"
    )

    # Show training screen
    screen = pygame.display.set_mode((1000, 700))
    pygame.display.set_caption("AI Platform Trainer - First Setup")
    clock = pygame.time.Clock()
    font_large = pygame.font.Font(None, 56)
    font_medium = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 32)

    # Training state variables
    training_complete = False
    training_failed = False
    progress = 0.0
    current_step = 0
    total_steps = 100000
    training_status = "Initializing training..."
    error_message = ""

    def training_thread():
        """Run training in background thread."""
        nonlocal training_complete, training_failed, progress
        nonlocal current_step, training_status, error_message

        try:
            from ai_platform_trainer.ai.training.train_missile_rl import (
                MissileRLTrainer,
            )

            # Check if stable_baselines3 is available
            if not STABLE_BASELINES_AVAILABLE:
                error_message = (
                    "stable_baselines3 not available - cannot train missile AI"
                )
                training_failed = True
                return

            training_status = "Creating training environment..."

            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            trainer = MissileRLTrainer(save_path="models/missile_rl_model")

            training_status = "Starting neural network training..."

            # Create progress update function
            def update_progress(current, total):
                nonlocal progress, current_step, training_status
                current_step = current
                progress = min(current / total, 1.0)

                # Update status based on progress
                if progress < 0.1:
                    training_status = "Learning basic movement..."
                elif progress < 0.3:
                    training_status = "Learning target tracking..."
                elif progress < 0.6:
                    training_status = "Learning interception strategies..."
                elif progress < 0.9:
                    training_status = "Optimizing missile trajectories..."
                else:
                    training_status = "Finalizing training..."

            # Start training with progress callback
            logging.info(f"Training missile AI with {total_steps:,} timesteps...")
            trainer.train(
                total_timesteps=total_steps, progress_callback=update_progress
            )

            training_status = "Training completed successfully!"
            training_complete = True

        except Exception as e:
            error_message = f"Training failed: {str(e)}"
            training_failed = True
            logging.error(error_message)

    # Start training thread
    training_thread_obj = threading.Thread(target=training_thread, daemon=False)
    training_thread_obj.start()

    # Main UI loop
    start_time = time.time()

    while not training_complete and not training_failed:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # Clear screen
        screen.fill((25, 35, 45))

        # Draw header
        title_text = font_large.render("AI Platform Trainer", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(500, 100))
        screen.blit(title_text, title_rect)

        subtitle_text = font_medium.render("First-Time Setup", True, (150, 200, 255))
        subtitle_rect = subtitle_text.get_rect(center=(500, 150))
        screen.blit(subtitle_text, subtitle_rect)

        # Draw main status
        status_text = font_medium.render("Training Missile AI", True, (100, 255, 150))
        status_rect = status_text.get_rect(center=(500, 220))
        screen.blit(status_text, status_rect)

        # Draw detailed status
        detail_text = font_small.render(training_status, True, (200, 200, 200))
        detail_rect = detail_text.get_rect(center=(500, 260))
        screen.blit(detail_text, detail_rect)

        # Draw progress bar background
        progress_bg = pygame.Rect(150, 320, 700, 40)
        pygame.draw.rect(screen, (60, 70, 80), progress_bg)
        pygame.draw.rect(screen, (100, 120, 140), progress_bg, 2)

        # Draw progress bar fill
        if progress > 0:
            fill_width = int(700 * progress)

            # Gradient effect
            for i in range(fill_width):
                color_intensity = min(255, 100 + (i / fill_width) * 155)
                color = (color_intensity // 3, color_intensity, color_intensity // 2)
                pygame.draw.line(screen, color, (150 + i, 320), (150 + i, 360))

        # Draw progress text
        progress_percent = f"{progress * 100:.1f}%"
        progress_text = font_medium.render(progress_percent, True, (255, 255, 255))
        progress_text_rect = progress_text.get_rect(center=(500, 340))
        screen.blit(progress_text, progress_text_rect)

        # Draw step counter
        step_text = font_small.render(
            f"Step {current_step:,} / {total_steps:,}", True, (150, 150, 150)
        )
        step_rect = step_text.get_rect(center=(500, 380))
        screen.blit(step_text, step_rect)

        # Draw time elapsed
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_text = font_small.render(
            f"Elapsed: {minutes:02d}:{seconds:02d}", True, (150, 150, 150)
        )
        time_rect = time_text.get_rect(center=(500, 410))
        screen.blit(time_text, time_rect)

        # Draw info text
        info_lines = [
            "This intelligent AI will make missiles chase enemies with precision",
            "Training happens only once and creates a smart homing system",
            "Please wait while the AI learns optimal missile trajectories",
        ]

        for i, line in enumerate(info_lines):
            info_text = font_small.render(line, True, (180, 180, 180))
            info_rect = info_text.get_rect(center=(500, 480 + i * 30))
            screen.blit(info_text, info_rect)

        # Draw animated elements
        spinner_angle = (time.time() * 180) % 360
        spinner_center = (500, 600)
        spinner_radius = 15
        for i in range(8):
            angle = spinner_angle + i * 45
            alpha = max(50, 255 - i * 30)
            x = spinner_center[0] + spinner_radius * np.cos(np.radians(angle))
            y = spinner_center[1] + spinner_radius * np.sin(np.radians(angle))
            pygame.draw.circle(screen, (alpha, alpha, alpha), (int(x), int(y)), 3)

        pygame.display.flip()
        clock.tick(60)

    # Show completion or error screen
    if training_complete:
        # Success screen
        for _ in range(180):  # Show for 3 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((20, 40, 30))

            title_text = font_large.render("Training Complete!", True, (100, 255, 100))
            title_rect = title_text.get_rect(center=(500, 250))
            screen.blit(title_text, title_rect)

            status_text = font_medium.render(
                "Missile AI is now ready!", True, (255, 255, 255)
            )
            status_rect = status_text.get_rect(center=(500, 320))
            screen.blit(status_text, status_rect)

            detail_text = font_small.render(
                "Missiles will now intelligently chase and intercept enemies",
                True,
                (200, 255, 200),
            )
            detail_rect = detail_text.get_rect(center=(500, 380))
            screen.blit(detail_text, detail_rect)

            continue_text = font_small.render("Starting game...", True, (150, 150, 150))
            continue_rect = continue_text.get_rect(center=(500, 450))
            screen.blit(continue_text, continue_rect)

            pygame.display.flip()
            clock.tick(60)

        logging.info("✅ First-time missile AI training completed successfully!")

        # Reload the missile AI manager to pick up the new RL model
        missile_ai_manager = MissileAIManager()

    elif training_failed:
        # Error screen
        for _ in range(240):  # Show for 4 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((40, 20, 20))

            title_text = font_large.render("Training Failed", True, (255, 100, 100))
            title_rect = title_text.get_rect(center=(500, 250))
            screen.blit(title_text, title_rect)

            status_text = font_medium.render(
                "Using basic missile homing instead", True, (255, 255, 255)
            )
            status_rect = status_text.get_rect(center=(500, 320))
            screen.blit(status_text, status_rect)

            if error_message:
                error_text = font_small.render(
                    error_message[:60], True, (255, 200, 200)
                )
                error_rect = error_text.get_rect(center=(500, 380))
                screen.blit(error_text, error_rect)

            continue_text = font_small.render("Starting game...", True, (150, 150, 150))
            continue_rect = continue_text.get_rect(center=(500, 450))
            screen.blit(continue_text, continue_rect)

            pygame.display.flip()
            clock.tick(60)
