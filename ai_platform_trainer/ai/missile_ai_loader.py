"""
Missile AI Model Loader

This module handles loading pre-trained missile AI models and provides
a unified interface for creating intelligent homing missiles.
"""
import logging
import os
import sys
import torch
import numpy as np
from typing import Optional, Union

try:
    from stable_baselines3 import PPO, SAC
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. RL missile AI disabled.")

from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.entities.smart_missile import SmartMissile


class MissileAIManager:
    """Manages missile AI models and provides smart missiles to the game."""
    
    def __init__(self):
        self.neural_network_model: Optional[MissileModel] = None
        self.rl_model: Optional = None
        self.rl_algorithm: str = "PPO"  # Track which algorithm is loaded
        self.use_rl = False
        self.models_loaded = False
        
        # Load models on initialization
        self._load_models()
    
    def _load_models(self):
        """Load available missile AI models."""
        # Try to load RL model first (most advanced) - check SAC and PPO
        if STABLE_BASELINES_AVAILABLE:
            # Check for SAC models first (potentially better performance)
            sac_paths = [
                "models/missile_sac_model_final.zip",
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "missile_sac_model_final.zip"),
                os.path.join(getattr(sys, '_MEIPASS', os.getcwd()), "models", "missile_sac_model_final.zip"),
            ]
            
            # Check for PPO models (fallback)
            ppo_paths = [
                "models/missile_rl_model_final.zip",
                os.path.join(os.path.dirname(__file__), "..", "..", "models", "missile_rl_model_final.zip"),
                os.path.join(getattr(sys, '_MEIPASS', os.getcwd()), "models", "missile_rl_model_final.zip"),
            ]
            
            rl_model_loaded = False
            
            # Try SAC first
            for sac_path in sac_paths:
                if os.path.exists(sac_path):
                    try:
                        self.rl_model = SAC.load(sac_path)
                        self.rl_algorithm = "SAC"
                        self.use_rl = True
                        logging.info(f"✅ Loaded SAC missile AI model from {sac_path} - missiles will be extremely smart!")
                        rl_model_loaded = True
                        break
                    except Exception as e:
                        logging.warning(f"Failed to load SAC missile model from {sac_path}: {e}")
            
            # Try PPO if SAC not found
            if not rl_model_loaded:
                for ppo_path in ppo_paths:
                    if os.path.exists(ppo_path):
                        try:
                            self.rl_model = PPO.load(ppo_path)
                            self.rl_algorithm = "PPO"
                            self.use_rl = True
                            logging.info(f"✅ Loaded PPO missile AI model from {ppo_path} - missiles will be very smart!")
                            rl_model_loaded = True
                            break
                        except Exception as e:
                            logging.warning(f"Failed to load PPO missile model from {ppo_path}: {e}")
            
            if not rl_model_loaded:
                logging.info("ℹ️  No RL missile AI model found - will try neural network fallback")
        
        # Try to load neural network model (fallback)
        possible_nn_paths = [
            "models/missile_model.pth",  # Standard location
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "missile_model.pth"),  # Relative to module
            os.path.join(getattr(sys, '_MEIPASS', os.getcwd()), "models", "missile_model.pth"),  # PyInstaller bundle
        ]
        
        nn_model_loaded = False
        for nn_model_path in possible_nn_paths:
            if os.path.exists(nn_model_path):
                try:
                    self.neural_network_model = MissileModel()
                    self.neural_network_model.load_state_dict(
                        torch.load(nn_model_path, map_location="cpu")
                    )
                    self.neural_network_model.eval()
                    
                    if not self.use_rl:  # Only log this if RL isn't available
                        logging.info(f"✅ Loaded neural network missile AI model from {nn_model_path} - missiles will be smart!")
                    nn_model_loaded = True
                    break
                        
                except Exception as e:
                    logging.warning(f"Failed to load neural network missile model from {nn_model_path}: {e}")
        
        if not nn_model_loaded and not self.use_rl:
            logging.warning("⚠️  No neural network missile AI model found")
        
        # Check if any models were loaded
        if self.rl_model or self.neural_network_model:
            self.models_loaded = True
            if self.rl_model and self.neural_network_model:
                ai_type = f"{self.rl_algorithm} + Neural Network"
            elif self.rl_model:
                ai_type = f"{self.rl_algorithm}"
            else:
                ai_type = "Neural Network"
            logging.info(f"🎯 Missile AI system ready with {ai_type} guidance")
        else:
            logging.warning("⚠️  No missile AI models found - missiles will use basic homing")
    
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
        lifespan: int = 20000
    ) -> SmartMissile:
        """
        Create a smart missile with AI guidance.
        
        Args:
            Standard missile parameters
            
        Returns:
            SmartMissile with appropriate AI model loaded
        """
        return SmartMissile(
            x=x, y=y,
            target_x=target_x, target_y=target_y,
            speed=speed, vx=vx, vy=vy,
            birth_time=birth_time, lifespan=lifespan,
            ai_model=self.neural_network_model,
            rl_model=self.rl_model,
            use_rl=self.use_rl
        )
    
    def get_ai_info(self) -> str:
        """Get information about loaded AI models."""
        if not self.models_loaded:
            return "Basic homing (no AI models loaded)"
        
        if self.use_rl and self.rl_model:
            if self.rl_algorithm == "SAC":
                return "Advanced SAC AI (extremely smart homing)"
            else:
                return "Advanced PPO AI (very smart homing)"
        elif self.neural_network_model:
            return "Neural Network AI (smart homing)"
        else:
            return "Basic homing"
    
    def is_ai_available(self) -> bool:
        """Check if any AI models are available."""
        return self.models_loaded
    
    def get_best_available_model_type(self) -> str:
        """Get the best available model type."""
        if self.use_rl and self.rl_model:
            return self.rl_algorithm
        elif self.neural_network_model:
            return "Neural Network"
        else:
            return "Basic"


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
    lifespan: int = 20000
) -> SmartMissile:
    """
    Convenience function to create a smart missile.
    
    This function uses the global missile AI manager to create
    missiles with the best available AI guidance.
    """
    return missile_ai_manager.create_smart_missile(
        x, y, target_x, target_y, speed, vx, vy, birth_time, lifespan
    )


def get_missile_ai_status() -> str:
    """Get current missile AI status for display."""
    return missile_ai_manager.get_ai_info()


def check_and_train_missile_ai():
    """Check if missile AI models exist, and train if needed with loading screen."""
    global missile_ai_manager
    import pygame
    import threading
    import time
    import numpy as np
    
    # Check if we're running from a PyInstaller bundle (executable)
    if hasattr(sys, '_MEIPASS'):
        logging.info("🎮 Running from executable - AI models should be pre-bundled")
        # In bundled executable, models should already be included
        # Just verify they loaded properly in the manager
        if missile_ai_manager.models_loaded:
            logging.info("✅ Bundled AI models loaded successfully")
        else:
            logging.warning("⚠️  Bundled AI models not found - falling back to basic homing")
        return
    
    # Check if RL model already exists (priority model) - for development/source runs
    possible_rl_paths = [
        "models/missile_rl_model_final.zip",
        os.path.join(os.path.dirname(__file__), "..", "..", "models", "missile_rl_model_final.zip"),
    ]
    
    rl_model_exists = any(os.path.exists(path) for path in possible_rl_paths)
    
    if rl_model_exists:
        logging.info("Advanced RL missile AI model found - skipping training")
        return
    
    # RL model doesn't exist - need to train the advanced AI
    logging.info("No advanced RL missile AI found - training intelligent missile system")
    
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
        nonlocal training_complete, training_failed, progress, current_step, training_status, error_message
        
        try:
            from ai_platform_trainer.ai.training.train_missile_rl import MissileRLTrainer
            
            # Check if stable_baselines3 is available
            if not STABLE_BASELINES_AVAILABLE:
                error_message = "stable_baselines3 not available - cannot train missile AI"
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
            model = trainer.train(total_timesteps=total_steps, progress_callback=update_progress)
            
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
            progress_fill = pygame.Rect(150, 320, fill_width, 40)
            
            # Gradient effect
            for i in range(fill_width):
                color_intensity = min(255, 100 + (i / fill_width) * 155)
                color = (color_intensity // 3, color_intensity, color_intensity // 2)
                pygame.draw.line(screen, color, 
                               (150 + i, 320), (150 + i, 360))
        
        # Draw progress text
        progress_percent = f"{progress * 100:.1f}%"
        progress_text = font_medium.render(progress_percent, True, (255, 255, 255))
        progress_text_rect = progress_text.get_rect(center=(500, 340))
        screen.blit(progress_text, progress_text_rect)
        
        # Draw step counter
        step_text = font_small.render(f"Step {current_step:,} / {total_steps:,}", True, (150, 150, 150))
        step_rect = step_text.get_rect(center=(500, 380))
        screen.blit(step_text, step_rect)
        
        # Draw time elapsed
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_text = font_small.render(f"Elapsed: {minutes:02d}:{seconds:02d}", True, (150, 150, 150))
        time_rect = time_text.get_rect(center=(500, 410))
        screen.blit(time_text, time_rect)
        
        # Draw info text
        info_lines = [
            "This intelligent AI will make missiles chase enemies with precision",
            "Training happens only once and creates a smart homing system",
            "Please wait while the AI learns optimal missile trajectories"
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
        for i in range(180):  # Show for 3 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            screen.fill((20, 40, 30))
            
            title_text = font_large.render("Training Complete!", True, (100, 255, 100))
            title_rect = title_text.get_rect(center=(500, 250))
            screen.blit(title_text, title_rect)
            
            status_text = font_medium.render("Missile AI is now ready!", True, (255, 255, 255))
            status_rect = status_text.get_rect(center=(500, 320))
            screen.blit(status_text, status_rect)
            
            detail_text = font_small.render("Missiles will now intelligently chase and intercept enemies", True, (200, 255, 200))
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
        for i in range(240):  # Show for 4 seconds
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            screen.fill((40, 20, 20))
            
            title_text = font_large.render("Training Failed", True, (255, 100, 100))
            title_rect = title_text.get_rect(center=(500, 250))
            screen.blit(title_text, title_rect)
            
            status_text = font_medium.render("Using basic missile homing instead", True, (255, 255, 255))
            status_rect = status_text.get_rect(center=(500, 320))
            screen.blit(status_text, status_rect)
            
            if error_message:
                error_text = font_small.render(error_message[:60], True, (255, 200, 200))
                error_rect = error_text.get_rect(center=(500, 380))
                screen.blit(error_text, error_rect)
            
            continue_text = font_small.render("Starting game...", True, (150, 150, 150))
            continue_rect = continue_text.get_rect(center=(500, 450))
            screen.blit(continue_text, continue_rect)
            
            pygame.display.flip()
            clock.tick(60)