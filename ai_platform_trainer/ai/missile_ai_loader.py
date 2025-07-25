"""
Missile AI Model Loader

This module handles loading pre-trained missile AI models and provides
a unified interface for creating intelligent homing missiles.
"""
import logging
import os
import torch
from typing import Optional, Union

try:
    from stable_baselines3 import PPO
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
        self.use_rl = False
        self.models_loaded = False
        
        # Load models on initialization
        self._load_models()
    
    def _load_models(self):
        """Load available missile AI models."""
        # Try to load RL model first (most advanced)
        if STABLE_BASELINES_AVAILABLE:
            rl_model_path = "models/missile_rl_model_final.zip"
            if os.path.exists(rl_model_path):
                try:
                    self.rl_model = PPO.load(rl_model_path)
                    self.use_rl = True
                    logging.info("✅ Loaded RL missile AI model - missiles will be very smart!")
                except Exception as e:
                    logging.error(f"Failed to load RL missile model: {e}")
        
        # Try to load neural network model (fallback)
        nn_model_path = "models/missile_model.pth"
        if os.path.exists(nn_model_path):
            try:
                self.neural_network_model = MissileModel()
                self.neural_network_model.load_state_dict(
                    torch.load(nn_model_path, map_location="cpu")
                )
                self.neural_network_model.eval()
                
                if not self.use_rl:  # Only log this if RL isn't available
                    logging.info("✅ Loaded neural network missile AI model - missiles will be smart!")
                    
            except Exception as e:
                logging.error(f"Failed to load neural network missile model: {e}")
        
        # Check if any models were loaded
        if self.rl_model or self.neural_network_model:
            self.models_loaded = True
            ai_type = "RL + Neural Network" if self.rl_model and self.neural_network_model else \
                     "RL" if self.rl_model else "Neural Network"
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
            return "Advanced RL AI (very smart homing)"
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
            return "RL"
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
    import pygame
    
    # Check if models already exist
    rl_model_path = "models/missile_rl_model_final.zip"
    nn_model_path = "models/missile_model.pth"
    
    if os.path.exists(rl_model_path) or os.path.exists(nn_model_path):
        logging.info("Missile AI models found - skipping training")
        return
    
    # Models don't exist - need to train
    logging.info("No missile AI models found - starting first-time training")
    
    # Show training screen
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("AI Platform Trainer - First Setup")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 32)
    
    # Start training in background
    try:
        from ai_platform_trainer.ai.training.train_missile_rl import MissileRLTrainer
        
        # Check if stable_baselines3 is available
        if not STABLE_BASELINES_AVAILABLE:
            logging.error("stable_baselines3 not available - cannot train missile AI")
            return
        
        trainer = MissileRLTrainer(save_path="models/missile_rl_model")
        total_timesteps = 100000  # More training for much better performance
        
        # Training loop with UI
        training_complete = False
        progress = 0.0
        
        def progress_callback(current_step, total_steps):
            nonlocal progress
            progress = current_step / total_steps
        
        # Start training in a separate thread would be ideal, but for simplicity
        # we'll train with periodic UI updates
        logging.info(f"Training missile AI with {total_timesteps:,} timesteps...")
        
        # Show initial screen
        for i in range(60):  # Show for 1 second
            screen.fill((20, 30, 40))
            
            # Title
            title_text = font.render("Setting up AI Platform Trainer", True, (255, 255, 255))
            title_rect = title_text.get_rect(center=(400, 200))
            screen.blit(title_text, title_rect)
            
            # Status
            status_text = small_font.render("Training Missile AI...", True, (100, 200, 255))
            status_rect = status_text.get_rect(center=(400, 280))
            screen.blit(status_text, status_rect)
            
            # Progress bar background
            progress_bg = pygame.Rect(200, 350, 400, 30)
            pygame.draw.rect(screen, (60, 60, 60), progress_bg)
            
            # Progress bar fill (animated while loading)
            fill_width = int(400 * ((i % 60) / 60.0))
            if fill_width > 0:
                progress_fill = pygame.Rect(200, 350, fill_width, 30)
                pygame.draw.rect(screen, (100, 200, 255), progress_fill)
            
            # Info text
            info_text = small_font.render("This only happens once and takes about 2-3 minutes", True, (200, 200, 200))
            info_rect = info_text.get_rect(center=(400, 420))
            screen.blit(info_text, info_rect)
            
            pygame.display.flip()
            clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        # Actually train the model (this will take time)
        model = trainer.train(total_timesteps=total_timesteps)
        
        # Show completion screen
        for i in range(120):  # Show for 2 seconds
            screen.fill((20, 30, 40))
            
            # Title
            title_text = font.render("Training Complete!", True, (100, 255, 100))
            title_rect = title_text.get_rect(center=(400, 250))
            screen.blit(title_text, title_rect)
            
            # Status
            status_text = small_font.render("Missile AI is now ready - missiles will chase enemies intelligently!", True, (255, 255, 255))
            status_rect = status_text.get_rect(center=(400, 320))
            screen.blit(status_text, status_rect)
            
            # Continue prompt
            continue_text = small_font.render("Starting game...", True, (200, 200, 200))
            continue_rect = continue_text.get_rect(center=(400, 380))
            screen.blit(continue_text, continue_rect)
            
            pygame.display.flip()
            clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        
        logging.info("✅ First-time missile AI training completed successfully!")
        
    except Exception as e:
        logging.error(f"Failed to train missile AI: {e}")
        
        # Show error screen
        for i in range(180):  # Show for 3 seconds
            screen.fill((40, 20, 20))
            
            # Title
            title_text = font.render("Training Failed", True, (255, 100, 100))
            title_rect = title_text.get_rect(center=(400, 250))
            screen.blit(title_text, title_rect)
            
            # Status
            status_text = small_font.render("Missiles will use basic homing instead", True, (255, 255, 255))
            status_rect = status_text.get_rect(center=(400, 320))
            screen.blit(status_text, status_rect)
            
            pygame.display.flip()
            clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return