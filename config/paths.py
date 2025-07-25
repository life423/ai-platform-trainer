"""
File paths configuration for AI Platform Trainer.

This module centralizes all file path management for the application.
"""
import os


class PathConfig:
    """File path configuration."""
    
    # Project root directory
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Data paths
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    TRAINING_DATA_FILE = os.path.join(RAW_DATA_DIR, "training_data.json")
    
    # Model paths
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    ENEMY_MODELS_DIR = os.path.join(MODELS_DIR, "enemy")
    MISSILE_MODELS_DIR = os.path.join(MODELS_DIR, "missile")
    
    # Specific model files
    ENEMY_AI_MODEL = os.path.join(ENEMY_MODELS_DIR, "enemy_ai_model.pth")
    ENEMY_RL_MODEL = os.path.join(ENEMY_MODELS_DIR, "enemy_rl_final.zip")
    MISSILE_MODEL = os.path.join(MISSILE_MODELS_DIR, "missile_model.pth")
    
    # Legacy paths for backward compatibility
    LEGACY_ENEMY_MODEL = os.path.join(MODELS_DIR, "enemy_ai_model.pth")
    LEGACY_MISSILE_MODEL = os.path.join(MODELS_DIR, "missile_model.pth")
    LEGACY_RL_MODEL = os.path.join(MODELS_DIR, "enemy_rl", "final_model.zip")
    
    # Asset paths
    ASSETS_DIR = os.path.join(ROOT_DIR, "assets")
    SPRITES_DIR = os.path.join(ASSETS_DIR, "sprites")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all necessary directories exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR, 
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.ENEMY_MODELS_DIR,
            cls.MISSILE_MODELS_DIR,
            cls.ASSETS_DIR,
            cls.SPRITES_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


# Create singleton instance
paths = PathConfig()