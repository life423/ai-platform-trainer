#!/usr/bin/env python
"""
Training Script for AI Models

This script trains the AI models using collected training data.
"""
import os
import sys
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ai_platform_trainer_new.ai.training.supervised import SupervisedTrainer


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    setup_logging()
    logging.info("Starting AI model training")
    
    try:
        # Create trainer
        trainer = SupervisedTrainer()
        
        # Train all models
        success = trainer.train_all_models()
        
        if success:
            logging.info("Training completed successfully!")
            return 0
        else:
            logging.error("Training failed!")
            return 1
            
    except Exception as e:
        logging.error(f"Training error: {e}")
        logging.exception("Exception details:")
        return 1


if __name__ == "__main__":
    sys.exit(main())