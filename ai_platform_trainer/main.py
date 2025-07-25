"""
Main entry point for AI Platform Trainer.

This module provides a consistent entry point for the package.
"""
import os
import sys

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_platform_trainer.gameplay.game_core import GameCore
from ai_platform_trainer.ai.missile_ai_loader import check_and_train_missile_ai
import pygame
import logging


def main():
    """Main entry point for the AI Platform Trainer."""
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("game.log"),
            logging.StreamHandler()
        ]
    )
    logging.info("Starting AI Platform Trainer from package entry point")
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Check and train missile AI if needed
        check_and_train_missile_ai()
        
        # Create and run the game
        game = GameCore()
        game.run()
        
        logging.info("Game completed successfully")
        return 0
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        logging.exception("Exception details:")
        return 1
    finally:
        pygame.quit()
        logging.info("Exiting AI Platform Trainer")


if __name__ == "__main__":
    sys.exit(main())