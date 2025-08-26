#!/usr/bin/env python
"""
Main launcher for AI Platform Trainer.

This is the primary entry point for the application.
"""
import os
import sys
import logging
import pygame

# Add the project root to sys.path to allow for module imports.
# This is a common pattern for making a project runnable without installation.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_platform_trainer.ai.missile_ai_loader import check_and_train_missile_ai
from ai_platform_trainer.gameplay.game_core import GameCore as Game


def setup_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("game.log"),
            logging.StreamHandler()
        ],
        force=True  # In Python 3.8+, this allows re-configuration of logging.
    )


def main() -> int:
    """Main entry point for the AI Platform Trainer."""
    setup_logging()
    logging.info("Starting AI Platform Trainer")

    try:
        pygame.init()
        check_and_train_missile_ai()

        # Create and run the game
        game = Game()
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
