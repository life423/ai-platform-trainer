#!/usr/bin/env python
"""
Main launcher for AI Platform Trainer.

This is the primary entry point for the application.
"""
import os
import sys
import logging
import pygame


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("game.log"),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for the AI Platform Trainer."""
    # Setup logging
    setup_logging()
    logging.info("Starting AI Platform Trainer")
    
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        # Initialize pygame
        pygame.init()
        
        # Check and train missile AI if needed
        from ai_platform_trainer.ai.missile_ai_loader import check_and_train_missile_ai
        check_and_train_missile_ai()
        
        # Import and run the preferred game system with light blue GUI
        from ai_platform_trainer.gameplay.game import Game
        
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