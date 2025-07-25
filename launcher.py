#!/usr/bin/env python
"""
Single Entry Point Launcher for AI Platform Trainer

This is the main launcher that provides a clean entry point for the application.
It replaces the multiple scattered launchers with a unified interface.
"""
import os
import sys
import logging
import pygame


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
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
        
        # Import and run the clean game system
        from src.ai_platform_trainer_new.game.core import Game
        
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