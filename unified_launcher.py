#!/usr/bin/env python
"""
Unified Launcher for AI Platform Trainer

This is a simplified launcher that directly runs the game without the complex
launcher system. It provides a clean entry point for the application.
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
        
        # Import and run the game directly
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