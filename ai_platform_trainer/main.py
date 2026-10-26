"""
Main entry point for AI Platform Trainer.

This module provides a consistent entry point for the package.
"""
import logging
import os
import sys

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pygame  # noqa: E402

from ai_platform_trainer.core.logging_config import setup_logging  # noqa: E402
from ai_platform_trainer.gameplay.game_core import GameCore  # noqa: E402


def main() -> int:
    """Run the game. Returns the process exit status."""
    setup_logging(level="INFO", log_file="game.log")
    logging.info("Starting AI Platform Trainer from package entry point")

    try:
        pygame.init()
        GameCore().run()
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
