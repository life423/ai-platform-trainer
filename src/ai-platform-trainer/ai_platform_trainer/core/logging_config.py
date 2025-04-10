import logging
import sys
from pathlib import Path

LOG_FILE = "game.log"

def setup_logging() -> logging.Logger:
    """
    Sets up a shared logger that writes to both file and console.
    Returns the configured logger instance for convenience.
    """
    logger = logging.getLogger("ai_platform_trainer")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if this is called more than once
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_format)

        # Stream (console) handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_format = logging.Formatter("%(levelname)s - %(message)s")
        stream_handler.setFormatter(stream_format)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger