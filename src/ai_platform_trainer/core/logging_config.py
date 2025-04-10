"""
Logging configuration module for AI Platform Trainer.

This module sets up the logging system for the application, including log
levels, formatting, and output destinations.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union


def ensure_log_directory(path: Union[str, Path]) -> Path:
    """
    Ensure the log directory exists.

    Args:
        path: Path to the log directory

    Returns:
        Path object for the log directory
    """
    log_dir = Path(path)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_dir: str = "logs",
    module_levels: Optional[Dict[str, str]] = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Default log level for all loggers
        log_file: Path to log file (within log_dir, if specified)
        console: Whether to log to console
        log_dir: Directory to store log files
        module_levels: Dict of module names to specific log levels
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Ensure the log directory exists
        log_path = ensure_log_directory(log_dir)
        file_path = log_path / log_file

        file_handler = RotatingFileHandler(
            file_path, maxBytes=10 * 1024 * 1024, backupCount=5, mode="a"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure specific module loggers if requested
    if module_levels:
        for module, level in module_levels.items():
            module_logger = logging.getLogger(module)
            module_logger.setLevel(getattr(logging, level.upper(), numeric_level))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the specified name.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create necessary log directories
ensure_log_directory("logs/gameplay")
ensure_log_directory("logs/training")
ensure_log_directory("logs/enemy_rl")
