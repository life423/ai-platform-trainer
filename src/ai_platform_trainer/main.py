"""
Main entry point for AI Platform Trainer.

This module initializes and runs the game, handling environment setup,
command-line arguments, and proper initialization of components.
"""
import argparse
import logging
import sys

from ai_platform_trainer.utils.environment import setup_gpu_environment, print_environment_info


def setup_logging(log_level: str = "INFO") -> None:
    """
    Set up logging configuration.

    Args:
        log_level: The logging level to use
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler("logs/gameplay/game.log", mode="w"),
        ],
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="AI Platform Trainer")
    parser.add_argument(
        "--mode", 
        choices=["play", "train", "benchmark"],
        default="play",
        help="Game mode (play, train, or benchmark)"
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run in headless mode (no visualization)"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    return parser.parse_args()


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print environment information
    print_environment_info()
    
    # Set up GPU environment
    device, cuda_working = setup_gpu_environment()
    logger.info(f"Using device: {device}")
    
    # Initialize game components
    try:
        # For now, just print mode information
        # In the future, this will initialize and run the actual game
        if args.mode == "play":
            logger.info("Starting game in play mode")
            # game = initialize_play_mode(device, headless=args.headless)
            # return game.run()
        elif args.mode == "train":
            logger.info("Starting game in training mode")
            # trainer = initialize_training_mode(device, headless=args.headless)
            # return trainer.run()
        elif args.mode == "benchmark":
            logger.info("Starting benchmark mode")
            # benchmark = initialize_benchmark_mode(device)
            # return benchmark.run()
        
        logger.info("Game mode placeholder executed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error running game: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
