"""
Entry point for running ai_platform_trainer as a module.

    python -m ai_platform_trainer            play the game
    python -m ai_platform_trainer --help     training, evaluation and benchmark commands
"""
from ai_platform_trainer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
