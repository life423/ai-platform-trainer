"""
Main entry point for AI Platform Trainer.

This module initializes and runs the game using the unified launcher.
The unified launcher provides support for multiple game initialization methods
and handles fallbacks if a specific method fails.
"""
from ai_platform_trainer.engine.core.unified_launcher import main

if __name__ == "__main__":
    main()
