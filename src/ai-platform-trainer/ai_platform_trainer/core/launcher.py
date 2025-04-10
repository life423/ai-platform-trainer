# ai_platform_trainer/core/launcher.py
from ai_platform_trainer.core.logging_config import setup_logging
from ai_platform_trainer.gameplay.game import Game

# If you had a config loader:
# from ai_platform_trainer.core.config_loader import load_config


def main():
    setup_logging()
    # config = load_config()  # If you have a config loader
    game = Game()  # Pass config here if needed: Game(config)
    game.run()
