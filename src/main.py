from gameplay.game import Game
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


if __name__ == "__main__":
    game = Game()
    game.run()
    
