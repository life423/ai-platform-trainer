import os


class Config:
    def __init__(self):
        # Screen settings
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.WINDOW_TITLE = "Pixel Pursuit"
        self.FRAME_RATE = 60
        self.SCREEN_SIZE = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        # Data paths
        self.DATA_PATH = os.path.join("data", "raw", "training_data.json")
        self.MODEL_PATH = os.path.join("models", "enemy_ai_model.pth")

        # Game settings
        self.WALL_MARGIN = 50
        self.MIN_DISTANCE = 100

        # Speed settings
        self.RANDOM_SPEED_FACTOR_MIN = 0.8
        self.RANDOM_SPEED_FACTOR_MAX = 1.2
        self.ENEMY_MIN_SPEED = 2


config = Config()
