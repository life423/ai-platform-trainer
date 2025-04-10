# 2D Platformer with AI Integration

This project is a 2D platformer game created using Python and Pygame, with an integrated AI model that controls an enemy. The AI is trained to track and follow the player, providing a dynamic gameplay experience.

## Project Structure

---

The project is organized into several directories and Python scripts:

```
├── ai_model
│   ├── model.py           # Defines the AI model for controlling the enemy
│   ├── train.py           # Script to train the AI model
│   └── saved_models       # Directory to store trained models
│       └── enemy_ai_model.pth
├── core
│   ├── config.py          # Configuration file for game settings
│   ├── utils.py           # Utility functions used in the game
├── data
│   └── collision_data.json # Stores collision data for AI training
├── entities
│   ├── enemy.py           # Enemy class, which integrates AI for decision making
│   ├── entity.py          # Base class for game entities
│   └── player.py          # Player class (if needed)
├── gameplay
│   ├── game_manager.py    # Main game loop and logic
│   ├── main.py            # Entry point for running the game
└── requirements.txt       # Dependencies for the project
```

## Features

---

- **AI Integration**: The game includes an AI-controlled enemy that attempts to track and reach the player. The AI model is built and trained using PyTorch.
- **Dynamic Player Movement**: The player can move around the game area with random or manual movement based on the game mode.
- **Collision Logging**: Collision data between the player and the enemy is logged to help train the AI model.
- **Flexible Modes**: The game can be run in either "training" or "play" mode.

## Installation

---

### Prerequisites

---
- Python 3.9 or above
- Virtual Environment (optional but recommended)

### Setup Instructions

---

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd 2d-platformer-pygame-ai-scratch
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Pygame and Torch (if not in requirements.txt)**:
   ```bash
   pip install pygame torch
   ```

## Running the Game

---

1. **Activate the Virtual Environment** (if not already active):
   ```bash
   source venv/bin/activate
   ```

2. **Run the Game**:
   ```bash
   python3 gameplay/main.py
   ```

## How to Play

---
- **Movement**: Use the arrow keys (`UP`, `DOWN`, `LEFT`, `RIGHT`) or `WASD` keys to move the player around the screen.
- **Quit**: Press `ESC` or `Q` to quit the game.

## Game Modes

---
- **Training Mode**: The player moves randomly, and collision data is logged for AI training purposes.
- **Play Mode**: The player is controlled manually, and the AI controls the enemy.

To switch between modes, modify the `mode` variable in `gameplay/main.py`:
```python
mode = 'play'  # or 'training'
```

## AI Model

---
The enemy AI model is defined in `ai_model/model.py` and can be trained using `ai_model/train.py`. The model uses PyTorch to learn from logged collision data and predict enemy movements.

### Training the Model

---
To train the AI model:
1. Ensure collision data is logged in `data/collision_data.json` by running the game in training mode.
2. Run the training script:
   ```bash
   python3 ai_model/train.py
   ```
3. The trained model will be saved in the `ai_model/saved_models` directory.

## Customization

---
- **Configuration**: Game settings, such as screen size, player speed, and colors, are managed in `core/config.py`.
- **AI Behavior**: The behavior of the AI enemy can be adjusted in `entities/enemy.py` by modifying the logic and model integration.

## Troubleshooting

---
- **Missing Pygame or Torch**: Ensure that `pygame` and `torch` are installed using `pip install pygame torch`.
- **Virtual Environment Not Activated**: Always activate your virtual environment before running or installing dependencies.

## License

---
This project is licensed under the MIT License.

## Contributions

---
Contributions are welcome! Feel free to submit a pull request or open an issue to report bugs or suggest improvements.

## Acknowledgments

---
- **Pygame**: For the 2D graphics and game engine.
- **PyTorch**: For building and training the AI model.

## Contact

---
If you have any questions or feedback, please contact us at: aiandrew631@gmail.com
