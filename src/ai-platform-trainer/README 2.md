# AI Platform Trainer

An enterprise-grade platform for training and evaluating AI agents using reinforcement learning in a game environment.

## Overview

AI Platform Trainer is a 2D platformer game environment designed for training and evaluating AI agents using deep reinforcement learning. The platform includes a neural network-based enemy AI, reinforcement learning training capabilities, real-time visualizations, and a high-performance C++/CUDA backend for accelerated training.

## Features

- **Game Environment**: A 2D game environment built with PyGame where entities can interact
- **Neural Network Models**: Pre-trained models for missile trajectory prediction and enemy movement
- **Reinforcement Learning**: GPU-accelerated reinforcement learning using PPO for training enemy behaviors
- **C++/CUDA Integration**: High-performance physics simulation with Python bindings
- **Visualizations**: Real-time training visualizations and performance metrics
- **Cross-platform**: Support for both CPU and GPU environments with automatic detection

## Directory Structure

```
ai_platform_trainer/
├── src/                       # All source code
│   └── ai_platform_trainer/   # Main package
│       ├── core/              # Core engine components
│       ├── ml/                # Machine learning components
│       │   ├── models/        # Neural network definitions
│       │   ├── training/      # Training pipelines
│       │   ├── rl/            # Reinforcement learning
│       │   └── inference/     # Model inference
│       ├── physics/           # Physics engine (with C++ bindings)
│       ├── entities/          # Game entities
│       ├── rendering/         # Visualization and rendering
│       └── utils/             # Utility functions
├── tests/                     # Test suite
├── docs/                      # Documentation
├── scripts/                   # Utility scripts
├── assets/                    # Game assets
└── deployment/                # Deployment configurations
```

## Requirements

- Python 3.8 or newer
- PyTorch 1.8.0 or newer
- CUDA Toolkit 10.2+ (optional, for GPU acceleration)
- CMake 3.10+ (for building C++ extensions)

## Installation

### CPU Environment

```bash
# Create a conda environment
conda env create -f environment-cpu.yml
conda activate ai-platform-cpu

# Install the package
pip install -e .
```

### GPU Environment (recommended for training)

```bash
# Create a conda environment with CUDA support
conda env create -f environment-gpu.yml
conda activate ai-platform-gpu

# Install the package
pip install -e .

# Build the C++ extensions
cd src/ai_platform_trainer/physics/cpp
python setup.py build_ext --inplace
```

## Usage

### Running the Game

```bash
# Run the game in play mode
python -m src.ai_platform_trainer.main
```

### Training Models

#### Training the Missile Model

```bash
# Train the missile trajectory prediction model
python -m src.ai_platform_trainer.ml.training.train_missile_model --epochs 100 --batch-size 32
```

#### Training the Enemy RL Agent

```bash
# Train the enemy agent using reinforcement learning
python -m src.ai_platform_trainer.ml.rl.train_enemy_rl --timesteps 1000000 --save-freq 10000
```

## Documentation

For more detailed information, refer to:

- [Project Structure](docs/design/project_structure.md): Details on code organization
- [Refactoring Next Steps](docs/refactoring_next_steps.md): Future improvements

## Development

To set up a development environment:

1. Install development dependencies
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks
```bash
pre-commit install
```

3. Run tests
```bash
python -m pytest
```

## License

[Insert license information]
