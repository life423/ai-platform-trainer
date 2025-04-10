# GPU-Accelerated Training Environment

This module implements a high-performance, CUDA-accelerated training environment for reinforcement learning, specifically designed to improve the missile avoidance behavior of enemy entities.

## Overview

Traditional game-based RL training can be slow due to Python's overhead and rendering requirements. This implementation moves the core game logic to C++/CUDA, allowing:

- **10-50x faster training** compared to the Python implementation
- **Parallel simulation** of multiple environments
- **Efficient GPU utilization** for physics calculations
- **Enhanced missile avoidance** capabilities through improved training

## Prerequisites

To build and use this extension, you need:

- NVIDIA CUDA Toolkit (11.0 or later)
- CMake (3.18 or later)
- PyBind11
- Python 3.7+
- PyTorch 1.7+
- Stable-Baselines3
- Gym

## Building the Extension

1. Make sure CUDA and other dependencies are installed
2. Build the extension in place:

```bash
cd ai_platform_trainer/cpp
python setup.py build_ext --inplace
```

## Component Overview

The system consists of:

- **C++ Entity Classes**: Lightweight versions of game entities
- **CUDA Physics Kernels**: Parallel physics calculations
  - Position updates
  - Collision detection
  - Missile trajectory prediction
  - Danger map calculation
  - Optimal evasion vector computation
- **Python Bindings**: PyBind11-based interface to C++
- **Environment Wrappers**: Gym-compatible RL environments
- **Training Scripts**: Ready-to-use training pipelines

## Using the Environment

### Basic Usage

```python
import torch
from stable_baselines3 import PPO
from ai_platform_trainer.cpp.gpu_environment import make_env

# Create environment
env = make_env()

# Create and train a model
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=1000000)

# Save trained model
model.save("missile_avoidance_model")
```

### Training with Custom Settings

```python
from ai_platform_trainer.cpp.gpu_environment import native_env, make_env

# Create configuration with enhanced missile avoidance settings
config = native_env.EnvironmentConfig()
config.screen_width = 800
config.screen_height = 600
config.missile_lifespan = 10000.0  # 10 seconds
config.missile_detection_radius = 250.0  # Increased detection radius
config.missile_danger_radius = 150.0    # Increased danger radius
config.evasion_strength = 2.5           # Higher evasion strength

# Create environment with custom config
env = make_env(config)
```

### Using the Training Script

The included training script offers a command-line interface for training:

```bash
python -m ai_platform_trainer.cpp.train_missile_avoidance --num-envs 4 --timesteps 1000000
```

Options:
- `--num-envs`: Number of parallel environments (default: 4)
- `--timesteps`: Total timesteps to train for (default: 1000000)
- `--save-dir`: Directory to save model checkpoints (default: models/gpu_rl)
- `--log-dir`: Directory to save tensorboard logs (default: logs/gpu_rl)
- `--save-freq`: Frequency to save checkpoints (default: 10000)

For evaluation:
```bash
python -m ai_platform_trainer.cpp.train_missile_avoidance --eval --model-path models/gpu_rl/final_model
```

## Tuning Missile Avoidance

The environment includes several parameters to tune missile avoidance behavior:

- `missile_detection_radius`: How far the enemy can detect missiles (250.0)
- `missile_danger_radius`: When to start emergency evasion (150.0)
- `evasion_strength`: How strongly to evade (2.5)
- `missile_prediction_steps`: How many frames to predict missile movement (30)

## Architecture Details

This implementation follows a data-oriented design pattern optimized for GPU computation:

1. **Entity Data**: Stored in struct-of-arrays format for coalesced memory access
2. **CUDA Kernels**: 
   - Each kernel performs a specific physics calculation
   - Optimized for parallel execution on GPU
   - Uses shared memory and efficient memory patterns
3. **Python Integration**:
   - Zero-copy transfers where possible
   - Efficient numpy/tensor conversions
   - Vectorized environments for training
