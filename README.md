# AI Platform Trainer

[![Build and Deploy Executables](https://github.com/life423/ai-platform-trainer/actions/workflows/deploy.yml/badge.svg)](https://github.com/life423/ai-platform-trainer/actions/workflows/deploy.yml)
[![Release](https://img.shields.io/github/v/release/life423/ai-platform-trainer)](https://github.com/life423/ai-platform-trainer/releases)
[![Downloads](https://img.shields.io/github/downloads/life423/ai-platform-trainer/total)](https://github.com/life423/ai-platform-trainer/releases)

## Project Structure Cleanup

This project has been restructured to follow a cleaner architecture. The duplicate code that was previously in both `ai_platform_trainer/` and `src/` directories has been unified.

## Directory Structure

```
ai-platform-trainer/  (root of repository)
├── ai_platform_trainer/        # Python package for the game
│   ├── __init__.py
│   ├── entities/               # Game entities (Player, Enemy, etc.)
│   ├── gameplay/               # Game mechanics, rules, and modes
│   ├── utils/                  # Utility modules (config loading, helpers, etc.)
│   ├── ai_model/               # AI agent, environment, policy network (RL code)
│   ├── cpp/                    # C++/CUDA extension (physics engine)
│   │   ├── CMakeLists.txt      # Single CMake for building the extension
│   │   ├── src/…, include/…    # C++ source and headers
│   │   └── pybind/…            # Pybind11 binding code
│   └── (other modules as needed)
├── assets/                     # Game assets (sprites, sounds) if any
├── tests/                      # Test cases for the codebase
├── README.md                   # Updated documentation (including CUDA notes)
├── setup.py / pyproject.toml   # Build configuration for the Python package
├── requirements.txt            # Python dependencies
└── config.json                 # Consolidated configuration file
```

## Important Notes

1. The `src/` directory has been removed to avoid code duplication.
2. All code is now maintained in the `ai_platform_trainer/` package.
3. Configuration has been consolidated into a single `config.json` file.

## CUDA Support

The project includes CUDA extensions for physics calculations. To build and use these extensions:

1. Ensure you have CUDA toolkit installed
2. Run `python build_cuda_extensions.py` to build the extensions
3. Verify CUDA availability with `python verify_cuda_usage.py`

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the game: `python -m ai_platform_trainer`