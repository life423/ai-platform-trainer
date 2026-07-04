# Test Suite for AI Platform Trainer

Tests for the AI Platform Trainer project, organized to mirror the
`ai_platform_trainer/` package layout.

## Directory Structure

```
tests/
├── unit/
│   ├── ai/                # ai_platform_trainer/ai/ - dataset and model unit tests
│   ├── core/               # ai_platform_trainer/core/ - config manager, etc.
│   ├── entities/           # ai_platform_trainer/entities/ - missile, player
│   │   └── behaviors/      # ai_platform_trainer/entities/behaviors/
│   └── gameplay/           # ai_platform_trainer/gameplay/ - collisions, etc.
└── integration/
    ├── test_game_mechanics.py     # player/enemy/missile interactions
    └── test_training_pipeline.py  # data collection -> retrain pipeline
```

## Running the Tests

```bash
# Run everything (headless - no real display needed)
SDL_VIDEODRIVER=dummy python -m pytest

# Run one directory
python -m pytest tests/unit/entities/

# With coverage
python -m pytest --cov=ai_platform_trainer tests/
```

## Adding New Tests

- Place new tests in the subdirectory matching the source module under test.
- Follow the `test_*.py` file / `test_*` function naming convention.
- Mock pygame/display dependencies where possible; the suite is designed to
  run fully headless in CI (`SDL_VIDEODRIVER=dummy`).
