# AI Platform Trainer

**Pixel Pursuit** - a 2D Pygame game where a player fights an AI-controlled
enemy with homing missiles, built as a testbed for comparing supervised
learning and reinforcement learning approaches to game AI side by side.

[![Release](https://img.shields.io/github/v/release/life423/ai-platform-trainer)](https://github.com/life423/ai-platform-trainer/releases)

## Getting Started

```bash
pip install -r requirements.txt
python run_game.py
```

(equivalently: `pip install -e .` then `ai-trainer`, or `python -m ai_platform_trainer`)

From the main menu:
- **Play** - the opponent is the AI Agent (a PPO policy trained by `train-enemy`). Pick the missile AI (SAC, PPO, or Supervised NN) and fight. Play refuses to start if a model it needs is missing, and the menu names the command that trains it.
- **Train** - runs scripted player/enemy patterns and logs gameplay data,
  used to retrain the Supervised NN missile model.

## Directory Structure

```
ai_platform_trainer/
├── ai/
│   ├── inference/         # runtime missile guidance controller
│   ├── models/             # neural net / RL model definitions
│   ├── training/           # standalone training scripts (SAC, PPO, supervised)
│   └── missile_ai_loader.py # loads/selects the 3 missile guidance models
├── core/                   # config manager, data logger, screen context
├── entities/                # Player, Enemy, Missile classes
│   └── behaviors/            # enemy AI controller logic
├── gameplay/
│   ├── modes/                 # Play / Train mode managers
│   ├── game_core.py            # main game loop
│   └── menu.py                  # main menu + submenus
└── utils/                   # data validation/retraining pipeline, helpers

assets/    # sprites
data/      # logged training data (raw/ + timestamped backups/)
models/    # trained model weights (.pth / stable-baselines3 .zip checkpoints)
logs/      # training run logs (evaluation curves, etc.)
tests/     # pytest suite, mirrors the ai_platform_trainer/ layout
```

## Development

```bash
# Run tests (headless, no real display needed)
SDL_VIDEODRIVER=dummy python -m pytest

# Lint/format/type-check
pre-commit run --all-files
```

## AI Models

Three interchangeable missile guidance models, selectable from the Play
submenu:

- **Supervised NN** - a small feedforward network trained by imitating a
  deterministic controller's logged decisions (see Train mode above).
- **SAC** (Soft Actor-Critic) and **PPO** (Proximal Policy Optimization) -
  reinforcement learning models trained from scratch in self-contained
  simulated environments (`ai/training/train_missile_sac.py` and
  `train_missile_rl.py`), with no dependency on logged gameplay data.

The opponent is the AI Agent: a PPO policy trained by `train-enemy` and saved to `models/enemy_ppo/enemy_ppo.zip`. Play has no scripted fallback.
