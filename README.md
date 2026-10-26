# AI Platform Trainer

**Pixel Pursuit** - a 2D Pygame game. You fire homing missiles at an enemy that
chases you. The enemy and the missiles are reinforcement-learning agents: each
is trained headlessly, then runs in the game.

[![Release](https://img.shields.io/github/v/release/life423/ai-platform-trainer)](https://github.com/life423/ai-platform-trainer/releases)

## Getting Started

```bash
pip install -r requirements.txt
python run_game.py
```

(equivalently: `pip install -e .` then `ai-trainer`, or `python -m ai_platform_trainer`)

Choose **Play** from the main menu, then pick the missile AI (SAC or PPO). The
opponent is the AI Agent. Play refuses to start if a model it needs is missing,
and the menu names the command that trains it.

### Controls

| Key | Action |
|---|---|
| Arrow keys / WASD | Move |
| Space | Fire a missile |
| M | Back to the menu |
| F | Toggle fullscreen |
| Esc | Quit |

## Directory Structure

```
ai_platform_trainer/
├── ai/
│   ├── envs/enemy_env.py               # Gymnasium env the enemy PPO trains in
│   ├── evaluation/enemy_benchmark.py   # seeded benchmark of every enemy
│   ├── training/                       # train-enemy, SAC and PPO missile trainers
│   ├── missile_ai_loader.py            # loads the SAC and PPO missile models for the game
│   └── model_paths.py                  # where models live, the missing-model error
├── arena/                              # headless simulation for training and benchmarks
├── core/                               # config manager, screen context, logging
├── entities/                           # player, enemy and missile classes
├── gameplay/                           # the pygame game: menu, renderer, Play mode
├── cli.py                              # play, train-enemy, evaluate-enemy, benchmark
└── __main__.py                         # python -m ai_platform_trainer

assets/      # sprites
logs/        # training logs
models/      # trained models (see AI Models below)
reports/     # benchmark output (reports/benchmarks)
tests/       # pytest suite, mirrors the package layout
```

## Development

```bash
# Run tests (headless, no real display needed)
SDL_VIDEODRIVER=dummy python -m pytest

# Lint/format/type-check
pre-commit run --all-files
```

## AI Models

The game runs three trained models. None of them has a scripted fallback: if a
model is missing or fails to load, Play says which command trains it and refuses
to start.

| Model | File | Trained by |
|---|---|---|
| AI Agent (enemy, PPO) | `models/enemy_ppo/enemy_ppo.zip` + JSON model card | `python -m ai_platform_trainer train-enemy` |
| Missile AI: SAC | `models/missile_sac_model_final.zip` | `python -m ai_platform_trainer.ai.training.train_missile_sac` |
| Missile AI: PPO | `models/missile_rl_model_final.zip` | `python -m ai_platform_trainer.ai.training.train_missile_rl` |

The enemy trains in `arena/`, a simulation with no window, against the same
missile guidance the game uses. `train-enemy` saves checkpoints as it goes;
`--resume` continues from the last one.

The model card next to the enemy records how it was trained. The game only loads
an enemy whose card says it was trained on the game's 800×600 world.

To compare every enemy on the same seeded rounds:

```bash
python -m ai_platform_trainer benchmark
```

Results are written to `reports/benchmarks/`.
