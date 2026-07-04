#!/usr/bin/env python
"""
Main launcher for AI Platform Trainer.

Thin shim so `python run_game.py`, the packaged `ai-trainer` console
script (see setup.py), and `python -m ai_platform_trainer` all run the
exact same startup logic, defined once in ai_platform_trainer/main.py.
"""
import sys

from ai_platform_trainer.main import main

if __name__ == "__main__":
    sys.exit(main())
