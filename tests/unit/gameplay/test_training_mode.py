"""
Tests for TrainingMode: what a collection session records, and that it is
never lost or written into the real data folder.

Every path is redirected into pytest's tmp_path, so these tests never touch
the project's own data/raw/training_data.json.
"""

import json
import os

import pytest

from ai_platform_trainer.ai.training.missile_dataset import SCHEMA_VERSION, row_problem
from ai_platform_trainer.core.screen_context import ScreenContext
from ai_platform_trainer.gameplay.config import config
from ai_platform_trainer.gameplay.modes.training_mode import TrainingMode

FRAME_MS = 16


class FakeGame:
    """The few game attributes TrainingMode reads. No model is loaded."""

    def __init__(self) -> None:
        self.screen_width = 800
        self.screen_height = 600
        self.missile_model = None
        self.is_respawning = False
        self.respawn_timer = 0
        self.respawn_delay = 1000
        self.player = None
        self.enemy = None
        self.data_logger = None

    def handle_respawn(self, current_time: int) -> None:
        pass


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """Point every data and model path into tmp_path."""
    data_dir = tmp_path / "data" / "raw"
    master = str(data_dir / "training_data.json")
    session = str(data_dir / "_training_session.json")
    monkeypatch.setattr(config, "DATA_PATH", master)
    monkeypatch.setattr(
        config, "MISSILE_MODEL_PATH", str(tmp_path / "models" / "m.pth")
    )
    monkeypatch.setattr(TrainingMode, "SESSION_DATA_PATH", session)
    ScreenContext.initialize(800, 600)
    return {"master": master, "session": session}


def _run_frames(mode: TrainingMode, frames: int) -> None:
    for i in range(frames):
        mode.update(i * FRAME_MS)


def test_session_records_valid_rows_without_a_model(redirected):
    """Labels come from the deterministic controller, so no model is needed."""
    game = FakeGame()
    mode = TrainingMode(game)

    _run_frames(mode, 600)

    rows = mode.data_logger.data
    assert rows, "the scripted player should have fired at least one missile"
    assert all(row_problem(row) is None for row in rows)
    assert all(row["schema"] == SCHEMA_VERSION for row in rows)


def test_session_labels_are_not_all_zero_without_a_model(redirected):
    game = FakeGame()
    mode = TrainingMode(game)

    _run_frames(mode, 600)

    actions = [row["missile_action"] for row in mode.data_logger.data]
    assert any(abs(a) > 1e-6 for a in actions)


def test_session_is_saved_as_it_goes(redirected):
    game = FakeGame()
    mode = TrainingMode(game)

    _run_frames(mode, 600)

    assert os.path.exists(redirected["session"])
    with open(redirected["session"], "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert len(saved) > 0


def test_finalize_merges_the_session_into_the_master(redirected):
    game = FakeGame()
    mode = TrainingMode(game)
    _run_frames(mode, 600)
    collected = len(mode.data_logger.data)

    assert mode.finalize(retrain=False)

    with open(redirected["master"], "r", encoding="utf-8") as f:
        assert len(json.load(f)) == collected
    assert not os.path.exists(redirected["session"])


def test_finalize_is_safe_to_call_twice(redirected):
    game = FakeGame()
    mode = TrainingMode(game)
    _run_frames(mode, 300)

    assert mode.finalize(retrain=False)
    assert not mode.finalize(retrain=False)

    with open(redirected["master"], "r", encoding="utf-8") as f:
        first = len(json.load(f))
    assert first > 0


def test_a_new_session_recovers_samples_left_by_a_crash(redirected):
    """Samples from a session that never finalized are merged on the next start."""
    os.makedirs(os.path.dirname(redirected["session"]), exist_ok=True)
    leftover = [
        {
            "schema": SCHEMA_VERSION,
            "features": [0.1] * 8 + [0.0],
            "missile_action": 2.0,
            "missile_collision": False,
            "timestamp": 500,
        }
    ]
    with open(redirected["session"], "w", encoding="utf-8") as f:
        json.dump(leftover, f)

    TrainingMode(FakeGame())

    with open(redirected["master"], "r", encoding="utf-8") as f:
        assert len(json.load(f)) == 1
    with open(redirected["session"], "r", encoding="utf-8") as f:
        assert json.load(f) == []
