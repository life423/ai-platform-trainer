"""Training must treat a clock timeout as a cut-off, not a death, and must not
let a fresh run inherit an older run's best policy."""
import json
import math
import os

import pytest

from ai_platform_trainer.ai.model_paths import MISSILE_SAC_MODEL
from ai_platform_trainer.ai.training.train_enemy_ppo import (
    SELECTION_METRIC,
    ArenaEvalCallback,
    model_card_path,
)

needs_sac = pytest.mark.skipif(
    not os.path.exists(MISSILE_SAC_MODEL), reason="SAC missile model not on disk"
)


@needs_sac
def test_a_round_cut_off_by_the_clock_is_marked_for_bootstrapping():
    from ai_platform_trainer.ai.envs.enemy_env import EnemyArenaEnv

    env = EnemyArenaEnv(max_frames=40, preload_prob=0.0)
    env.reset(seed=0)
    for _ in range(100):
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        if truncated:
            assert info["TimeLimit.truncated"] is True
            assert terminated is False
            return
        assert info["TimeLimit.truncated"] is False
    pytest.fail("the round never reached the clock")


def test_a_fresh_run_does_not_inherit_an_old_best(tmp_path):
    out = tmp_path / "out"
    logs = tmp_path / "logs"
    out.mkdir()
    logs.mkdir()
    best = out / "best_model.zip"
    best.write_bytes(b"old policy")
    with open(model_card_path(str(best)), "w", encoding="utf-8") as fh:
        json.dump({"evaluation": {SELECTION_METRIC: 9.0}}, fh)

    fresh = ArenaEvalCallback(1000, 4, str(out), str(logs), seed=0, resume=False)
    resumed = ArenaEvalCallback(1000, 4, str(out), str(logs), seed=0, resume=True)

    assert fresh.best_score == -math.inf
    assert resumed.best_score == pytest.approx(9.0)
