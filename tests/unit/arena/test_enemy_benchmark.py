"""Fixed-seed benchmark: reproducibility and metric definitions."""
import pytest

from ai_platform_trainer.ai.evaluation.enemy_benchmark import (
    RoundResult,
    evaluate_policy,
    format_table,
    play_round,
    summarize,
)
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.enemy_policies import ChaseEnemy, RandomEnemy

CONFIG = ArenaConfig(width=1280, height=720)


def test_rounds_are_reproducible_per_seed():
    for policy in (ChaseEnemy(), RandomEnemy()):
        assert play_round(policy, CONFIG, "kite", seed=42) == play_round(
            policy, CONFIG, "kite", seed=42
        )


def test_summary_metrics():
    rounds = [
        RoundResult("idle", "caught", 120, 500.0, 40.0, 1, 0),
        RoundResult("idle", "hit", 60, 300.0, 250.0, 2, 1),
        RoundResult("kite", "timeout", 900, 400.0, 400.0, 5, 5),
    ]
    s = summarize(rounds)  # 1080 frames = 18 s
    assert s["catch_rate"] == pytest.approx(1 / 3)
    assert s["hit_rate"] == pytest.approx(1 / 3)
    assert s["timeout_rate"] == pytest.approx(1 / 3)
    assert s["mean_return"] == 0.0
    assert s["net_per_min"] == 0.0
    # (1 - 120/900) + (-1 - 60/900) + (-900/900), averaged over 3 rounds
    assert s["objective"] == pytest.approx(-0.4)
    assert s["catches_per_min"] == pytest.approx(60 / 18)
    assert s["seconds_per_hit"] == pytest.approx(18.0)
    assert s["time_to_catch_s"] == pytest.approx(2.0)
    assert s["closing_px_per_s"] == pytest.approx(510.0 / 18.0)
    assert s["missile_evasion_rate"] == pytest.approx(6 / 7)


def test_evaluate_policy_breaks_results_down_by_bot():
    summary = evaluate_policy(ChaseEnemy(), CONFIG, rounds=16, seed=0)
    assert summary["rounds"] == 16
    assert set(summary["by_bot"]) == {
        "idle",
        "wander",
        "kite",
        "strafe",
        "juke",
        "reverser",
        "charger",
        "orbit",
    }
    assert "Direct chase" in format_table({"policies": {"chase": summary}})
