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
        RoundResult("idle", 1, 0, 120, 500.0, 40.0, 1, 0),
        RoundResult("idle", 0, 1, 60, 300.0, 250.0, 2, 1),
        RoundResult("kite", 2, 3, 900, 400.0, 400.0, 5, 5),
    ]
    s = summarize(rounds)  # 1080 frames = 18 s; 3 catches, 4 hits
    assert s["catches_per_round"] == pytest.approx(1.0)
    assert s["hits_per_round"] == pytest.approx(4 / 3)
    assert s["catches_per_min"] == pytest.approx(10.0)
    assert s["hits_per_min"] == pytest.approx(40 / 3)
    assert s["net_per_min"] == pytest.approx(-10 / 3)
    # per round: (1 - 120/900), (-1 - 60/900), (2 - 3 - 900/900), averaged over 3
    assert s["objective"] == pytest.approx(-2.2 / 3)
    assert s["closing_px_per_s"] == pytest.approx(510.0 / 18.0)
    # 6 missiles ran out of fuel, 4 hit; 6 / (6 + 4) of the missiles were evaded
    assert s["missile_evasion_rate"] == pytest.approx(0.6)


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
