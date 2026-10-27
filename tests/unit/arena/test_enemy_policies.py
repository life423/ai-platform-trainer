"""Scripted enemies: the dodger runs from closing missiles and holds its move."""
import pytest

from ai_platform_trainer.ai.evaluation.enemy_benchmark import evaluate_policy
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.enemy_policies import ChaseEnemy, DodgeChaseEnemy
from ai_platform_trainer.arena.state import ArenaState, Body, MissileState

CONFIG = ArenaConfig(width=800, height=600)


def no_turn(missile, target_x, target_y, config):
    """Missiles fly straight, so these tests do not need the SAC model."""
    return None


def frame(missiles=()):
    """Enemy at (100, 300) with the player 300 px to its right, no wrap involved."""
    return ArenaState(
        config=CONFIG,
        player=Body(400.0, 300.0),
        enemy=Body(100.0, 300.0),
        missiles=list(missiles),
    )


def missile(vx, vy):
    """A missile 80 px right of the enemy and 20 px above it."""
    return MissileState(x=200.0, y=300.0, vx=vx, vy=vy, life_frames=150)


def test_dodge_runs_straight_away_from_a_closing_missile():
    move = DodgeChaseEnemy().act(frame([missile(vx=-5.0, vy=0.0)]))
    # Away from the missile is left and a little down, at full speed on the larger axis.
    assert move == pytest.approx((-CONFIG.enemy_speed, 0.25 * CONFIG.enemy_speed))


def test_dodge_ignores_a_missile_flying_away():
    move = DodgeChaseEnemy().act(frame([missile(vx=5.0, vy=0.0)]))
    assert move == pytest.approx((CONFIG.enemy_speed, 0.0))  # just chases the player


def test_dodge_holds_each_move_for_the_decision_clock():
    policy = DodgeChaseEnemy()
    threatened = frame([missile(vx=-5.0, vy=0.0)])
    clear = frame()
    first = policy.act(threatened)
    for _ in range(3):
        assert policy.act(clear) == first  # held for the rest of the 4 frames
    assert policy.act(clear) == pytest.approx((CONFIG.enemy_speed, 0.0))


def test_dodge_loses_fewer_hits_than_chase():
    chase = evaluate_policy(ChaseEnemy(), CONFIG, rounds=16, seed=0, guidance=no_turn)
    dodge = evaluate_policy(
        DodgeChaseEnemy(), CONFIG, rounds=16, seed=0, guidance=no_turn
    )
    assert dodge["hits_per_min"] < chase["hits_per_min"]
