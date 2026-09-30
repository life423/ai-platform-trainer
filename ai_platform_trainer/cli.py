"""Command line interface.

    python -m ai_platform_trainer                  play the game (default)
    python -m ai_platform_trainer train-enemy      train the PPO enemy (checkpointed)
    python -m ai_platform_trainer evaluate-enemy   fixed-seed evaluation of one enemy
    python -m ai_platform_trainer benchmark        every enemy on the same seeded rounds

Training never runs inside the game; each job is an explicit, resumable command.
"""
import argparse
import logging
import os
import sys
from typing import List, Optional

from ai_platform_trainer.ai.model_paths import (
    BENCHMARK_DIR,
    ENEMY_PPO_DIR,
    ENEMY_PPO_LOG_DIR,
    ENEMY_PPO_MODEL,
)
from ai_platform_trainer.arena.config import DEFAULT_SCREEN


def _add_arena_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rounds", type=int, default=1000, help="rounds per enemy")
    parser.add_argument("--seed", type=int, default=0, help="first round seed")
    parser.add_argument("--width", type=int, default=DEFAULT_SCREEN[0])
    parser.add_argument("--height", type=int, default=DEFAULT_SCREEN[1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai_platform_trainer", description="Pixel Pursuit AI platform"
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("play", help="launch the game (default)")

    train = sub.add_parser(
        "train-enemy", help="train the PPO enemy on the headless arena"
    )
    train.add_argument("--timesteps", type=int, default=10_000_000)
    train.add_argument("--envs", type=int, default=None, help="parallel environments")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--resume", action="store_true", help="continue from the last checkpoint"
    )
    train.add_argument(
        "--out",
        default=ENEMY_PPO_DIR,
        help="model directory; the game loads models/enemy_ppo",
    )
    train.add_argument(
        "--logs", default=ENEMY_PPO_LOG_DIR, help="training log directory"
    )

    evaluate = sub.add_parser(
        "evaluate-enemy", help="fixed-seed evaluation of one enemy"
    )
    evaluate.add_argument("--model", default=ENEMY_PPO_MODEL, help="SB3 model zip")
    _add_arena_args(evaluate)

    bench = sub.add_parser("benchmark", help="every enemy on the same seeded rounds")
    bench.add_argument(
        "--policies", default="all", help="comma separated names, or all (default)"
    )
    bench.add_argument(
        "--model", default=ENEMY_PPO_MODEL, help="PPO model zip, if present"
    )
    bench.add_argument("--out", default=BENCHMARK_DIR, help="report directory")
    _add_arena_args(bench)
    return parser


def _benchmark(args: argparse.Namespace) -> int:
    from ai_platform_trainer.ai.evaluation.enemy_benchmark import (
        format_table,
        run_benchmark,
        save_report,
    )
    from ai_platform_trainer.arena.config import ArenaConfig
    from ai_platform_trainer.arena.enemy_policies import SB3Enemy, baseline_policies

    policies = baseline_policies()
    if os.path.exists(args.model):
        policies.append(SB3Enemy.load(args.model))
    if args.policies != "all":
        wanted = [name.strip() for name in args.policies.split(",")]
        unknown = set(wanted) - {p.name for p in policies}
        if unknown:
            print(
                f"Unknown or unavailable policies: {sorted(unknown)}", file=sys.stderr
            )
            return 2
        policies = [p for p in policies if p.name in wanted]
    config = ArenaConfig(width=args.width, height=args.height)
    report = run_benchmark(policies, config, args.rounds, seed=args.seed)
    print(format_table(report))
    for path in save_report(report, args.out):
        print("wrote", path)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from ai_platform_trainer.ai.evaluation.enemy_benchmark import (
        evaluate_policy,
        format_table,
    )
    from ai_platform_trainer.arena.config import ArenaConfig
    from ai_platform_trainer.arena.enemy_policies import SB3Enemy

    if not os.path.exists(args.model):
        print(f"No model at {args.model} (run train-enemy first)", file=sys.stderr)
        return 2
    config = ArenaConfig(width=args.width, height=args.height)
    summary = evaluate_policy(
        SB3Enemy.load(args.model), config, args.rounds, seed=args.seed
    )
    print(format_table({"policies": {"ppo": summary}}))
    for style, rates in summary["by_bot"].items():
        catch, hit = rates["catch_rate"], rates["hit_rate"]
        print(f"  vs {style:9s} catch {catch:6.1%}   hit {hit:6.1%}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "play"):
        from ai_platform_trainer.main import main as play

        play()
        return 0
    logging.basicConfig(level=logging.WARNING)
    if args.command == "train-enemy":
        from ai_platform_trainer.ai.training.train_enemy_ppo import train

        train(
            timesteps=args.timesteps,
            n_envs=args.envs,
            seed=args.seed,
            resume=args.resume,
            out_dir=args.out,
            log_dir=args.logs,
        )
        return 0
    if args.command == "evaluate-enemy":
        return _evaluate(args)
    if args.command == "benchmark":
        return _benchmark(args)
    return 2
