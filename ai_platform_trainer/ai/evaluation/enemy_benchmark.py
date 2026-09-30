"""Fixed-seed benchmark of enemy policies on the canonical arena.

Every policy plays the same list of rounds (same seeds, spawn points and bot
personalities), so differences come from the policies rather than from luck.

    python -m ai_platform_trainer benchmark --rounds 2000
"""
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from ai_platform_trainer.ai.envs.enemy_env import round_return
from ai_platform_trainer.arena.config import FPS, ArenaConfig
from ai_platform_trainer.arena.enemy_policies import POLICY_NAMES, EnemyPolicy
from ai_platform_trainer.arena.episode import DEFAULT_MAX_FRAMES, ArenaEpisode
from ai_platform_trainer.arena.observations import enemy_player_distance
from ai_platform_trainer.arena.player_bots import BOT_STYLES, make_bot


@dataclass(frozen=True)
class RoundResult:
    bot_style: str
    outcome: str  # caught, hit or timeout
    frames: int
    start_distance: float
    end_distance: float
    missiles_fired: int
    missiles_expired: int


def play_round(
    policy: EnemyPolicy,
    config: ArenaConfig,
    bot_style: str,
    seed: int,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> RoundResult:
    """Play one round with independent, seeded RNG streams for bot, sim and policy."""
    rng = random.Random(seed)
    bot = make_bot(bot_style, random.Random(rng.getrandbits(32)))
    episode = ArenaEpisode(config, bot, random.Random(rng.getrandbits(32)), max_frames)
    state = episode.reset()
    policy.reset(state, random.Random(rng.getrandbits(32)))
    start = enemy_player_distance(state)
    while True:
        dx, dy = policy.act(state)
        result = episode.step(dx, dy, capped=policy.capped)
        if result.terminated or result.truncated:
            break
    return RoundResult(
        bot_style=bot_style,
        outcome=result.event or "timeout",
        frames=state.frame + (1 if result.terminated else 0),
        start_distance=start,
        end_distance=enemy_player_distance(state),
        missiles_fired=episode.missiles_fired,
        missiles_expired=episode.missiles_expired,
    )


def _ci95(p: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n) if n else 0.0


def summarize(rounds: Sequence[RoundResult]) -> Dict[str, Any]:
    """Aggregate round results into rates, per-minute figures and 95% intervals."""
    n = len(rounds)
    caught = sum(r.outcome == "caught" for r in rounds)
    hit = sum(r.outcome == "hit" for r in rounds)
    seconds = sum(r.frames for r in rounds) / FPS
    catch_frames = [r.frames for r in rounds if r.outcome == "caught"]
    expired = sum(r.missiles_expired for r in rounds)
    closed = sum(r.start_distance - r.end_distance for r in rounds)
    return {
        "rounds": n,
        "catch_rate": caught / n,
        "catch_rate_ci95": _ci95(caught / n, n),
        "hit_rate": hit / n,
        "hit_rate_ci95": _ci95(hit / n, n),
        "timeout_rate": (n - caught - hit) / n,
        "mean_return": (caught - hit) / n,
        # The training objective: catches and hits with the per-frame time cost.
        "objective": sum(round_return(r.outcome, r.frames) for r in rounds) / n,
        "catches_per_min": caught / seconds * 60.0,
        "hits_per_min": hit / seconds * 60.0,
        # The game score: catches minus missile hits per minute of play.
        "net_per_min": (caught - hit) / seconds * 60.0,
        "seconds_per_hit": seconds / hit if hit else None,
        "time_to_catch_s": (sum(catch_frames) / len(catch_frames) / FPS)
        if catch_frames
        else None,
        "closing_px_per_s": closed / seconds,
        "missile_evasion_rate": expired / (expired + hit) if expired + hit else None,
        "mean_round_s": seconds / n,
    }


def evaluate_policy(
    policy: EnemyPolicy,
    config: ArenaConfig,
    rounds: int,
    seed: int = 0,
    styles: Sequence[str] = BOT_STYLES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Dict[str, Any]:
    started = time.perf_counter()
    results = [
        play_round(policy, config, styles[i % len(styles)], seed + i, max_frames)
        for i in range(rounds)
    ]
    summary = summarize(results)
    summary["by_bot"] = {}
    for style in styles:
        subset = [r for r in results if r.bot_style == style]
        if subset:
            part = summarize(subset)
            summary["by_bot"][style] = {k: part[k] for k in ("catch_rate", "hit_rate")}
    summary["wall_seconds"] = time.perf_counter() - started
    return summary


def run_benchmark(
    policies: Sequence[EnemyPolicy],
    config: ArenaConfig,
    rounds: int,
    seed: int = 0,
    styles: Sequence[str] = BOT_STYLES,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arena": asdict(config),
        "rounds_per_policy": rounds,
        "seed": seed,
        "max_round_s": max_frames / FPS,
        "bot_styles": list(styles),
        "policies": {},
    }
    for policy in policies:
        report["policies"][policy.name] = evaluate_policy(
            policy, config, rounds, seed, styles, max_frames
        )
    return report


def _pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100.0 * value:.1f}%"


def _num(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


TABLE_COLUMNS = (
    "Enemy",
    "Objective",
    "Net/min",
    "Catch rate",
    "Missile-hit rate",
    "Timeout",
    "Catches/min",
    "Hits/min",
    "Time to catch (s)",
    "Closing (px/s)",
    "Missiles evaded",
)


def format_table(report: Dict[str, Any]) -> str:
    """Markdown table, one row per policy."""
    lines = ["| " + " | ".join(TABLE_COLUMNS) + " |", "|" + "---|" * len(TABLE_COLUMNS)]
    for name, m in report["policies"].items():
        cells = [
            POLICY_NAMES.get(name, name),
            _num(m["objective"], 3),
            _num(m["net_per_min"], 2),
            _pct(m["catch_rate"]),
            _pct(m["hit_rate"]),
            _pct(m["timeout_rate"]),
            _num(m["catches_per_min"], 2),
            _num(m["hits_per_min"], 2),
            _num(m["time_to_catch_s"], 2),
            _num(m["closing_px_per_s"], 0),
            _pct(m["missile_evasion_rate"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return chr(10).join(lines)


def save_report(report: Dict[str, Any], out_dir: str, stem: str = "enemy") -> List[str]:
    """Write <stem>_<timestamp>.json/.md plus <stem>_latest copies."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    table = format_table(report)
    paths = []
    for name in (f"{stem}_{stamp}", f"{stem}_latest"):
        json_path = os.path.join(out_dir, name + ".json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        md_path = os.path.join(out_dir, name + ".md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(table + "\n")
        paths += [json_path, md_path]
    return paths
