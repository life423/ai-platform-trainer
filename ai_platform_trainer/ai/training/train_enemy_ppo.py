"""Train the base enemy agent with PPO on the canonical arena.

    python -m ai_platform_trainer train-enemy --timesteps 10000000
    python -m ai_platform_trainer train-enemy --resume

Outputs in models/enemy_ppo/:
    checkpoints/enemy_ppo_<steps>_steps.zip   periodic snapshots, used by --resume
    best_model.zip (+ .json)                  best fixed-seed evaluation so far
    enemy_ppo.zip + enemy_ppo.json            the policy the game loads, with metadata
and in logs/enemy_ppo/: progress.csv (SB3 metrics) and evaluations.jsonl.

Models are selected on the training objective (catches minus hits minus time
spent), measured on evaluation seeds that never overlap the benchmark seeds.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ai_platform_trainer.ai.envs.enemy_env import GAMMA, REWARD_SPEC, EnemyArenaEnv
from ai_platform_trainer.ai.evaluation.enemy_benchmark import evaluate_policy
from ai_platform_trainer.ai.model_paths import (
    ENEMY_PPO_DIR,
    ENEMY_PPO_LOG_DIR,
    ENEMY_PPO_MODEL,
    model_card_path,
)
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.enemy_policies import SB3Enemy
from ai_platform_trainer.arena.observations import (
    ENEMY_DECISION_FRAMES,
    ENEMY_OBS_FEATURES,
    ENEMY_OBS_SIZE,
    ENEMY_OBS_VERSION,
)

CHECKPOINT_PREFIX = "enemy_ppo"
EVAL_SEED = 10_000  # evaluation rounds never overlap the benchmark seeds (0..N)
SELECTION_METRIC = "objective"


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def write_metadata(
    path: str, model: Any, evaluation: Dict[str, Any], seed: int
) -> None:
    """Model card: everything needed to load, trust and reproduce the policy."""
    meta = {
        "algorithm": "PPO",
        "observation_version": ENEMY_OBS_VERSION,
        "observation_size": ENEMY_OBS_SIZE,
        "observation_features": list(ENEMY_OBS_FEATURES),
        "action": "velocity in [-1, 1] per axis, times enemy_speed px per frame",
        "decision_frames": ENEMY_DECISION_FRAMES,
        "gamma": GAMMA,
        "reward": REWARD_SPEC,
        "selection_metric": SELECTION_METRIC,
        "timesteps": int(model.num_timesteps),
        "seed": seed,
        "eval_arena": asdict(ArenaConfig()),
        "evaluation": evaluation,
        "git_commit": git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        print(json.dumps(meta, indent=2), file=fh)


def checkpoint_steps(path: str) -> int:
    name = os.path.basename(path)
    return int(name[len(CHECKPOINT_PREFIX) + 1 : -len("_steps.zip")])


def latest_checkpoint(directory: str) -> Optional[str]:
    paths = glob.glob(os.path.join(directory, CHECKPOINT_PREFIX + "_*_steps.zip"))
    return max(paths, key=checkpoint_steps) if paths else None


class ArenaEvalCallback(BaseCallback):
    """Fixed-seed evaluation every N steps; logs eval/* metrics and keeps the best model."""

    def __init__(
        self, every: int, rounds: int, out_dir: str, log_dir: str, seed: int
    ) -> None:
        super().__init__()
        self.every = every
        self.rounds = rounds
        self.out_dir = out_dir
        self.log_path = os.path.join(log_dir, "evaluations.jsonl")
        self.seed = seed
        self.best_path = os.path.join(out_dir, "best_model.zip")
        self.best_score = -math.inf
        self.best_eval: Optional[Dict[str, Any]] = None
        self._last = 0
        meta_path = model_card_path(self.best_path)
        if os.path.exists(meta_path) and os.path.exists(self.best_path):
            with open(meta_path, encoding="utf-8") as fh:
                self.best_eval = json.load(fh)["evaluation"]
            self.best_score = self.best_eval[SELECTION_METRIC]

    def _on_training_start(self) -> None:
        self._last = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self.evaluate()
        return True

    def evaluate(self) -> Dict[str, Any]:
        summary = evaluate_policy(
            SB3Enemy(self.model), ArenaConfig(), self.rounds, seed=EVAL_SEED
        )
        for key in (
            "objective",
            "net_per_min",
            "catch_rate",
            "hit_rate",
            "timeout_rate",
            "mean_return",
        ):
            self.logger.record("eval/" + key, summary[key])
        with open(self.log_path, "a", encoding="utf-8") as fh:
            print(
                json.dumps({"timesteps": int(self.num_timesteps), **summary}), file=fh
            )
        if summary[SELECTION_METRIC] > self.best_score:
            self.best_score = summary[SELECTION_METRIC]
            self.best_eval = summary
            self.model.save(self.best_path)
            best_meta = model_card_path(self.best_path)
            write_metadata(best_meta, self.model, summary, self.seed)
        steps = format(self.num_timesteps, ",")
        net = format(summary["net_per_min"], "+.2f")
        best = format(self.best_score, "+.3f")
        score = format(summary[SELECTION_METRIC], "+.3f")
        catch = format(summary["catch_rate"], ".1%")
        hit = format(summary["hit_rate"], ".1%")
        message = (
            f"[eval at {steps} steps] objective {score}  net {net}/min  "
            f"catch {catch}  hit {hit}  (best {best})"
        )
        print(message, flush=True)
        return summary


def train(
    timesteps: int = 10_000_000,
    n_envs: Optional[int] = None,
    seed: int = 0,
    resume: bool = False,
    out_dir: str = ENEMY_PPO_DIR,
    log_dir: str = ENEMY_PPO_LOG_DIR,
    eval_every: int = 500_000,
    eval_rounds: int = 240,
    checkpoint_every: int = 500_000,
) -> str:
    n_envs = n_envs or min(8, os.cpu_count() or 1)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(EnemyArenaEnv, n_envs=n_envs, seed=seed, vec_env_cls=vec_cls)

    start_from = latest_checkpoint(ckpt_dir) if resume else None
    if start_from:
        model = PPO.load(start_from, env=env, device="cpu")
        print(f"Resuming from {start_from} ({model.num_timesteps:,} steps)", flush=True)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=1024,
            n_epochs=10,
            gamma=GAMMA,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
            seed=seed,
            device="cpu",
            verbose=1,
        )
    model.set_logger(configure(log_dir, ["stdout", "csv"]))

    evaluator = ArenaEvalCallback(eval_every, eval_rounds, out_dir, log_dir, seed)
    checkpoints = CheckpointCallback(
        save_freq=max(checkpoint_every // n_envs, 1),
        save_path=ckpt_dir,
        name_prefix=CHECKPOINT_PREFIX,
    )
    try:
        model.learn(
            total_timesteps=max(0, timesteps - model.num_timesteps),
            callback=CallbackList([checkpoints, evaluator]),
            reset_num_timesteps=not start_from,
        )
    finally:
        env.close()
    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)
    final_eval = evaluator.evaluate()

    deploy_from = (
        evaluator.best_path if os.path.exists(evaluator.best_path) else final_path
    )
    model_path = os.path.join(out_dir, os.path.basename(ENEMY_PPO_MODEL))
    shutil.copyfile(deploy_from, model_path)
    meta_path = model_card_path(model_path)
    deployed = PPO.load(model_path, device="cpu")
    write_metadata(meta_path, deployed, evaluator.best_eval or final_eval, seed)
    print(f"Deployed {deploy_from} -> {model_path}", flush=True)
    return model_path
