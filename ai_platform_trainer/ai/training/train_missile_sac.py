"""Train the SAC missile on the canonical arena rules.

    python -m ai_platform_trainer.ai.training.train_missile_sac --timesteps 300000

The model is saved to --out (default models/missile_sac_model_final, which the game
and the arena load as models/missile_sac_model_final.zip). The environment is
ai_platform_trainer.ai.envs.missile_env, which runs the arena's rules.
"""
import argparse
import logging
from typing import List, Optional

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from ai_platform_trainer.ai.envs.missile_env import MissileArenaEnv


def train(timesteps: int, out_path: str, seed: int = 0) -> str:
    """Train a SAC missile and save it. Returns the saved file name."""
    env = Monitor(MissileArenaEnv())
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=2_000,
        batch_size=256,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    logging.info(f"Training the SAC missile for {timesteps} steps")
    model.learn(total_timesteps=timesteps, progress_bar=False)
    model.save(out_path)
    logging.info(f"Saved {out_path}.zip")
    return f"{out_path}.zip"


def hit_rate(
    model_path: str, episodes: int = 500, seed: int = 12345, still_fraction: float = 0.3
) -> float:
    """The share of episodes in which the missile hits, with the policy run deterministically."""
    model = SAC.load(model_path, device="cpu")
    env = MissileArenaEnv(still_fraction=still_fraction)
    hits = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        info = {"hit": False}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        hits += bool(info["hit"])
    return hits / episodes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the SAC missile on the arena rules"
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", default="models/missile_sac_model_final")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train(args.timesteps, args.out, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
