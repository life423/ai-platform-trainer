"""Train the AI Agent (the enemy) on the canonical arena rules.

Two stages:
1. Behaviour cloning. The scripted chase enemy catches the player reliably, so the
   policy first learns to copy its moves over many rounds and bots. This gives a
   working chaser instead of one that has to discover chasing from nothing.
2. PPO fine-tuning in ai_platform_trainer.ai.envs.chaser_env. The reward pays for
   catches and charges for missile hits, so the fine-tuned enemy learns to dodge.

    python -m ai_platform_trainer train-enemy --rounds 300 --timesteps 1000000

The model and its JSON card are written to --out (models/enemy/enemy.zip).
"""
import dataclasses
import json
import logging
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO

from ai_platform_trainer.ai.envs.chaser_env import (
    CATCH_REWARD,
    HIT_REWARD,
    ROUND_FRAMES,
    THREAT_WEIGHT,
    ChaserEnv,
)
from ai_platform_trainer.arena.config import ArenaConfig
from ai_platform_trainer.arena.enemy_policies import ChaseEnemy

ENEMY_OBS_NET = (128, 128)
BC_EPOCHS = 30
BC_BATCH = 256
BC_LEARNING_RATE = 1e-3
PPO_LEARNING_RATE = 1e-4
FINE_TUNE_LOG_STD = math.log(0.3)  # exploration around the cloned moves


def collect_demonstrations(
    env: ChaserEnv, rounds: int, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Observation and move pairs from the chase enemy, one per decision."""
    chase = ChaseEnemy()
    speed = env.config.enemy_speed
    observations, actions = [], []
    for round_index in range(rounds):
        env.reset(seed=seed + round_index)
        chase.reset(env.episode.state, env.rng)
        done = False
        while not done:
            dx, dy = chase.act(env.episode.state)
            target = np.clip([dx / speed, dy / speed], -1.0, 1.0).astype(np.float32)
            observations.append(env._observe())
            actions.append(target)
            _, _, _, truncated, _ = env.step(target)
            done = truncated
    return np.stack(observations), np.stack(actions)


def clone_behaviour(model: PPO, observations: np.ndarray, actions: np.ndarray) -> float:
    """Fit the policy's mean move to the demonstrations. Returns the final loss."""
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=BC_LEARNING_RATE)
    obs = torch.as_tensor(observations, dtype=torch.float32)
    target = torch.as_tensor(actions, dtype=torch.float32)
    loss_value = float("nan")
    for epoch in range(BC_EPOCHS):
        order = torch.randperm(len(obs))
        for start in range(0, len(obs), BC_BATCH):
            index = order[start : start + BC_BATCH]
            features = policy.extract_features(obs[index], policy.features_extractor)
            mean = policy.action_net(policy.mlp_extractor.forward_actor(features))
            loss = torch.nn.functional.mse_loss(mean, target[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
        if epoch % 10 == 0 or epoch == BC_EPOCHS - 1:
            logging.info(f"behaviour cloning epoch {epoch + 1}/{BC_EPOCHS}: loss {loss_value:.4f}")
    return loss_value


def write_card(path: str, card: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
        fh.write("\n")


def train(
    rounds: int,
    timesteps: int,
    out_path: str,
    seed: int = 0,
    config: Optional[ArenaConfig] = None,
    hit_reward: float = HIT_REWARD,
    threat_weight: float = THREAT_WEIGHT,
) -> str:
    """Behaviour-clone the chase enemy, fine-tune it with PPO, save model and card."""
    started = time.time()
    env = ChaserEnv(config=config, hit_reward=hit_reward, threat_weight=threat_weight)
    logging.info(f"collecting {rounds} rounds of chase demonstrations")
    observations, actions = collect_demonstrations(env, rounds, seed)
    logging.info(f"{len(observations)} decisions collected")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": dict(pi=list(ENEMY_OBS_NET), vf=list(ENEMY_OBS_NET))},
        learning_rate=PPO_LEARNING_RATE,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    final_loss = clone_behaviour(model, observations, actions)
    model.policy.log_std.data.fill_(FINE_TUNE_LOG_STD)

    logging.info(f"fine-tuning with PPO for {timesteps} decisions")
    model.learn(total_timesteps=timesteps, progress_bar=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)

    card_config = dataclasses.asdict(env.config)
    write_card(
        os.path.splitext(out_path)[0] + ".json",
        {
            "name": "AI Agent",
            "algorithm": "behaviour cloning of the chase enemy, then PPO",
            "observation_size": int(env.observation_space.shape[0]),
            "decision_frames": 4,
            "round_frames": ROUND_FRAMES,
            "eval_arena": card_config,
            "missile_guidance": "sac",
            "clone_rounds": rounds,
            "clone_loss": final_loss,
            "timesteps": int(model.num_timesteps),
            "reward": (
                f"+{CATCH_REWARD:g} per catch, {hit_reward:g} per missile hit, "
                f"potential shaping for distance to the player and for closing "
                f"missiles (threat weight {threat_weight:g}); no time cost"
            ),
            "seed": seed,
            "minutes": round((time.time() - started) / 60.0, 1),
        },
    )
    logging.info(f"saved {out_path}")
    return out_path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the AI Agent enemy")
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--out", default="models/enemy/enemy.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hit-reward", type=float, default=HIT_REWARD)
    parser.add_argument("--threat-weight", type=float, default=THREAT_WEIGHT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    train(
        args.rounds,
        args.timesteps,
        args.out,
        args.seed,
        hit_reward=args.hit_reward,
        threat_weight=args.threat_weight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
