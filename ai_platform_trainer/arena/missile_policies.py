"""Missile guidance for the arena, matching what the game does.

SacGuidance steers a simulated missile with the same SAC model, the same
observation and the same turn scale as SmartMissile in the game. Training and
benchmarks use it, so the enemy learns against the missiles the player faces.

The SAC actor is also copied into plain numpy (NumpySacActor). The copy gives
the same deterministic action and is about 40x faster than SB3's predict(),
which matters because the arena asks for a missile action every frame.
"""
import math
from typing import Any, List

import numpy as np
import torch

from ai_platform_trainer.arena.observations import (
    SAC_TURN_SCALE_DEG,
    sac_missile_observation,
)
from ai_platform_trainer.arena.state import MissileState

# The largest difference allowed between the numpy copy and SB3's own output.
MAX_ACTION_DIFFERENCE = 1e-5


class NumpySacActor:
    """The deterministic SAC actor as numpy: action = tanh(mu(latent(obs)))."""

    def __init__(self, sb3_sac_model: Any) -> None:
        actor = sb3_sac_model.policy.actor
        self._hidden: List[tuple] = []
        for layer in actor.latent_pi:
            if isinstance(layer, torch.nn.Linear):
                self._hidden.append(
                    ("linear", _np(layer.weight), _np(layer.bias))
                )
            elif isinstance(layer, torch.nn.ReLU):
                self._hidden.append(("relu", None, None))
            elif isinstance(layer, torch.nn.Tanh):
                self._hidden.append(("tanh", None, None))
            else:
                raise TypeError(f"Unsupported SAC actor layer: {layer}")
        self._mu_weight = _np(actor.mu.weight)
        self._mu_bias = _np(actor.mu.bias)

    def act(self, observation: np.ndarray) -> np.ndarray:
        h = observation
        for kind, weight, bias in self._hidden:
            if kind == "linear":
                h = h @ weight.T + bias
            elif kind == "relu":
                h = np.maximum(h, 0.0)
            else:
                h = np.tanh(h)
        return np.tanh(h @ self._mu_weight.T + self._mu_bias)


def _np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32)


class SacGuidance:
    """Guides an arena missile the way SmartMissile guides a game missile."""

    name = "sac"

    def __init__(self, model_path: str) -> None:
        from stable_baselines3 import SAC

        self.model = SAC.load(model_path, device="cpu")
        self.actor = NumpySacActor(self.model)
        self._verify_copy()

    def _verify_copy(self, samples: int = 256) -> None:
        """Refuse to run if the numpy copy disagrees with SB3's own predict()."""
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(samples):
            obs = rng.uniform(-1.0, 1.0, size=(11,)).astype(np.float32)
            expected, _ = self.model.predict(obs, deterministic=True)
            worst = max(worst, float(np.max(np.abs(self.actor.act(obs) - expected))))
        if worst > MAX_ACTION_DIFFERENCE:
            raise RuntimeError(
                f"numpy SAC actor differs from SB3 by {worst:.2e}; not using it"
            )

    def __call__(
        self, missile: MissileState, target_x: float, target_y: float, config: Any
    ) -> None:
        """One frame of guidance: the same steps as SmartMissile.update_with_ai."""
        target_vx = target_x - missile.last_target_x
        target_vy = target_y - missile.last_target_y
        missile.last_target_x = target_x
        missile.last_target_y = target_y

        obs = sac_missile_observation(
            missile.x,
            missile.y,
            missile.vx,
            missile.vy,
            target_x,
            target_y,
            target_vx,
            target_vy,
            config.width,
            config.height,
        )
        action = float(self.actor.act(obs)[0])
        _apply_turn(missile, action * SAC_TURN_SCALE_DEG, config.missile_speed)


def _apply_turn(missile: MissileState, turn_deg: float, speed: float) -> None:
    """Turn the missile's heading by turn_deg, keeping its speed (SmartMissile._apply_turn)."""
    angle = math.atan2(missile.vy, missile.vx) + math.radians(turn_deg)
    missile.vx = speed * math.cos(angle)
    missile.vy = speed * math.sin(angle)
