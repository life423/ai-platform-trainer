"""Enemy controllers that act on an ArenaState: baselines and learned agents.

Every policy returns a per-frame displacement in pixels. The sim clips capped
policies to the arena enemy_speed per axis, the same cap for every enemy. The
shipped Adaptive AI can opt out of the cap to reproduce its speed ramp to 10+ px
per frame.
"""
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from ai_platform_trainer.arena.observations import (
    ENEMY_DECISION_FRAMES,
    ENEMY_OBS_SIZE,
    build_enemy_observation,
    enemy_action_to_displacement,
    enemy_to_player,
)
from ai_platform_trainer.arena.state import ArenaState

Displacement = Tuple[float, float]


class EnemyPolicy:
    """Interface: reset once per episode, then act once per frame."""

    name = "base"
    capped = True

    def reset(self, state: ArenaState, rng: random.Random) -> None:
        """Called at the start of every episode."""

    def act(self, state: ArenaState) -> Displacement:
        raise NotImplementedError


class RandomEnemy(EnemyPolicy):
    """Holds a random velocity for a random 10-40 frames, repeatedly."""

    name = "random"

    def reset(self, state: ArenaState, rng: random.Random) -> None:
        self.rng = rng
        self._hold = 0
        self._move: Displacement = (0.0, 0.0)

    def act(self, state: ArenaState) -> Displacement:
        if self._hold <= 0:
            speed = state.config.enemy_speed
            self._move = (
                self.rng.uniform(-speed, speed),
                self.rng.uniform(-speed, speed),
            )
            self._hold = self.rng.randint(10, 40)
        self._hold -= 1
        return self._move


class ChaseEnemy(EnemyPolicy):
    """Pure pursuit along the shortest wrapped path at full speed; never dodges."""

    name = "chase"

    def act(self, state: ArenaState) -> Displacement:
        dx, dy = enemy_to_player(state)
        scale = max(abs(dx), abs(dy))
        if scale < 1e-9:
            return 0.0, 0.0
        speed = state.config.enemy_speed
        return dx / scale * speed, dy / scale * speed


class AdaptiveEnemy(EnemyPolicy):
    """The shipped AdaptiveStagedEnemyAI heuristics, driven on arena state.

    It starts fully warmed up (nightmare stage), which is how it plays after
    the first few seconds of a real session. capped=True holds it to the arena
    enemy speed like every other enemy; capped=False keeps its own speed ramp
    (up to 10 px per frame plus a chase boost), i.e. the enemy as shipped.
    """

    def __init__(self, capped: bool = True, warmup_frames: int = 900) -> None:
        self.capped = capped
        self.name = "adaptive" if capped else "adaptive_shipped"
        self.warmup_frames = warmup_frames
        self._ai: Any = None

    def reset(self, state: ArenaState, rng: random.Random) -> None:
        from ai_platform_trainer.entities.enemy_learning import AdaptiveStagedEnemyAI

        # The legacy class draws from the global RNG; seed it per episode.
        random.seed(rng.getrandbits(32))
        cfg = state.config
        self._ai = AdaptiveStagedEnemyAI(cfg.width, cfg.height)
        self._ai.total_frames = self.warmup_frames

    def act(self, state: ArenaState) -> Displacement:
        ai = self._ai
        ai.pos = {"x": state.enemy.x, "y": state.enemy.y}
        ai.total_frames += 1
        ai._update_behavior_stage()
        # Play mode hands it every missile, including ones still exploding.
        missiles = [SimpleNamespace(pos={"x": m.x, "y": m.y}) for m in state.missiles]
        return ai._get_movement_decision(state.player.x, state.player.y, missiles)


class LegacySupervisedEnemy(EnemyPolicy):
    """models/enemy_ai_model.pth, the network behind the old Trained AI option."""

    name = "legacy_supervised"

    def __init__(self, model_path: str = "models/enemy_ai_model.pth") -> None:
        import torch

        from ai_platform_trainer.ai.models.enemy_movement_model import (
            EnemyMovementModel,
        )

        self._torch = torch
        self.model = EnemyMovementModel(input_size=5, hidden_size=64, output_size=2)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self._context: Any = None

    def reset(self, state: ArenaState, rng: random.Random) -> None:
        from ai_platform_trainer.core.screen_context import ScreenContext

        ScreenContext.initialize(state.config.width, state.config.height)
        self._context = ScreenContext.get_instance()

    def act(self, state: ArenaState) -> Displacement:
        obs = self._context.create_enemy_observation(
            {"x": state.player.x, "y": state.player.y},
            {"x": state.enemy.x, "y": state.enemy.y},
            state.config.player_speed,
        )
        row = [
            obs["player_x"],
            obs["player_y"],
            obs["enemy_x"],
            obs["enemy_y"],
            obs["distance"],
        ]
        with self._torch.no_grad():
            out = self.model(self._torch.tensor([row], dtype=self._torch.float32))[0]
        speed = 5.0  # EnemyPlay.speed
        return float(out[0]) * speed, float(out[1]) * speed


class SB3Enemy(EnemyPolicy):
    """A trained Stable-Baselines3 policy acting through the shared encoder.

    As in training, it picks a move every ENEMY_DECISION_FRAMES frames and
    holds it in between. Training evaluation, the benchmark and the game all
    run trained models through this one class.
    """

    def __init__(
        self, model: Any, name: str = "ppo", deterministic: bool = True
    ) -> None:
        shape = tuple(model.observation_space.shape)
        if shape != (ENEMY_OBS_SIZE,):
            raise ValueError(
                f"model expects observations of shape {shape}, "
                f"but the arena encoder produces ({ENEMY_OBS_SIZE},)"
            )
        self.model = model
        self.name = name
        self.deterministic = deterministic
        self._move: Displacement = (0.0, 0.0)
        self._frames_left = 0

    @classmethod
    def load(cls, path: str, name: str = "ppo") -> "SB3Enemy":
        from stable_baselines3 import PPO

        return cls(PPO.load(path, device="cpu"), name=name)

    def reset(
        self, state: Optional[ArenaState] = None, rng: Optional[random.Random] = None
    ) -> None:
        self._frames_left = 0

    def act(self, state: ArenaState) -> Displacement:
        if self._frames_left <= 0:
            action, _ = self.model.predict(
                build_enemy_observation(state), deterministic=self.deterministic
            )
            self._move = enemy_action_to_displacement(action, state.config)
            self._frames_left = ENEMY_DECISION_FRAMES
        self._frames_left -= 1
        return self._move


def baseline_policies() -> List[EnemyPolicy]:
    """Scripted and legacy enemies every learned agent is compared against."""
    return [
        RandomEnemy(),
        ChaseEnemy(),
        AdaptiveEnemy(capped=True),
        AdaptiveEnemy(capped=False),
        LegacySupervisedEnemy(),
    ]


POLICY_NAMES: Dict[str, str] = {
    "random": "Random movement",
    "chase": "Direct chase",
    "adaptive": "Adaptive AI (scripted)",
    "adaptive_shipped": "Adaptive AI (as shipped, speed 10+)",
    "legacy_supervised": "Legacy supervised NN",
    "ppo": "PPO agent",
}
