"""Enemy controllers that act on an ArenaState: baselines and learned agents.

Every policy returns a per-frame displacement in pixels. The sim clips capped
policies to the arena enemy_speed per axis, the same cap for every enemy.
"""
import random
from typing import Any, Dict, List, Optional, Tuple

from ai_platform_trainer.ai.model_paths import load_model_card
from ai_platform_trainer.arena.observations import (
    ENEMY_DECISION_FRAMES,
    ENEMY_OBS_SIZE,
    build_enemy_observation,
    enemy_action_to_displacement,
    enemy_to_player,
    missile_threats,
)
from ai_platform_trainer.arena.state import ArenaState

Displacement = Tuple[float, float]
DODGE_RADIUS_PX = 100.0  # a closing missile this near makes the dodger run


def _at_speed(dx: float, dy: float, speed: float) -> Displacement:
    """Scale a direction so its larger axis moves at speed, as the arena caps it."""
    scale = max(abs(dx), abs(dy))
    if scale < 1e-9:
        return 0.0, 0.0
    return dx / scale * speed, dy / scale * speed


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
        return _at_speed(dx, dy, state.config.enemy_speed)


class DodgeChaseEnemy(EnemyPolicy):
    """Chase the player, but run straight away from any missile closing in.

    Missiles fly slower than the enemy, so running directly away from one keeps
    it from catching up. It decides on the same 4-frame clock as the learned
    agent, so the comparison is fair. Scripted, not learned: it is a benchmark
    baseline and the teacher the AI Agent copies before PPO fine-tuning.
    """

    name = "dodge"

    def __init__(
        self,
        radius: float = DODGE_RADIUS_PX,
        decision_frames: int = ENEMY_DECISION_FRAMES,
    ) -> None:
        self.radius = radius
        self.decision_frames = decision_frames
        self._chase = ChaseEnemy()
        self._move: Displacement = (0.0, 0.0)
        self._frames_left = 0

    def reset(
        self, state: Optional[ArenaState] = None, rng: Optional[random.Random] = None
    ) -> None:
        self._frames_left = 0

    def act(self, state: ArenaState) -> Displacement:
        if self._frames_left <= 0:
            self._move = self.decide(state)
            self._frames_left = self.decision_frames
        self._frames_left -= 1
        return self._move

    def decide(self, state: ArenaState) -> Displacement:
        """The move for this state. It has no decision clock, so the caller holds it."""
        away_x = away_y = 0.0
        for distance, dx, dy, missile in missile_threats(state):
            if distance > self.radius:
                break  # threats are sorted nearest first
            # (dx, dy) points from the enemy to the missile; it is closing when
            # its velocity points back at the enemy.
            closing = missile.vx * dx + missile.vy * dy < 0.0
            if closing and distance > 1e-6:
                away_x -= dx / distance
                away_y -= dy / distance
        if away_x or away_y:
            return _at_speed(away_x, away_y, state.config.enemy_speed)
        return self._chase.act(state)


class SB3Enemy(EnemyPolicy):
    """A trained Stable-Baselines3 policy acting through the shared encoder.

    It picks a move every decision_frames frames and holds it in between, at
    the rate it was trained with (read from its model card when loaded from
    disk). Training evaluation, the benchmark and the game all run models
    through this one class.
    """

    def __init__(
        self,
        model: Any,
        name: str = "ppo",
        deterministic: bool = True,
        decision_frames: int = ENEMY_DECISION_FRAMES,
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
        self.decision_frames = decision_frames
        self._move: Displacement = (0.0, 0.0)
        self._frames_left = 0

    @classmethod
    def load(cls, path: str, name: str = "ppo") -> "SB3Enemy":
        from stable_baselines3 import PPO

        card = load_model_card(path) or {}
        frames = int(card.get("decision_frames", ENEMY_DECISION_FRAMES))
        return cls(PPO.load(path, device="cpu"), name=name, decision_frames=frames)

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
            self._frames_left = self.decision_frames
        self._frames_left -= 1
        return self._move


def baseline_policies() -> List[EnemyPolicy]:
    """Scripted enemies every learned agent is compared against."""
    return [RandomEnemy(), ChaseEnemy(), DodgeChaseEnemy()]


POLICY_NAMES: Dict[str, str] = {
    "random": "Random movement",
    "chase": "Direct chase",
    "dodge": "Scripted dodge",
    "ppo": "PPO agent",
}
