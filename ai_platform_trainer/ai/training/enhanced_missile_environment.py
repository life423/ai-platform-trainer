"""
Enhanced Missile Environment with Improved Reward Function

This module addresses the reward function issues identified:
- Time penalty to prevent stalling
- Balanced progress rewards to prevent oscillation
- Curriculum learning support
- Better behavior analysis
"""
import logging
import os
import numpy as np
import torch
import pygame
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.core.screen_context import ScreenContext


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    name: str
    target_speed_range: Tuple[float, float]
    target_movement_pattern: str
    target_agility: float  # How often target changes direction (0-1)
    episode_length: int
    success_threshold: float  # Distance for successful hit
    min_success_rate: float  # Success rate needed to advance to next stage


class EnhancedMissileEnvironment(gym.Env):
    """
    Enhanced missile environment with improved reward function and curriculum learning.
    
    Key improvements:
    - Time penalty to encourage fast interception
    - Balanced progress rewards to prevent oscillation
    - Curriculum learning support
    - Better behavioral analysis
    """
    
    def __init__(self, screen_width: int = 800, screen_height: int = 600, 
                 curriculum_stage: Optional[CurriculumStage] = None):
        super().__init__()
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Initialize ScreenContext for resolution independence
        ScreenContext.initialize(screen_width, screen_height)
        
        # Action space: turn rate (-1 to 1, normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Enhanced observation space with curriculum info
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(13,), dtype=np.float32
        )
        
        # Initialize pygame for headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.init()
        self.screen = pygame.Surface((screen_width, screen_height))
        
        # Curriculum learning
        self.curriculum_stage = curriculum_stage or self._get_default_curriculum_stage()
        
        # Enhanced reward parameters
        self.time_penalty_weight = 0.001  # Small penalty per timestep
        self.progress_smoothing_window = 5  # Smooth progress over multiple steps
        self.oscillation_penalty_threshold = 3  # Consecutive direction changes
        self.efficiency_bonus_threshold = 0.8  # Bonus for efficient trajectories
        
        # Behavioral tracking
        self.distance_history = []
        self.action_history = []
        self.trajectory_efficiency = 0.0
        self.oscillation_count = 0
        self.last_action_direction = 0
        
        self.reset()
    
    def _get_default_curriculum_stage(self) -> CurriculumStage:
        """Get default curriculum stage for standard training."""
        return CurriculumStage(
            name="standard",
            target_speed_range=(2.0, 4.0),
            target_movement_pattern="linear",
            target_agility=0.1,
            episode_length=500,
            success_threshold=20,
            min_success_rate=0.8
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment with curriculum-aware initialization."""
        super().reset(seed=seed)
        
        # Create missile at random position
        missile_x = np.random.uniform(50, self.screen_width - 50)
        missile_y = np.random.uniform(50, self.screen_height - 50)
        
        # Random initial velocity
        angle = np.random.uniform(0, 2 * np.pi)
        initial_speed = np.random.uniform(6.0, 10.0)
        initial_vx = initial_speed * np.cos(angle)
        initial_vy = initial_speed * np.sin(angle)
        
        self.missile = Missile(missile_x, missile_y, speed=initial_speed, vx=initial_vx, vy=initial_vy)
        
        # Create target based on curriculum stage
        self._create_curriculum_target()
        
        # Reset tracking variables
        self.steps = 0
        self.max_steps = self.curriculum_stage.episode_length
        self.distance_history = [self._calculate_distance()]
        self.action_history = []
        self.trajectory_efficiency = 0.0
        self.oscillation_count = 0
        self.last_action_direction = 0
        self.initial_distance = self.distance_history[0]
        
        return self._get_observation(), {}
    
    def _create_curriculum_target(self):
        """Create target based on curriculum stage configuration."""
        # Position target away from missile
        target_x = np.random.uniform(100, self.screen_width - 100)
        target_y = np.random.uniform(100, self.screen_height - 100)
        
        # Ensure minimum distance from missile
        while np.sqrt((target_x - self.missile.pos["x"])**2 + (target_y - self.missile.pos["y"])**2) < 200:
            target_x = np.random.uniform(100, self.screen_width - 100)
            target_y = np.random.uniform(100, self.screen_height - 100)
        
        # Set target speed based on curriculum
        speed = np.random.uniform(*self.curriculum_stage.target_speed_range)
        angle = np.random.uniform(0, 2 * np.pi)
        
        self.target = {
            "x": target_x,
            "y": target_y,
            "vx": speed * np.cos(angle),
            "vy": speed * np.sin(angle),
            "pattern": self.curriculum_stage.target_movement_pattern,
            "agility": self.curriculum_stage.target_agility,
            "pattern_time": 0.0,
            "last_direction_change": 0
        }
    
    def step(self, action):
        """Execute step with enhanced reward calculation."""
        self.steps += 1
        
        # Track action for oscillation detection
        action_value = action[0]
        self.action_history.append(action_value)
        
        # Detect oscillation
        if len(self.action_history) >= 2:
            current_direction = 1 if action_value > 0 else -1
            if current_direction != self.last_action_direction:
                self.oscillation_count += 1
            self.last_action_direction = current_direction
        
        # Apply missile action
        turn_rate = np.clip(action_value * 20.0, -20.0, 20.0)
        current_angle = np.arctan2(self.missile.vy, self.missile.vx)
        new_angle = current_angle + np.radians(turn_rate)
        
        self.missile.vx = self.missile.speed * np.cos(new_angle)
        self.missile.vy = self.missile.speed * np.sin(new_angle)
        self.missile.update()
        
        # Update target with curriculum-aware movement
        self._update_curriculum_target()
        
        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(action_value)
        
        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.max_steps
        
        # Update tracking
        current_distance = self._calculate_distance()
        self.distance_history.append(current_distance)
        self._update_trajectory_efficiency()
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _update_curriculum_target(self):
        """Update target movement based on curriculum settings."""
        self.target["pattern_time"] += 0.1
        
        if self.target["pattern"] == "static":
            # Static target - no movement
            pass
        
        elif self.target["pattern"] == "linear":
            # Linear movement with curriculum-based direction changes
            self.target["x"] += self.target["vx"]
            self.target["y"] += self.target["vy"]
            
            # Curriculum-based agility (direction changes)
            if np.random.random() < self.target["agility"] / 100:
                angle = np.random.uniform(0, 2 * np.pi)
                speed = np.linalg.norm([self.target["vx"], self.target["vy"]])
                self.target["vx"] = speed * np.cos(angle)
                self.target["vy"] = speed * np.sin(angle)
                self.target["last_direction_change"] = self.steps
            
            # Bounce off walls
            if self.target["x"] <= 20 or self.target["x"] >= self.screen_width - 20:
                self.target["vx"] *= -1
            if self.target["y"] <= 20 or self.target["y"] >= self.screen_height - 20:
                self.target["vy"] *= -1
        
        elif self.target["pattern"] == "circular":
            # Circular movement
            center_x = self.screen_width / 2
            center_y = self.screen_height / 2
            radius = 150
            angle = self.target["pattern_time"] * self.target["agility"]
            self.target["x"] = center_x + radius * np.cos(angle)
            self.target["y"] = center_y + radius * np.sin(angle)
            
        elif self.target["pattern"] == "evasive":
            # Evasive movement - moves away from missile
            missile_x, missile_y = self.missile.pos["x"], self.missile.pos["y"]
            target_x, target_y = self.target["x"], self.target["y"]
            
            # Vector from target to missile
            dx = missile_x - target_x
            dy = missile_y - target_y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Move away from missile
                evasion_strength = self.target["agility"] * 3.0
                self.target["vx"] = -dx / distance * evasion_strength
                self.target["vy"] = -dy / distance * evasion_strength
            
            self.target["x"] += self.target["vx"]
            self.target["y"] += self.target["vy"]
        
        # Keep target in bounds
        self.target["x"] = np.clip(self.target["x"], 20, self.screen_width - 20)
        self.target["y"] = np.clip(self.target["y"], 20, self.screen_height - 20)
    
    def _calculate_enhanced_reward(self, action_value: float) -> float:
        """
        Enhanced reward function addressing identified issues:
        1. Time penalty to prevent stalling
        2. Balanced progress rewards
        3. Oscillation penalty
        4. Trajectory efficiency bonus
        """
        distance = self._calculate_distance()
        
        # 1. SUCCESS REWARD (unchanged - clear signal)
        if distance < self.curriculum_stage.success_threshold:
            # Bonus for faster completion
            time_bonus = max(0, (self.max_steps - self.steps) / self.max_steps * 2.0)
            return 20.0 + time_bonus
        
        # 2. TIME PENALTY - discourage stalling
        time_penalty = -self.time_penalty_weight * self.steps
        
        # 3. DISTANCE-BASED REWARD (primary signal)
        max_distance = np.sqrt(self.screen_width**2 + self.screen_height**2)
        distance_reward = -(distance / max_distance) * 2.0
        
        # 4. SMOOTHED PROGRESS REWARD (prevent oscillation exploitation)
        progress_reward = 0.0
        if len(self.distance_history) >= self.progress_smoothing_window:
            # Compare current distance to smoothed historical average
            recent_avg = np.mean(self.distance_history[-self.progress_smoothing_window:])
            older_avg = np.mean(self.distance_history[-self.progress_smoothing_window*2:-self.progress_smoothing_window]) if len(self.distance_history) >= self.progress_smoothing_window*2 else recent_avg
            
            smoothed_progress = older_avg - recent_avg
            if smoothed_progress > 0:
                progress_reward = min(smoothed_progress / max_distance * 3.0, 1.0)  # Capped
            else:
                progress_reward = max(smoothed_progress / max_distance * 1.0, -0.5)  # Less penalty
        
        # 5. OSCILLATION PENALTY - prevent erratic behavior
        oscillation_penalty = 0.0
        if self.oscillation_count > self.oscillation_penalty_threshold:
            excess_oscillations = self.oscillation_count - self.oscillation_penalty_threshold
            oscillation_penalty = -excess_oscillations * 0.1
        
        # 6. TRAJECTORY EFFICIENCY BONUS
        efficiency_bonus = 0.0
        if self.trajectory_efficiency > self.efficiency_bonus_threshold:
            efficiency_bonus = (self.trajectory_efficiency - self.efficiency_bonus_threshold) * 1.0
        
        # 7. VELOCITY ALIGNMENT REWARD (smoother than before)
        missile_to_target = np.array([
            self.target["x"] - self.missile.pos["x"],
            self.target["y"] - self.missile.pos["y"]
        ])
        missile_velocity = np.array([self.missile.vx, self.missile.vy])
        
        alignment_reward = 0.0
        if np.linalg.norm(missile_to_target) > 0 and np.linalg.norm(missile_velocity) > 0:
            cos_angle = np.dot(missile_to_target, missile_velocity) / (
                np.linalg.norm(missile_to_target) * np.linalg.norm(missile_velocity)
            )
            # Smooth alignment reward - only positive when well-aligned
            alignment_reward = max(0, cos_angle - 0.5) * 0.4
        
        # 8. BOUNDARY PENALTY (smooth)
        boundary_penalty = self._calculate_boundary_penalty()
        
        # 9. ACTION SMOOTHNESS REWARD
        action_smoothness = -abs(action_value) * 0.02  # Prefer smaller corrections
        
        # Combine all rewards with careful weighting
        total_reward = (
            distance_reward * 1.0 +           # Primary signal
            progress_reward * 0.8 +           # Reduced to prevent exploitation
            time_penalty * 1.0 +              # NEW: Encourage speed
            oscillation_penalty * 1.0 +       # NEW: Discourage oscillation
            efficiency_bonus * 0.5 +          # NEW: Reward efficient paths
            alignment_reward * 0.3 +          # Reduced weight
            boundary_penalty * 1.0 +          # Unchanged
            action_smoothness * 1.0           # NEW: Encourage smooth control
        )
        
        # Clip to prevent extreme values
        return np.clip(total_reward, -10.0, 25.0)
    
    def _update_trajectory_efficiency(self):
        """Calculate trajectory efficiency to reward direct paths."""
        if len(self.distance_history) < 2:
            return
        
        # Calculate ideal vs actual path efficiency
        initial_distance = self.distance_history[0]
        current_distance = self.distance_history[-1]
        
        # Ideal progress (straight line)
        ideal_progress = initial_distance - current_distance
        
        # Actual path length (sum of movements)
        actual_path_length = 0
        for i in range(1, len(self.distance_history)):
            actual_path_length += abs(self.distance_history[i-1] - self.distance_history[i])
        
        # Efficiency = how close to ideal straight-line path
        if actual_path_length > 0:
            self.trajectory_efficiency = min(1.0, ideal_progress / actual_path_length)
        else:
            self.trajectory_efficiency = 0.0
    
    def _calculate_boundary_penalty(self) -> float:
        """Calculate smooth boundary penalty."""
        penalty = 0.0
        margin = 50
        
        # X boundaries
        if self.missile.pos["x"] < margin:
            penalty -= (margin - self.missile.pos["x"]) / margin * 0.5
        elif self.missile.pos["x"] > self.screen_width - margin:
            penalty -= (self.missile.pos["x"] - (self.screen_width - margin)) / margin * 0.5
        
        # Y boundaries
        if self.missile.pos["y"] < margin:
            penalty -= (margin - self.missile.pos["y"]) / margin * 0.5
        elif self.missile.pos["y"] > self.screen_height - margin:
            penalty -= (self.missile.pos["y"] - (self.screen_height - margin)) / margin * 0.5
        
        return penalty
    
    def _get_observation(self) -> np.ndarray:
        """Enhanced observation with curriculum and behavioral info."""
        # Base observation (same as before)
        missile_x_norm = (self.missile.pos["x"] / self.screen_width) * 2 - 1
        missile_y_norm = (self.missile.pos["y"] / self.screen_height) * 2 - 1
        target_x_norm = (self.target["x"] / self.screen_width) * 2 - 1
        target_y_norm = (self.target["y"] / self.screen_height) * 2 - 1
        
        missile_vx_norm = self.missile.vx / 10.0
        missile_vy_norm = self.missile.vy / 10.0
        target_vx_norm = self.target["vx"] / 5.0
        target_vy_norm = self.target["vy"] / 5.0
        
        dx = self.target["x"] - self.missile.pos["x"]
        dy = self.target["y"] - self.missile.pos["y"]
        distance = np.sqrt(dx * dx + dy * dy)
        distance_norm = distance / (np.sqrt(self.screen_width**2 + self.screen_height**2))
        
        angle_to_target = np.arctan2(dy, dx)
        angle_to_target_norm = angle_to_target / np.pi
        
        missile_angle = np.arctan2(self.missile.vy, self.missile.vx)
        relative_angle = angle_to_target - missile_angle
        relative_angle_norm = np.arctan2(np.sin(relative_angle), np.cos(relative_angle)) / np.pi
        
        # Enhanced observations
        time_remaining_norm = (self.max_steps - self.steps) / self.max_steps
        trajectory_efficiency_norm = self.trajectory_efficiency
        
        return np.array([
            missile_x_norm, missile_y_norm, 
            missile_vx_norm, missile_vy_norm,
            target_x_norm, target_y_norm, 
            target_vx_norm, target_vy_norm,
            distance_norm, angle_to_target_norm, relative_angle_norm,
            time_remaining_norm, trajectory_efficiency_norm
        ], dtype=np.float32)
    
    def _calculate_distance(self) -> float:
        """Calculate distance between missile and target."""
        dx = self.missile.pos["x"] - self.target["x"]
        dy = self.missile.pos["y"] - self.target["y"]
        return np.sqrt(dx * dx + dy * dy)
    
    def _is_terminated(self) -> bool:
        """Check termination conditions."""
        # Hit target
        if self._calculate_distance() < self.curriculum_stage.success_threshold:
            return True
        
        # Missile way out of bounds
        if (self.missile.pos["x"] < -100 or self.missile.pos["x"] > self.screen_width + 100 or
            self.missile.pos["y"] < -100 or self.missile.pos["y"] > self.screen_height + 100):
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for analysis."""
        return {
            "distance": self._calculate_distance(),
            "trajectory_efficiency": self.trajectory_efficiency,
            "oscillation_count": self.oscillation_count,
            "curriculum_stage": self.curriculum_stage.name,
            "target_pattern": self.target["pattern"],
            "steps": self.steps,
            "success": self._calculate_distance() < self.curriculum_stage.success_threshold
        }


class CurriculumManager:
    """Manages curriculum learning progression for missile training."""
    
    def __init__(self):
        self.stages = self._create_curriculum_stages()
        self.current_stage_idx = 0
        self.stage_episode_count = 0
        self.stage_success_count = 0
        self.evaluation_window = 100  # Episodes to evaluate before advancing
    
    def _create_curriculum_stages(self) -> List[CurriculumStage]:
        """Create progressive curriculum stages."""
        return [
            # Stage 1: Static target - learn basic homing
            CurriculumStage(
                name="static_target",
                target_speed_range=(0.0, 0.0),
                target_movement_pattern="static",
                target_agility=0.0,
                episode_length=300,
                success_threshold=25,
                min_success_rate=0.9
            ),
            
            # Stage 2: Slow linear movement
            CurriculumStage(
                name="slow_linear",
                target_speed_range=(1.0, 2.0),
                target_movement_pattern="linear",
                target_agility=0.05,
                episode_length=400,
                success_threshold=20,
                min_success_rate=0.8
            ),
            
            # Stage 3: Medium speed with direction changes
            CurriculumStage(
                name="medium_agile",
                target_speed_range=(2.0, 3.5),
                target_movement_pattern="linear",
                target_agility=0.15,
                episode_length=450,
                success_threshold=18,
                min_success_rate=0.75
            ),
            
            # Stage 4: Fast targets
            CurriculumStage(
                name="fast_linear",
                target_speed_range=(3.0, 5.0),
                target_movement_pattern="linear",
                target_agility=0.25,
                episode_length=500,
                success_threshold=15,
                min_success_rate=0.7
            ),
            
            # Stage 5: Circular movement
            CurriculumStage(
                name="circular_movement",
                target_speed_range=(2.0, 4.0),
                target_movement_pattern="circular",
                target_agility=0.8,
                episode_length=500,
                success_threshold=15,
                min_success_rate=0.65
            ),
            
            # Stage 6: Evasive targets (final challenge)
            CurriculumStage(
                name="evasive_target",
                target_speed_range=(3.0, 5.0),
                target_movement_pattern="evasive",
                target_agility=0.6,
                episode_length=600,
                success_threshold=15,
                min_success_rate=0.6
            )
        ]
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    def record_episode_result(self, success: bool) -> bool:
        """
        Record episode result and check if should advance to next stage.
        
        Returns:
            True if advanced to next stage, False otherwise
        """
        self.stage_episode_count += 1
        if success:
            self.stage_success_count += 1
        
        # Check if ready to evaluate advancement
        if self.stage_episode_count >= self.evaluation_window:
            success_rate = self.stage_success_count / self.stage_episode_count
            current_stage = self.get_current_stage()
            
            if success_rate >= current_stage.min_success_rate:
                # Advance to next stage
                if self.current_stage_idx < len(self.stages) - 1:
                    self.current_stage_idx += 1
                    self.stage_episode_count = 0
                    self.stage_success_count = 0
                    logging.info(f"🎓 Advanced to curriculum stage: {self.get_current_stage().name}")
                    return True
                else:
                    logging.info("🏆 Completed all curriculum stages!")
            else:
                # Reset counters to continue current stage
                self.stage_episode_count = 0
                self.stage_success_count = 0
                logging.info(f"📚 Continuing stage '{current_stage.name}' - success rate: {success_rate:.2%}")
        
        return False
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get curriculum progress information."""
        current_stage = self.get_current_stage()
        success_rate = (self.stage_success_count / max(1, self.stage_episode_count))
        
        return {
            "current_stage": current_stage.name,
            "stage_index": self.current_stage_idx + 1,
            "total_stages": len(self.stages),
            "episodes_in_stage": self.stage_episode_count,
            "success_rate": success_rate,
            "target_success_rate": current_stage.min_success_rate,
            "ready_for_next": (success_rate >= current_stage.min_success_rate and 
                             self.stage_episode_count >= self.evaluation_window)
        }