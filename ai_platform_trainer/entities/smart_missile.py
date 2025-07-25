"""
Smart Missile with AI homing capabilities.

This module provides an enhanced missile that can use either traditional neural networks
or reinforcement learning models for intelligent homing behavior.
"""
import math
import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple

try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

from ai_platform_trainer.entities.missile import Missile
from ai_platform_trainer.ai.models.missile_model import MissileModel
from ai_platform_trainer.core.screen_context import ScreenContext


class SmartMissile(Missile):
    """
    Enhanced missile with AI-powered homing capabilities.
    
    Can use either traditional neural networks or RL models for guidance.
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        target_x: float = 0.0,
        target_y: float = 0.0,
        speed: float = 8.0,
        vx: float = 8.0,
        vy: float = 0.0,
        birth_time: int = 0,
        lifespan: int = 5000,  # Reduced lifespan to prevent endless circling
        ai_model: Optional[MissileModel] = None,
        rl_model: Optional = None,
        use_rl: bool = False
    ):
        super().__init__(x, y, speed, vx, vy, birth_time, lifespan)
        
        # AI components
        self.ai_model = ai_model
        self.rl_model = rl_model
        self.use_rl = use_rl and STABLE_BASELINES_AVAILABLE and rl_model is not None
        
        # Target tracking
        self.target_pos = {"x": target_x, "y": target_y}
        self.last_target_pos = {"x": target_x, "y": target_y}
        
        # Homing parameters
        self.max_turn_rate = 12.0  # Increased turn rate for better tracking
        self.prediction_strength = 0.5  # Increased prediction for better interception
        
        # Performance tracking
        self.frames_alive = 0
        self.distance_to_target_history = []
        
        logging.info(f"SmartMissile created with {'RL' if self.use_rl else 'Neural Network' if self.ai_model else 'Basic'} AI")
    
    def update_with_ai(
        self, 
        player_pos: Dict[str, float], 
        target_pos: Dict[str, float],
        shared_input_tensor: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update missile trajectory using AI guidance.
        
        Args:
            player_pos: Player position for context
            target_pos: Current target position
            shared_input_tensor: Pre-allocated tensor for efficiency
        """
        if not target_pos:
            logging.warning("update_with_ai called with no target_pos")
            return
            
        # Update target tracking
        self.last_target_pos = self.target_pos.copy()
        self.target_pos = target_pos.copy()
        self.frames_alive += 1
        
        # Calculate current distance for performance tracking
        current_distance = self._calculate_distance_to_target()
        self.distance_to_target_history.append(current_distance)
        
        if self.use_rl and self.rl_model:
            self._update_with_rl(player_pos, target_pos)
        elif self.ai_model:
            self._update_with_neural_network(player_pos, target_pos, shared_input_tensor)
        else:
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)
        
        # Actually update the position after AI calculations
        super().update()
    
    def _update_with_rl(self, player_pos: Dict[str, float], target_pos: Dict[str, float]) -> None:
        """Update using reinforcement learning model."""
        try:
            # Calculate target velocity
            target_vx = target_pos["x"] - self.last_target_pos["x"]
            target_vy = target_pos["y"] - self.last_target_pos["y"]
            
            # Create observation for RL model
            observation = self._create_rl_observation(target_pos, target_vx, target_vy)
            
            # Get action from RL model
            action, _ = self.rl_model.predict(observation, deterministic=True)
            turn_rate = action[0] * self.max_turn_rate  # Scale to turn rate
            
            # Apply the turn
            self._apply_turn(turn_rate)
            
        except Exception as e:
            logging.error(f"Error in RL missile guidance: {e}")
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)
    
    def _update_with_neural_network(
        self, 
        player_pos: Dict[str, float], 
        target_pos: Dict[str, float],
        shared_input_tensor: Optional[torch.Tensor] = None
    ) -> None:
        """Update using traditional neural network model."""
        try:
            # Get screen context for normalization
            screen_context = ScreenContext.get_instance()
            
            # Create normalized observation
            observation = screen_context.create_missile_observation(
                player_pos, target_pos, self.pos, {"x": self.vx, "y": self.vy}
            )
            
            current_angle = math.atan2(self.vy, self.vx)
            
            # Prepare input tensor with normalized values
            if shared_input_tensor is not None:
                input_tensor = shared_input_tensor
                input_tensor[0] = torch.tensor([
                    observation["player_x"], observation["player_y"],
                    observation["target_x"], observation["target_y"],
                    observation["missile_x"], observation["missile_y"],
                    current_angle, observation["distance_to_target"], 0.0
                ])
            else:
                input_tensor = torch.tensor([[
                    observation["player_x"], observation["player_y"],
                    observation["target_x"], observation["target_y"],
                    observation["missile_x"], observation["missile_y"],
                    current_angle, observation["distance_to_target"], 0.0
                ]], dtype=torch.float32)
            
            # Get prediction from neural network
            with torch.no_grad():
                turn_rate = self.ai_model(input_tensor).item()
            
            # Apply turn rate limits
            turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, turn_rate))
            self._apply_turn(turn_rate)
            
        except Exception as e:
            logging.error(f"Error in neural network missile guidance: {e}")
            # Fallback to basic homing
            self._update_with_basic_homing(target_pos)
    
    def _update_with_basic_homing(self, target_pos: Dict[str, float]) -> None:
        """Fallback basic homing behavior."""
        # Predict where target will be
        target_vx = target_pos["x"] - self.last_target_pos["x"]
        target_vy = target_pos["y"] - self.last_target_pos["y"]
        
        predicted_x = target_pos["x"] + target_vx * self.prediction_strength
        predicted_y = target_pos["y"] + target_vy * self.prediction_strength
        
        # Calculate desired angle
        desired_angle = math.atan2(
            predicted_y - self.pos["y"],
            predicted_x - self.pos["x"]
        )
        
        current_angle = math.atan2(self.vy, self.vx)
        
        # Calculate turn needed
        angle_diff = desired_angle - current_angle
        
        # Normalize angle difference
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Convert to degrees and limit turn rate
        turn_rate = math.degrees(angle_diff)
        turn_rate = max(-self.max_turn_rate, min(self.max_turn_rate, turn_rate))
        
        self._apply_turn(turn_rate)
    
    def _apply_turn(self, turn_rate_degrees: float) -> None:
        """Apply a turn to the missile."""
        current_angle = math.atan2(self.vy, self.vx)
        new_angle = current_angle + math.radians(turn_rate_degrees)
        
        # Update velocity components
        self.vx = self.speed * math.cos(new_angle)
        self.vy = self.speed * math.sin(new_angle)
        
        # Update direction for rendering
        self.direction = (math.cos(new_angle), math.sin(new_angle))
    
    def _create_rl_observation(
        self, 
        target_pos: Dict[str, float], 
        target_vx: float, 
        target_vy: float
    ) -> np.ndarray:
        """Create observation vector for RL model using ScreenContext."""
        # Use ScreenContext for resolution-independent observations
        screen_context = ScreenContext.get_instance()
        
        # Create normalized observation
        observation = screen_context.create_missile_observation(
            {"x": 0, "y": 0},  # Player pos not needed for this observation
            target_pos, 
            self.pos, 
            {"x": self.vx, "y": self.vy}
        )
        
        # Normalize target velocities
        target_vx_norm = target_vx / 5.0
        target_vy_norm = target_vy / 5.0
        
        # Calculate angle to target
        angle_to_target = math.atan2(
            target_pos["y"] - self.pos["y"],
            target_pos["x"] - self.pos["x"]
        )
        angle_norm = angle_to_target / math.pi
        
        return np.array([
            observation["missile_x"], observation["missile_y"], 
            observation["velocity_x"], observation["velocity_y"],
            observation["target_x"], observation["target_y"], 
            target_vx_norm, target_vy_norm,
            observation["distance_to_target"], angle_norm
        ], dtype=np.float32)
    
    def _calculate_distance_to_target(self) -> float:
        """Calculate current distance to target."""
        dx = self.pos["x"] - self.target_pos["x"]
        dy = self.pos["y"] - self.target_pos["y"]
        return math.sqrt(dx * dx + dy * dy)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this missile."""
        if not self.distance_to_target_history:
            return {"avg_distance": 0.0, "min_distance": 0.0, "improvement": 0.0}
        
        avg_distance = sum(self.distance_to_target_history) / len(self.distance_to_target_history)
        min_distance = min(self.distance_to_target_history)
        
        # Calculate improvement (negative means getting closer)
        if len(self.distance_to_target_history) > 1:
            initial_distance = self.distance_to_target_history[0]
            final_distance = self.distance_to_target_history[-1]
            improvement = (initial_distance - final_distance) / initial_distance
        else:
            improvement = 0.0
        
        return {
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "improvement": improvement,
            "frames_alive": self.frames_alive
        }
    
    def is_effective(self) -> bool:
        """Check if missile is performing well."""
        if self.frames_alive < 20:  # Need some data
            return True
        
        stats = self.get_performance_stats()
        
        # Check if missile is stuck in a circle (getting farther from target)
        if stats["improvement"] < -0.3:  # Getting significantly farther
            return False
            
        # Check if missile is getting closer over time
        if len(self.distance_to_target_history) > 30:
            recent_avg = sum(self.distance_to_target_history[-10:]) / 10
            earlier_avg = sum(self.distance_to_target_history[-30:-20]) / 10
            if recent_avg >= earlier_avg:  # Not improving
                return False
        
        return True