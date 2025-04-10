"""
GPU-accelerated reinforcement learning environment for missile avoidance training.

This module integrates with the native C++ CUDA implementation to provide
high-performance training environments for improving enemy AI missile avoidance.
"""
import os
import sys
import gym
import numpy as np
import torch
from typing import Dict, Any, Tuple, List, Optional, Union

# Add the path to the compiled extension
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import will be available after the extension is built
try:
    import gpu_environment as native_env
    HAS_GPU_ENV = True
except ImportError:
    print("Native GPU environment not found. You may need to build it first.")
    print("Run 'cd ai_platform_trainer/cpp && python setup.py build_ext --inplace'")
    HAS_GPU_ENV = False
    # Create a dummy for IDE assistance
    class native_env:
        class Environment:
            pass
        class EnvironmentConfig:
            pass

# Define a gym-compatible wrapper
class GPUGameEnv(gym.Env):
    """
    A gym environment that wraps the CUDA-accelerated game environment.
    
    This class provides a standard gym interface for training RL agents
    with a focus on missile avoidance behavior.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None):
        """
        Initialize the GPU game environment.
        
        Args:
            config: Configuration parameters for the environment
        """
        if not HAS_GPU_ENV:
            raise ImportError("GPU environment extension not available")
        
        # Create config object if none provided
        if config is None:
            config = self._default_config()
            
        # Create the native environment
        self.env = native_env.Environment(config)
        
        # Set up gym spaces
        obs_shape = self.env.get_observation_shape()
        action_shape = self.env.get_action_shape()
        
        self.observation_space = gym.spaces.Box(
            low=-float('inf'),
            high=float('inf'),
            shape=obs_shape
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=action_shape
        )
        
        # State variables
        self.current_obs = None
        self.steps = 0
        self.episode_reward = 0.0
        self.debug_data = {}
        
    def reset(self, seed=None):
        """
        Reset the environment and return the initial observation.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        seed = seed or np.random.randint(0, 2**31 - 1)
        self.current_obs = self.env.reset(seed)
        self.steps = 0
        self.episode_reward = 0.0
        self.debug_data = {}
        return self.current_obs
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert action to numpy array if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Take step in native environment
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Update state variables
        self.current_obs = obs
        self.steps += 1
        self.episode_reward += reward
        info['episode_reward'] = self.episode_reward
        info['steps'] = self.steps
        
        # Get debug data
        self.debug_data = self.env.get_debug_data()
        info['debug'] = self.debug_data
        
        return obs, reward, done, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented for headless).
        
        Args:
            mode: Rendering mode
            
        Returns:
            Rendered frame (rgb_array) or None (human)
        """
        if mode == 'rgb_array':
            # Return an RGB frame from debug data
            if 'render_frame' in self.debug_data:
                return self.debug_data['render_frame']
            # Fallback to empty frame
            return np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Human mode has no rendering in headless environment
        return None
    
    def close(self):
        """
        Clean up environment resources.
        """
        # Native environment handles cleanup automatically
        pass
    
    def seed(self, seed=None):
        """
        Set random seed.
        
        Args:
            seed: Random seed
            
        Returns:
            List of seeds used
        """
        # Will be used in reset()
        self._seed = seed or np.random.randint(0, 2**31 - 1)
        return [self._seed]
    
    @staticmethod
    def _default_config():
        """
        Create default environment configuration.
        
        Returns:
            Default configuration object
        """
        config = native_env.EnvironmentConfig()
        config.screen_width = 800
        config.screen_height = 600
        config.max_missiles = 5
        config.player_size = 50.0
        config.enemy_size = 50.0
        config.missile_size = 10.0
        config.player_speed = 5.0
        config.enemy_speed = 5.0
        config.missile_speed = 5.0
        config.missile_lifespan = 10000.0  # 10 seconds
        config.respawn_delay = 500.0
        config.max_steps = 1000
        config.enable_missile_avoidance = True
        config.missile_prediction_steps = 30
        config.missile_detection_radius = 250.0
        config.missile_danger_radius = 150.0
        config.evasion_strength = 2.5
        return config


# Vectorized environment for parallel training
class VectorizedGPUEnv:
    """
    Vectorized environment for efficient parallel training.
    
    This class allows running multiple environments in parallel for
    improved training throughput.
    """
    
    def __init__(self, num_envs=4, config=None):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            config: Configuration for each environment
        """
        if not HAS_GPU_ENV:
            raise ImportError("GPU environment extension not available")
        
        # Create config object if none provided
        if config is None:
            config = native_env.EnvironmentConfig()
        
        self.num_envs = num_envs
        self.env = native_env.Environment(config)
        
        # Set up observation and action shapes
        self.observation_shape = self.env.get_observation_shape()
        self.action_shape = self.env.get_action_shape()
    
    def reset(self, seeds=None):
        """
        Reset all environments.
        
        Args:
            seeds: Optional list of seeds for each environment
            
        Returns:
            List of initial observations
        """
        if seeds is None:
            seeds = np.array([])
        
        return self.env.batch_reset(self.num_envs, seeds)
    
    def step(self, actions):
        """
        Take a step in all environments.
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        results = self.env.batch_step(actions)
        
        # Unzip results
        observations = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for obs, reward, done, truncated, info in results:
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
        
        return observations, rewards, dones, truncateds, infos


def make_env(config=None):
    """
    Create a GPU-accelerated game environment.
    
    Args:
        config: Optional environment configuration
        
    Returns:
        A gymnasium-compatible environment
    """
    return GPUGameEnv(config)


def create_vectorized_env(num_envs=4, config=None):
    """
    Create a vectorized GPU-accelerated environment.
    
    Args:
        num_envs: Number of parallel environments
        config: Optional environment configuration
        
    Returns:
        A vectorized environment for parallel training
    """
    return VectorizedGPUEnv(num_envs, config)


if __name__ == "__main__":
    """Example usage of the GPU environment"""
    if not HAS_GPU_ENV:
        print("Native GPU environment not available. Please build it first.")
        sys.exit(1)
    
    # Create and test a single environment
    env = make_env()
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward = {reward}, done = {done}")
        
        if done:
            print("Episode finished!")
            obs = env.reset()
    
    env.close()
    
    # Create and test a vectorized environment
    vec_env = create_vectorized_env(num_envs=2)
    obs_list = vec_env.reset()
    print(f"Vectorized environment initialized with {len(obs_list)} parallel environments")
    
    for i in range(5):
        # Create random actions for all environments
        actions = [np.random.uniform(-1, 1, size=2) for _ in range(2)]
        observations, rewards, dones, truncateds, infos = vec_env.step(actions)
        print(f"Vectorized step {i}: rewards = {rewards}")
