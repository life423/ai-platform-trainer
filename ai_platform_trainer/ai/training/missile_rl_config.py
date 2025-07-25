"""
Hyperparameter Configuration for Missile RL Training

This module provides optimized hyperparameter sets for different RL algorithms
based on research and empirical testing for continuous control tasks.
"""
import logging
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    """Configuration for a specific RL algorithm."""
    name: str
    algorithm_class: str
    hyperparameters: Dict[str, Any]
    expected_timesteps: int
    description: str


class MissileRLConfigurations:
    """Optimized configurations for missile RL training."""
    
    @staticmethod
    def get_ppo_baseline() -> AlgorithmConfig:
        """Baseline PPO configuration (current implementation)."""
        return AlgorithmConfig(
            name="PPO_Baseline",
            algorithm_class="PPO",
            hyperparameters={
                "learning_rate": 1e-3,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.3,
                "ent_coef": 0.005,
                "policy_kwargs": {"net_arch": [128, 128]}
            },
            expected_timesteps=100000,
            description="Current PPO implementation - stable but sample inefficient"
        )
    
    @staticmethod
    def get_ppo_optimized() -> AlgorithmConfig:
        """Optimized PPO configuration based on SB3 zoo recommendations."""
        return AlgorithmConfig(
            name="PPO_Optimized",
            algorithm_class="PPO",
            hyperparameters={
                "learning_rate": 3e-4,  # Lower for stability
                "n_steps": 2048,
                "batch_size": 128,  # Larger for better gradients
                "n_epochs": 20,  # More epochs for better learning
                "gamma": 0.99,  # Standard discount
                "gae_lambda": 0.95,  # SB3 zoo recommendation
                "clip_range": 0.2,  # Standard PPO clip range
                "ent_coef": 0.0,  # No entropy bonus for focused policy
                "vf_coef": 0.5,  # Value function coefficient
                "max_grad_norm": 0.5,  # Gradient clipping
                "policy_kwargs": {
                    "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
                    "activation_fn": "torch.nn.Tanh"  # Better for continuous control
                }
            },
            expected_timesteps=80000,
            description="PPO with SB3 zoo hyperparameters for continuous control"
        )
    
    @staticmethod
    def get_sac_default() -> AlgorithmConfig:
        """Default SAC configuration - sample efficient."""
        return AlgorithmConfig(
            name="SAC_Default",
            algorithm_class="SAC",
            hyperparameters={
                "learning_rate": 3e-4,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
                "target_update_interval": 1,
                "policy_kwargs": {
                    "net_arch": [256, 256],
                    "activation_fn": "torch.nn.ReLU"
                }
            },
            expected_timesteps=50000,
            description="Standard SAC configuration - off-policy, sample efficient"
        )
    
    @staticmethod
    def get_sac_aggressive() -> AlgorithmConfig:
        """Aggressive SAC configuration for faster learning."""
        return AlgorithmConfig(
            name="SAC_Aggressive",
            algorithm_class="SAC",
            hyperparameters={
                "learning_rate": 1e-3,  # Higher learning rate
                "buffer_size": 50000,  # Smaller buffer for faster updates
                "learning_starts": 500,  # Start learning earlier
                "batch_size": 512,  # Larger batches
                "tau": 0.01,  # Faster soft updates
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 2,  # More gradient steps
                "ent_coef": "auto",
                "target_update_interval": 1,
                "policy_kwargs": {
                    "net_arch": [512, 512],  # Larger networks
                    "activation_fn": "torch.nn.ReLU"
                }
            },
            expected_timesteps=30000,
            description="Aggressive SAC - faster learning, higher sample efficiency"
        )
    
    @staticmethod
    def get_sac_stable() -> AlgorithmConfig:
        """Conservative SAC configuration for stable learning."""
        return AlgorithmConfig(
            name="SAC_Stable",
            algorithm_class="SAC",
            hyperparameters={
                "learning_rate": 1e-4,  # Lower learning rate
                "buffer_size": 200000,  # Larger buffer for stability
                "learning_starts": 2000,  # More exploration before learning
                "batch_size": 128,  # Smaller batches
                "tau": 0.002,  # Slower soft updates
                "gamma": 0.995,  # Higher discount for long-term thinking
                "train_freq": 4,  # Less frequent updates
                "gradient_steps": 1,
                "ent_coef": "auto",
                "target_update_interval": 2,
                "policy_kwargs": {
                    "net_arch": [256, 256, 128],  # Deeper network
                    "activation_fn": "torch.nn.Tanh"
                }
            },
            expected_timesteps=60000,
            description="Conservative SAC - very stable, longer training"
        )
    
    @staticmethod
    def get_ddpg_config() -> AlgorithmConfig:
        """DDPG configuration for comparison."""
        return AlgorithmConfig(
            name="DDPG",
            algorithm_class="DDPG",
            hyperparameters={
                "learning_rate": 1e-3,
                "buffer_size": 100000,
                "learning_starts": 1000,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "train_freq": 1,
                "gradient_steps": 1,
                "action_noise": "NormalActionNoise",  # Will be handled specially
                "policy_kwargs": {
                    "net_arch": [256, 256],
                    "activation_fn": "torch.nn.ReLU"
                }
            },
            expected_timesteps=70000,
            description="DDPG - deterministic policy, good for precise control"
        )
    
    @staticmethod
    def get_all_configurations() -> List[AlgorithmConfig]:
        """Get all available configurations for comparison."""
        return [
            MissileRLConfigurations.get_ppo_baseline(),
            MissileRLConfigurations.get_ppo_optimized(),
            MissileRLConfigurations.get_sac_default(),
            MissileRLConfigurations.get_sac_aggressive(),
            MissileRLConfigurations.get_sac_stable(),
            MissileRLConfigurations.get_ddpg_config()
        ]
    
    @staticmethod
    def get_recommended_for_task(task_type: str = "missile_guidance") -> List[AlgorithmConfig]:
        """Get recommended configurations for specific task."""
        if task_type == "missile_guidance":
            return [
                MissileRLConfigurations.get_sac_default(),
                MissileRLConfigurations.get_sac_aggressive(),
                MissileRLConfigurations.get_ppo_optimized()
            ]
        else:
            return MissileRLConfigurations.get_all_configurations()


class HyperparameterTuner:
    """Utilities for hyperparameter tuning and optimization."""
    
    @staticmethod
    def create_learning_rate_sweep(base_config: AlgorithmConfig, 
                                  rates: List[float] = None) -> List[AlgorithmConfig]:
        """Create configurations with different learning rates."""
        if rates is None:
            rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        
        configs = []
        for rate in rates:
            config = AlgorithmConfig(
                name=f"{base_config.name}_LR_{rate}",
                algorithm_class=base_config.algorithm_class,
                hyperparameters={**base_config.hyperparameters, "learning_rate": rate},
                expected_timesteps=base_config.expected_timesteps,
                description=f"{base_config.description} (LR: {rate})"
            )
            configs.append(config)
        
        return configs
    
    @staticmethod
    def create_batch_size_sweep(base_config: AlgorithmConfig,
                               sizes: List[int] = None) -> List[AlgorithmConfig]:
        """Create configurations with different batch sizes."""
        if sizes is None:
            if base_config.algorithm_class == "PPO":
                sizes = [32, 64, 128, 256]
            else:  # SAC/DDPG
                sizes = [64, 128, 256, 512]
        
        configs = []
        for size in sizes:
            config = AlgorithmConfig(
                name=f"{base_config.name}_BS_{size}",
                algorithm_class=base_config.algorithm_class,
                hyperparameters={**base_config.hyperparameters, "batch_size": size},
                expected_timesteps=base_config.expected_timesteps,
                description=f"{base_config.description} (Batch Size: {size})"
            )
            configs.append(config)
        
        return configs
    
    @staticmethod
    def create_network_architecture_sweep(base_config: AlgorithmConfig) -> List[AlgorithmConfig]:
        """Create configurations with different network architectures."""
        architectures = [
            [128, 128],
            [256, 256],
            [512, 512],
            [256, 256, 256],
            [512, 256, 128]
        ]
        
        configs = []
        for arch in architectures:
            arch_str = "_".join(map(str, arch))
            if base_config.algorithm_class == "PPO":
                policy_kwargs = {**base_config.hyperparameters.get("policy_kwargs", {})}
                policy_kwargs["net_arch"] = [{"pi": arch, "vf": arch}]
            else:
                policy_kwargs = {**base_config.hyperparameters.get("policy_kwargs", {})}
                policy_kwargs["net_arch"] = arch
            
            config = AlgorithmConfig(
                name=f"{base_config.name}_ARCH_{arch_str}",
                algorithm_class=base_config.algorithm_class,
                hyperparameters={
                    **base_config.hyperparameters, 
                    "policy_kwargs": policy_kwargs
                },
                expected_timesteps=base_config.expected_timesteps,
                description=f"{base_config.description} (Architecture: {arch})"
            )
            configs.append(config)
        
        return configs


def print_configuration_summary():
    """Print a summary of all available configurations."""
    configs = MissileRLConfigurations.get_all_configurations()
    
    print("\\n=== Missile RL Algorithm Configurations ===\\n")
    
    for config in configs:
        print(f"📋 {config.name}")
        print(f"   Algorithm: {config.algorithm_class}")
        print(f"   Expected Training Time: {config.expected_timesteps:,} timesteps")
        print(f"   Description: {config.description}")
        print(f"   Key Hyperparameters:")
        
        important_params = ["learning_rate", "batch_size", "gamma", "tau"]
        for param in important_params:
            if param in config.hyperparameters:
                print(f"     - {param}: {config.hyperparameters[param]}")
        print()
    
    print("🔍 Recommended for missile guidance:")
    recommended = MissileRLConfigurations.get_recommended_for_task("missile_guidance")
    for config in recommended:
        print(f"   • {config.name} - {config.description}")
    print()


if __name__ == "__main__":
    print_configuration_summary()