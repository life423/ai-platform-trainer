"""
Algorithm Comparison and Benchmarking for Missile RL

This module provides comprehensive benchmarking and comparison tools for 
different RL algorithms on the missile guidance task.
"""
import logging
import os
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor
import traceback

try:
    from stable_baselines3 import PPO, SAC, DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.callbacks import BaseCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. Benchmarking disabled.")

from ai_platform_trainer.ai.training.train_missile_rl import MissileRLEnvironment
from ai_platform_trainer.ai.training.train_missile_sac import MissileSACEnvironment
from ai_platform_trainer.ai.training.missile_rl_config import (
    AlgorithmConfig, MissileRLConfigurations, HyperparameterTuner
)


@dataclass
class BenchmarkResult:
    """Results from benchmarking a single configuration."""
    config_name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    
    # Training metrics
    total_training_time: float
    training_timesteps: int
    final_episode_reward: float
    training_success_rate: float
    
    # Evaluation metrics
    eval_avg_reward: float
    eval_hit_rate: float
    eval_avg_steps: float
    eval_consistency: float  # Std dev of rewards
    
    # Sample efficiency metrics
    timesteps_to_convergence: int
    sample_efficiency_score: float
    
    # Additional metrics
    model_size_mb: float
    inference_time_ms: float
    
    # Training curve data
    episode_rewards: List[float]
    episode_lengths: List[float]
    success_rates: List[float]


class BenchmarkTracker(BaseCallback):
    """Callback to track training progress and metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.timestep_rewards = []
        self.convergence_threshold = 0.7  # 70% success rate
        self.convergence_timestep = None
        
    def _on_step(self) -> bool:
        """Called after each step."""
        # Track rewards at each timestep
        if 'rewards' in self.locals:
            self.timestep_rewards.append(float(self.locals['rewards'][0]))
        
        # Check for episode completion
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            if 'episode' in info:
                reward = float(info['episode']['r'])
                length = int(info['episode']['l'])
                
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                
                # Determine if episode was successful (hit target)
                # This is a heuristic based on reward structure
                success = reward > 5.0  # Adjust based on reward function
                self.episode_successes.append(success)
                
                # Check for convergence
                if (len(self.episode_successes) >= 100 and 
                    self.convergence_timestep is None):
                    recent_success_rate = np.mean(self.episode_successes[-100:])
                    if recent_success_rate >= self.convergence_threshold:
                        self.convergence_timestep = self.num_timesteps
        
        return True
    
    def get_success_rate_history(self, window_size: int = 100) -> List[float]:
        """Get rolling success rate history."""
        if len(self.episode_successes) < window_size:
            return []
        
        success_rates = []
        for i in range(window_size, len(self.episode_successes) + 1):
            rate = np.mean(self.episode_successes[i-window_size:i])
            success_rates.append(rate)
        
        return success_rates


class MissileRLBenchmark:
    """Comprehensive benchmarking suite for missile RL algorithms."""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup logging for benchmarks
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(f"{results_dir}/benchmark.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def benchmark_single_config(self, config: AlgorithmConfig, 
                               test_episodes: int = 50) -> BenchmarkResult:
        """Benchmark a single algorithm configuration."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for benchmarking")
        
        self.logger.info(f"Starting benchmark for {config.name}")
        start_time = time.time()
        
        try:
            # Create environment
            if config.algorithm_class == "SAC":
                env = MissileSACEnvironment()
            else:
                env = MissileRLEnvironment()
            
            # Create algorithm
            algorithm_class = self._get_algorithm_class(config.algorithm_class)
            
            # Handle special parameters
            model_kwargs = config.hyperparameters.copy()
            if config.algorithm_class == "DDPG" and "action_noise" in model_kwargs:
                # Create action noise for DDPG
                model_kwargs["action_noise"] = NormalActionNoise(
                    mean=np.zeros(env.action_space.shape), 
                    sigma=0.1 * np.ones(env.action_space.shape)
                )
            
            # Create model
            model = algorithm_class("MlpPolicy", env, verbose=0, **model_kwargs)
            
            # Create tracking callback
            tracker = BenchmarkTracker()
            
            # Train model
            training_start = time.time()
            model.learn(total_timesteps=config.expected_timesteps, callback=tracker)
            training_time = time.time() - training_start
            
            # Save model for evaluation
            model_path = f"{self.results_dir}/{config.name}_model"
            model.save(model_path)
            
            # Calculate model size
            model_size_mb = self._calculate_model_size(model_path + ".zip")
            
            # Evaluate model
            eval_results = self._evaluate_model(model, env, test_episodes)
            
            # Calculate sample efficiency
            timesteps_to_convergence = (tracker.convergence_timestep or 
                                      config.expected_timesteps)
            sample_efficiency = self._calculate_sample_efficiency(
                eval_results["hit_rate"], timesteps_to_convergence
            )
            
            # Measure inference time
            inference_time = self._measure_inference_time(model, env)
            
            # Create result
            result = BenchmarkResult(
                config_name=config.name,
                algorithm=config.algorithm_class,
                hyperparameters=config.hyperparameters,
                total_training_time=training_time,
                training_timesteps=config.expected_timesteps,
                final_episode_reward=np.mean(tracker.episode_rewards[-10:]) if tracker.episode_rewards else 0.0,
                training_success_rate=np.mean(tracker.episode_successes[-100:]) if len(tracker.episode_successes) >= 100 else 0.0,
                eval_avg_reward=eval_results["avg_reward"],
                eval_hit_rate=eval_results["hit_rate"],
                eval_avg_steps=eval_results["avg_steps"],
                eval_consistency=eval_results["consistency"],
                timesteps_to_convergence=timesteps_to_convergence,
                sample_efficiency_score=sample_efficiency,
                model_size_mb=model_size_mb,
                inference_time_ms=inference_time,
                episode_rewards=tracker.episode_rewards,
                episode_lengths=tracker.episode_lengths,
                success_rates=tracker.get_success_rate_history()
            )
            
            self.logger.info(f"Completed benchmark for {config.name} in {time.time() - start_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Benchmark failed for {config.name}: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _get_algorithm_class(self, algorithm_name: str):
        """Get the algorithm class from string name."""
        algorithms = {
            "PPO": PPO,
            "SAC": SAC,
            "DDPG": DDPG
        }
        return algorithms[algorithm_name]
    
    def _calculate_model_size(self, model_path: str) -> float:
        """Calculate model size in MB."""
        if os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            return size_bytes / (1024 * 1024)
        return 0.0
    
    def _evaluate_model(self, model, env, num_episodes: int) -> Dict[str, float]:
        """Evaluate trained model."""
        rewards = []
        steps_list = []
        hits = 0
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if hasattr(env, '_calculate_distance'):
                        threshold = getattr(env, 'success_threshold', 25)
                        if env._calculate_distance() < threshold:
                            hits += 1
                    break
            
            rewards.append(total_reward)
            steps_list.append(steps)
        
        return {
            "avg_reward": np.mean(rewards),
            "hit_rate": hits / num_episodes,
            "avg_steps": np.mean(steps_list),
            "consistency": np.std(rewards)
        }
    
    def _calculate_sample_efficiency(self, hit_rate: float, timesteps: int) -> float:
        """Calculate sample efficiency score."""
        # Higher hit rate with fewer timesteps = higher efficiency
        if timesteps == 0:
            return 0.0
        return hit_rate / (timesteps / 10000)  # Normalize by 10K timesteps
    
    def _measure_inference_time(self, model, env, num_steps: int = 1000) -> float:
        """Measure average inference time in milliseconds."""
        obs, _ = env.reset()
        
        start_time = time.time()
        for _ in range(num_steps):
            action, _ = model.predict(obs, deterministic=True)
            # Just predict, don't step environment for pure inference timing
        end_time = time.time()
        
        avg_time_per_step = (end_time - start_time) / num_steps
        return avg_time_per_step * 1000  # Convert to milliseconds
    
    def run_comparison(self, configs: List[AlgorithmConfig], 
                      test_episodes: int = 50) -> List[BenchmarkResult]:
        """Run comparison across multiple configurations."""
        results = []
        
        self.logger.info(f"Starting comparison of {len(configs)} configurations")
        
        for i, config in enumerate(configs):
            self.logger.info(f"Running benchmark {i+1}/{len(configs)}: {config.name}")
            try:
                result = self.benchmark_single_config(config, test_episodes)
                results.append(result)
                
                # Save intermediate results
                self._save_results(results, f"intermediate_results_{i+1}.json")
                
            except Exception as e:
                self.logger.error(f"Skipping {config.name} due to error: {str(e)}")
                continue
        
        # Save final results
        self._save_results(results, "final_results.json")
        self._generate_report(results)
        
        return results
    
    def _save_results(self, results: List[BenchmarkResult], filename: str):
        """Save results to JSON file."""
        results_data = [asdict(result) for result in results]
        
        with open(f"{self.results_dir}/{filename}", 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
    
    def _generate_report(self, results: List[BenchmarkResult]):
        """Generate comprehensive benchmark report."""
        self.logger.info("Generating benchmark report")
        
        # Sort by sample efficiency
        results_sorted = sorted(results, key=lambda x: x.sample_efficiency_score, reverse=True)
        
        report_path = f"{self.results_dir}/benchmark_report.md"
        with open(report_path, 'w') as f:
            f.write("# Missile RL Algorithm Benchmark Report\\n\\n")
            
            # Summary table
            f.write("## Performance Summary\\n\\n")
            f.write("| Algorithm | Hit Rate | Avg Reward | Sample Efficiency | Training Time | Inference Time |\\n")
            f.write("|-----------|----------|------------|-------------------|---------------|----------------|\\n")
            
            for result in results_sorted:
                f.write(f"| {result.config_name} | {result.eval_hit_rate:.2%} | "
                       f"{result.eval_avg_reward:.2f} | {result.sample_efficiency_score:.3f} | "
                       f"{result.total_training_time:.1f}s | {result.inference_time_ms:.2f}ms |\\n")
            
            f.write("\\n## Detailed Analysis\\n\\n")
            
            # Best performing algorithm
            best = results_sorted[0]
            f.write(f"### 🏆 Best Overall: {best.config_name}\\n")
            f.write(f"- **Hit Rate**: {best.eval_hit_rate:.2%}\\n")
            f.write(f"- **Sample Efficiency**: {best.sample_efficiency_score:.3f}\\n")
            f.write(f"- **Training Time**: {best.total_training_time:.1f} seconds\\n")
            f.write(f"- **Convergence**: {best.timesteps_to_convergence:,} timesteps\\n\\n")
            
            # Algorithm-specific insights
            f.write("### Algorithm Insights\\n\\n")
            
            sac_results = [r for r in results if r.algorithm == "SAC"]
            ppo_results = [r for r in results if r.algorithm == "PPO"]
            
            if sac_results:
                best_sac = max(sac_results, key=lambda x: x.sample_efficiency_score)
                f.write(f"**SAC Performance**: Best hit rate {best_sac.eval_hit_rate:.2%} "
                       f"with {best_sac.timesteps_to_convergence:,} timesteps to convergence\\n")
            
            if ppo_results:
                best_ppo = max(ppo_results, key=lambda x: x.sample_efficiency_score)
                f.write(f"**PPO Performance**: Best hit rate {best_ppo.eval_hit_rate:.2%} "
                       f"with {best_ppo.timesteps_to_convergence:,} timesteps to convergence\\n")
            
            f.write("\\n### Recommendations\\n\\n")
            f.write("Based on the benchmark results:\\n\\n")
            
            if sac_results and ppo_results:
                avg_sac_efficiency = np.mean([r.sample_efficiency_score for r in sac_results])
                avg_ppo_efficiency = np.mean([r.sample_efficiency_score for r in ppo_results])
                
                if avg_sac_efficiency > avg_ppo_efficiency:
                    f.write("- **SAC is more sample efficient** for missile guidance tasks\\n")
                    f.write("- Consider using SAC for faster training and better continuous control\\n")
                else:
                    f.write("- **PPO remains competitive** despite being on-policy\\n")
                    f.write("- PPO may be more stable for some hyperparameter configurations\\n")
        
        self.logger.info(f"Report saved to {report_path}")
    
    def plot_training_curves(self, results: List[BenchmarkResult]):
        """Plot training curves for comparison."""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Missile RL Algorithm Comparison', fontsize=16)
        
        # Episode rewards
        ax1 = axes[0, 0]
        for result in results:
            if result.episode_rewards:
                # Moving average for smoother curves
                window = min(100, len(result.episode_rewards) // 10)
                smoothed = np.convolve(result.episode_rewards, 
                                     np.ones(window)/window, mode='valid')
                ax1.plot(smoothed, label=result.config_name, alpha=0.8)
        ax1.set_title('Episode Rewards (Moving Average)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rates
        ax2 = axes[0, 1]
        for result in results:
            if result.success_rates:
                ax2.plot(result.success_rates, label=result.config_name, alpha=0.8)
        ax2.set_title('Success Rate (Rolling Average)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sample efficiency comparison
        ax3 = axes[1, 0]
        algorithms = [r.config_name for r in results]
        efficiencies = [r.sample_efficiency_score for r in results]
        colors = ['blue' if 'SAC' in alg else 'orange' if 'PPO' in alg else 'green' 
                 for alg in algorithms]
        
        bars = ax3.bar(range(len(algorithms)), efficiencies, color=colors, alpha=0.7)
        ax3.set_title('Sample Efficiency Comparison')
        ax3.set_xlabel('Algorithm')
        ax3.set_ylabel('Efficiency Score')
        ax3.set_xticks(range(len(algorithms)))
        ax3.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Hit rate vs training time
        ax4 = axes[1, 1]
        hit_rates = [r.eval_hit_rate for r in results]
        training_times = [r.total_training_time for r in results]
        
        scatter = ax4.scatter(training_times, hit_rates, 
                            c=['blue' if 'SAC' in r.config_name else 'orange' if 'PPO' in r.config_name else 'green' 
                               for r in results], 
                            s=100, alpha=0.7)
        
        for i, result in enumerate(results):
            ax4.annotate(result.config_name, 
                        (training_times[i], hit_rates[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        ax4.set_title('Hit Rate vs Training Time')
        ax4.set_xlabel('Training Time (seconds)')
        ax4.set_ylabel('Hit Rate')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.results_dir}/comparison_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Plots saved to {plot_path}")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark of all algorithms."""
    if not STABLE_BASELINES_AVAILABLE:
        print("stable_baselines3 is not available. Please install it to run benchmarks.")
        return
    
    # Setup
    benchmark = MissileRLBenchmark("missile_rl_benchmark_results")
    
    # Get configurations to test
    configs = MissileRLConfigurations.get_recommended_for_task("missile_guidance")
    
    print(f"Running comprehensive benchmark with {len(configs)} configurations...")
    
    # Run benchmark
    results = benchmark.run_comparison(configs, test_episodes=30)
    
    # Generate plots
    benchmark.plot_training_curves(results)
    
    print(f"Benchmark completed! Results saved to {benchmark.results_dir}")
    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()