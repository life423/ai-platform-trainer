"""
Enhanced Evaluation and Analysis for Missile RL Training

This module provides detailed behavior analysis and evaluation callbacks
to help identify and fix reward function issues.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from ai_platform_trainer.ai.training.enhanced_missile_environment import (
    EnhancedMissileEnvironment, CurriculumManager
)


@dataclass
class BehaviorAnalysis:
    """Analysis of missile behavior patterns."""
    avg_trajectory_efficiency: float
    avg_oscillation_count: float
    avg_episode_length: float
    hit_rate: float
    near_miss_rate: float  # Close but not hit
    timeout_rate: float
    avg_time_to_hit: float
    reward_variance: float
    common_failure_modes: List[str]


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    timestep: int
    stage_name: str
    behavior_analysis: BehaviorAnalysis
    reward_breakdown: Dict[str, float]
    trajectory_samples: List[Dict[str, Any]]
    recommendations: List[str]


class BehaviorAnalysisCallback(BaseCallback):
    """
    Enhanced callback that analyzes missile behavior and provides insights
    for reward function tuning.
    """
    
    def __init__(self, eval_env: EnhancedMissileEnvironment,
                 curriculum_manager: Optional[CurriculumManager] = None,
                 eval_freq: int = 5000,
                 n_eval_episodes: int = 20,
                 save_path: str = "evaluation_analysis/",
                 verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.curriculum_manager = curriculum_manager
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = save_path
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Tracking variables
        self.evaluation_reports = []
        self.best_hit_rate = 0.0
        self.episodes_since_improvement = 0
        
        # Behavior pattern detection
        self.trajectory_data = []
        self.reward_component_tracking = []
    
    def _on_step(self) -> bool:
        """Called after each step."""
        # Regular evaluation
        if self.n_calls % self.eval_freq == 0:
            self._run_comprehensive_evaluation()
        
        return True
    
    def _run_comprehensive_evaluation(self):
        """Run comprehensive evaluation with behavior analysis."""
        logging.info(f"Running comprehensive evaluation at timestep {self.num_timesteps}")
        
        # Run evaluation episodes
        episode_data = []
        trajectory_samples = []
        
        for episode in range(self.n_eval_episodes):
            episode_info = self._run_single_evaluation_episode()
            episode_data.append(episode_info)
            
            # Collect detailed trajectory samples for first few episodes
            if episode < 3:
                trajectory_samples.append(episode_info)
        
        # Analyze behavior patterns
        behavior_analysis = self._analyze_behavior_patterns(episode_data)
        
        # Create evaluation report
        current_stage = self.curriculum_manager.get_current_stage().name if self.curriculum_manager else "standard"
        report = EvaluationReport(
            timestep=self.num_timesteps,
            stage_name=current_stage,
            behavior_analysis=behavior_analysis,
            reward_breakdown=self._analyze_reward_components(episode_data),
            trajectory_samples=trajectory_samples,
            recommendations=self._generate_recommendations(behavior_analysis)
        )
        
        self.evaluation_reports.append(report)
        
        # Save detailed analysis
        self._save_evaluation_report(report)
        
        # Check for improvement and curriculum advancement
        self._check_improvement_and_curriculum(behavior_analysis)
        
        # Log summary
        self._log_evaluation_summary(report)
    
    def _run_single_evaluation_episode(self) -> Dict[str, Any]:
        """Run a single evaluation episode with detailed tracking."""
        obs, _ = self.eval_env.reset()
        
        episode_data = {
            "trajectory": [],
            "actions": [],
            "rewards": [],
            "reward_components": [],
            "distances": [],
            "success": False,
            "episode_length": 0,
            "final_distance": 0,
            "trajectory_efficiency": 0,
            "oscillation_count": 0,
            "failure_mode": "unknown"
        }
        
        total_reward = 0
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Safety limit
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Store pre-step data
            episode_data["trajectory"].append({
                "missile_x": self.eval_env.missile.pos["x"],
                "missile_y": self.eval_env.missile.pos["y"],
                "target_x": self.eval_env.target["x"],
                "target_y": self.eval_env.target["y"]
            })
            episode_data["actions"].append(float(action[0]))
            
            # Execute step
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            
            # Store post-step data
            episode_data["rewards"].append(reward)
            episode_data["distances"].append(info["distance"])
            
            if done:
                episode_data["success"] = info["success"]
                episode_data["final_distance"] = info["distance"]
                episode_data["trajectory_efficiency"] = info["trajectory_efficiency"]
                episode_data["oscillation_count"] = info["oscillation_count"]
                episode_data["episode_length"] = step_count
                
                # Determine failure mode
                if not info["success"]:
                    if step_count >= 1000:
                        episode_data["failure_mode"] = "timeout"
                    elif info["distance"] < 30:
                        episode_data["failure_mode"] = "near_miss"
                    elif self.eval_env.missile.pos["x"] < 0 or self.eval_env.missile.pos["x"] > self.eval_env.screen_width:
                        episode_data["failure_mode"] = "out_of_bounds"
                    elif info["oscillation_count"] > 10:
                        episode_data["failure_mode"] = "oscillation"
                    else:
                        episode_data["failure_mode"] = "poor_guidance"
                else:
                    episode_data["failure_mode"] = "success"
        
        return episode_data
    
    def _analyze_behavior_patterns(self, episode_data: List[Dict[str, Any]]) -> BehaviorAnalysis:
        """Analyze behavior patterns across episodes."""
        if not episode_data:
            return BehaviorAnalysis(0, 0, 0, 0, 0, 0, 0, 0, [])
        
        # Calculate metrics
        trajectory_efficiencies = [ep["trajectory_efficiency"] for ep in episode_data]
        oscillation_counts = [ep["oscillation_count"] for ep in episode_data]
        episode_lengths = [ep["episode_length"] for ep in episode_data]
        total_rewards = [sum(ep["rewards"]) for ep in episode_data]
        
        successes = [ep["success"] for ep in episode_data]
        hit_rate = np.mean(successes)
        
        # Near miss analysis (close but didn't hit)
        near_misses = [ep["final_distance"] < 30 and not ep["success"] for ep in episode_data]
        near_miss_rate = np.mean(near_misses)
        
        # Timeout analysis
        timeouts = [ep["episode_length"] >= 500 for ep in episode_data]
        timeout_rate = np.mean(timeouts)
        
        # Time to hit for successful episodes
        successful_episodes = [ep for ep in episode_data if ep["success"]]
        avg_time_to_hit = np.mean([ep["episode_length"] for ep in successful_episodes]) if successful_episodes else 0
        
        # Identify common failure modes
        failure_modes = [ep["failure_mode"] for ep in episode_data if not ep["success"]]
        failure_mode_counts = {}
        for mode in failure_modes:
            failure_mode_counts[mode] = failure_mode_counts.get(mode, 0) + 1
        
        common_failure_modes = sorted(failure_mode_counts.items(), key=lambda x: x[1], reverse=True)
        common_failure_modes = [mode for mode, count in common_failure_modes[:3]]
        
        return BehaviorAnalysis(
            avg_trajectory_efficiency=np.mean(trajectory_efficiencies),
            avg_oscillation_count=np.mean(oscillation_counts),
            avg_episode_length=np.mean(episode_lengths),
            hit_rate=hit_rate,
            near_miss_rate=near_miss_rate,
            timeout_rate=timeout_rate,
            avg_time_to_hit=avg_time_to_hit,
            reward_variance=np.std(total_rewards),
            common_failure_modes=common_failure_modes
        )
    
    def _analyze_reward_components(self, episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze which reward components are most impactful."""
        if not episode_data:
            return {}
        
        # This is a simplified analysis - in practice you'd track individual reward components
        all_rewards = []
        for episode in episode_data:
            all_rewards.extend(episode["rewards"])
        
        return {
            "avg_step_reward": np.mean(all_rewards),
            "reward_std": np.std(all_rewards),
            "min_reward": np.min(all_rewards),
            "max_reward": np.max(all_rewards),
            "positive_reward_ratio": np.mean([r > 0 for r in all_rewards])
        }
    
    def _generate_recommendations(self, analysis: BehaviorAnalysis) -> List[str]:
        """Generate recommendations based on behavior analysis."""
        recommendations = []
        
        # Hit rate analysis
        if analysis.hit_rate < 0.5:
            recommendations.append("🎯 Low hit rate - consider increasing hit reward or decreasing success threshold")
        
        # Near miss analysis
        if analysis.near_miss_rate > 0.3:
            recommendations.append("🎯 High near-miss rate - missile gets close but doesn't hit. Consider tighter success threshold or stronger final approach reward")
        
        # Timeout analysis
        if analysis.timeout_rate > 0.4:
            recommendations.append("⏱️ High timeout rate - increase time penalty or reduce episode length")
        
        # Oscillation analysis
        if analysis.avg_oscillation_count > 8:
            recommendations.append("🌊 High oscillation - increase oscillation penalty or smooth progress rewards more")
        
        # Trajectory efficiency analysis
        if analysis.avg_trajectory_efficiency < 0.6:
            recommendations.append("📈 Low trajectory efficiency - increase efficiency bonus or add more directional guidance")
        
        # Episode length analysis
        if analysis.avg_time_to_hit > 400:
            recommendations.append("🏃 Slow interception - increase time penalty or add speed bonus")
        
        # Failure mode analysis
        if "near_miss" in analysis.common_failure_modes:
            recommendations.append("🎯 Common near misses - fine-tune final approach rewards")
        
        if "oscillation" in analysis.common_failure_modes:
            recommendations.append("🌊 Oscillation failures - reduce progress reward weight or increase smoothing")
        
        if "timeout" in analysis.common_failure_modes:
            recommendations.append("⏱️ Timeout failures - increase urgency signals or time penalties")
        
        if not recommendations:
            recommendations.append("✅ Behavior looks good! Consider advancing difficulty or fine-tuning hyperparameters")
        
        return recommendations
    
    def _save_evaluation_report(self, report: EvaluationReport):
        """Save detailed evaluation report."""
        filename = f"evaluation_report_{report.timestep:08d}.json"
        filepath = os.path.join(self.save_path, filename)
        
        # Convert to serializable format
        report_dict = asdict(report)
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Also save trajectory visualization
        self._save_trajectory_plots(report)
    
    def _save_trajectory_plots(self, report: EvaluationReport):
        """Save trajectory visualization plots."""
        if not report.trajectory_samples:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Missile Behavior Analysis - Timestep {report.timestep}', fontsize=14)
        
        # Plot 1: Sample trajectories
        ax1 = axes[0, 0]
        for i, sample in enumerate(report.trajectory_samples):
            trajectory = sample["trajectory"]
            if trajectory:
                missile_x = [point["missile_x"] for point in trajectory]
                missile_y = [point["missile_y"] for point in trajectory]
                target_x = [point["target_x"] for point in trajectory]
                target_y = [point["target_y"] for point in trajectory]
                
                ax1.plot(missile_x, missile_y, alpha=0.7, label=f'Missile {i+1}')
                ax1.plot(target_x, target_y, '--', alpha=0.5, label=f'Target {i+1}')
                
                # Mark start and end
                ax1.plot(missile_x[0], missile_y[0], 'go', markersize=8)  # Start
                ax1.plot(missile_x[-1], missile_y[-1], 'ro', markersize=8)  # End
        
        ax1.set_title('Sample Trajectories')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Action patterns
        ax2 = axes[0, 1]
        for i, sample in enumerate(report.trajectory_samples):
            actions = sample["actions"]
            ax2.plot(actions, alpha=0.7, label=f'Episode {i+1}')
        
        ax2.set_title('Action Patterns')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Turn Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Distance over time
        ax3 = axes[1, 0]
        for i, sample in enumerate(report.trajectory_samples):
            distances = sample["distances"]
            ax3.plot(distances, alpha=0.7, label=f'Episode {i+1}')
        
        ax3.set_title('Distance to Target Over Time')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Distance')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Reward progression
        ax4 = axes[1, 1]
        for i, sample in enumerate(report.trajectory_samples):
            rewards = sample["rewards"]
            cumulative_rewards = np.cumsum(rewards)
            ax4.plot(cumulative_rewards, alpha=0.7, label=f'Episode {i+1}')
        
        ax4.set_title('Cumulative Reward Over Time')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Cumulative Reward')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"trajectory_analysis_{report.timestep:08d}.png"
        plot_filepath = os.path.join(self.save_path, plot_filename)
        plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _check_improvement_and_curriculum(self, analysis: BehaviorAnalysis):
        """Check for improvement and handle curriculum advancement."""
        # Track improvement
        if analysis.hit_rate > self.best_hit_rate:
            self.best_hit_rate = analysis.hit_rate
            self.episodes_since_improvement = 0
            logging.info(f"🎉 New best hit rate: {analysis.hit_rate:.2%}")
        else:
            self.episodes_since_improvement += 1
        
        # Curriculum advancement
        if self.curriculum_manager:
            # Record results for curriculum
            for _ in range(self.n_eval_episodes):
                success = np.random.random() < analysis.hit_rate  # Approximate
                advanced = self.curriculum_manager.record_episode_result(success)
                if advanced:
                    # Update environment to new stage
                    new_stage = self.curriculum_manager.get_current_stage()
                    self.eval_env.curriculum_stage = new_stage
                    logging.info(f"📚 Advanced to new curriculum stage: {new_stage.name}")
                    break
    
    def _log_evaluation_summary(self, report: EvaluationReport):
        """Log evaluation summary."""
        analysis = report.behavior_analysis
        
        logging.info(f"📊 Evaluation Summary (Timestep {report.timestep}):")
        logging.info(f"   Hit Rate: {analysis.hit_rate:.2%}")
        logging.info(f"   Avg Episode Length: {analysis.avg_episode_length:.1f}")
        logging.info(f"   Trajectory Efficiency: {analysis.avg_trajectory_efficiency:.2f}")
        logging.info(f"   Oscillation Count: {analysis.avg_oscillation_count:.1f}")
        logging.info(f"   Near Miss Rate: {analysis.near_miss_rate:.2%}")
        
        if analysis.common_failure_modes:
            logging.info(f"   Common Failures: {', '.join(analysis.common_failure_modes)}")
        
        if report.recommendations:
            logging.info("💡 Recommendations:")
            for rec in report.recommendations:
                logging.info(f"     {rec}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of entire training session."""
        if not self.evaluation_reports:
            return {}
        
        # Track improvement over time
        hit_rates = [report.behavior_analysis.hit_rate for report in self.evaluation_reports]
        timesteps = [report.timestep for report in self.evaluation_reports]
        
        return {
            "total_evaluations": len(self.evaluation_reports),
            "best_hit_rate": max(hit_rates),
            "final_hit_rate": hit_rates[-1],
            "improvement_trend": hit_rates[-1] - hit_rates[0] if len(hit_rates) > 1 else 0,
            "timesteps": timesteps,
            "hit_rates": hit_rates,
            "curriculum_completed": (self.curriculum_manager.current_stage_idx == len(self.curriculum_manager.stages) - 1) if self.curriculum_manager else False
        }