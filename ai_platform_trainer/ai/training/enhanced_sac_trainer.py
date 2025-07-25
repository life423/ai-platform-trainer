"""
Enhanced SAC Trainer with Improved Reward Function and Curriculum Learning

This trainer incorporates all the reward function improvements:
- Time penalty to prevent stalling
- Balanced progress rewards to prevent oscillation
- Curriculum learning
- Enhanced evaluation and behavior analysis
"""
import logging
import os
import numpy as np
from typing import Optional, Dict, Any

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    logging.warning("stable_baselines3 not available. Enhanced training disabled.")

from ai_platform_trainer.ai.training.enhanced_missile_environment import (
    EnhancedMissileEnvironment, CurriculumManager
)
from ai_platform_trainer.ai.training.enhanced_evaluation import BehaviorAnalysisCallback


class EnhancedSACTrainer:
    """
    Enhanced SAC trainer with improved reward function and curriculum learning.
    
    Features:
    - Curriculum learning progression
    - Enhanced reward function with time penalties and oscillation prevention
    - Detailed behavior analysis and recommendations
    - Adaptive training based on performance
    """
    
    def __init__(self, save_path: str = "models/enhanced_missile_sac",
                 use_curriculum: bool = True,
                 analysis_dir: str = "enhanced_analysis/"):
        self.save_path = save_path
        self.use_curriculum = use_curriculum
        self.analysis_dir = analysis_dir
        
        # Create directories
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Initialize curriculum manager
        self.curriculum_manager = CurriculumManager() if use_curriculum else None
        
        # Training state
        self.model = None
        self.env = None
        self.eval_env = None
        self.behavior_callback = None
    
    def create_environments(self):
        """Create training and evaluation environments."""
        if self.use_curriculum:
            # Start with first curriculum stage
            initial_stage = self.curriculum_manager.get_current_stage()
            self.env = EnhancedMissileEnvironment(curriculum_stage=initial_stage)
            self.eval_env = EnhancedMissileEnvironment(curriculum_stage=initial_stage)
        else:
            # Standard environment
            self.env = EnhancedMissileEnvironment()
            self.eval_env = EnhancedMissileEnvironment()
        
        return self.env, self.eval_env
    
    def train(self, total_timesteps: int = 50000, 
              eval_freq: int = 2500,
              save_freq: int = 5000,
              progress_callback=None) -> 'SAC':
        """
        Train enhanced SAC model with curriculum learning and behavior analysis.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Frequency of detailed evaluations
            save_freq: Frequency of model saves
            progress_callback: Optional progress callback
            
        Returns:
            Trained SAC model
        """
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for training")
        
        # Create environments
        env, eval_env = self.create_environments()
        
        # Create enhanced SAC model with optimized hyperparameters
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            target_update_interval=1,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn='torch.nn.ReLU',
            ),
            seed=42
        )
        
        # Create behavior analysis callback
        self.behavior_callback = BehaviorAnalysisCallback(
            eval_env=eval_env,
            curriculum_manager=self.curriculum_manager,
            eval_freq=eval_freq,
            n_eval_episodes=15,
            save_path=self.analysis_dir,
            verbose=1
        )
        
        # Create curriculum callback if using curriculum
        curriculum_callback = None
        if self.use_curriculum:
            curriculum_callback = CurriculumProgressCallback(
                curriculum_manager=self.curriculum_manager,
                env=env,
                eval_env=eval_env
            )
        
        # Combine callbacks
        callbacks = [self.behavior_callback]
        if curriculum_callback:
            callbacks.append(curriculum_callback)
        
        if progress_callback:
            callbacks.append(progress_callback)
        
        callback_list = CallbackList(callbacks)
        
        # Training with enhanced monitoring
        logging.info(f"🚀 Starting enhanced SAC training for {total_timesteps} timesteps")
        if self.use_curriculum:
            initial_stage = self.curriculum_manager.get_current_stage()
            logging.info(f"📚 Starting with curriculum stage: {initial_stage.name}")
        
        logging.info("🎯 Enhanced reward function features:")
        logging.info("   - Time penalty to prevent stalling")
        logging.info("   - Smoothed progress rewards to prevent oscillation")
        logging.info("   - Trajectory efficiency bonuses")
        logging.info("   - Oscillation detection and penalties")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=False
        )
        
        # Save final model
        final_path = f"{self.save_path}_final"
        self.model.save(final_path)
        
        # Generate final training report
        self._generate_training_report()
        
        logging.info(f"✅ Enhanced SAC training completed! Model saved to {final_path}")
        
        return self.model
    
    def _generate_training_report(self):
        """Generate comprehensive training report."""
        if not self.behavior_callback:
            return
        
        # Get training summary
        summary = self.behavior_callback.get_training_summary()
        
        # Create report
        report_path = os.path.join(self.analysis_dir, "training_summary.md")
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced Missile SAC Training Report\\n\\n")
            
            # Training overview
            f.write("## Training Overview\\n\\n")
            f.write(f"- **Total Evaluations**: {summary.get('total_evaluations', 0)}\\n")
            f.write(f"- **Best Hit Rate**: {summary.get('best_hit_rate', 0):.2%}\\n")
            f.write(f"- **Final Hit Rate**: {summary.get('final_hit_rate', 0):.2%}\\n")
            f.write(f"- **Improvement**: {summary.get('improvement_trend', 0):.2%}\\n\\n")
            
            # Curriculum progress
            if self.curriculum_manager:
                progress = self.curriculum_manager.get_progress_info()
                f.write("## Curriculum Learning Progress\\n\\n")
                f.write(f"- **Current Stage**: {progress['current_stage']} ({progress['stage_index']}/{progress['total_stages']})\\n")
                f.write(f"- **Stage Success Rate**: {progress['success_rate']:.2%}\\n")
                f.write(f"- **Target Success Rate**: {progress['target_success_rate']:.2%}\\n")
                f.write(f"- **Episodes in Stage**: {progress['episodes_in_stage']}\\n")
                if progress['ready_for_next']:
                    f.write("- **Status**: Ready for next stage 🎓\\n")
                else:
                    f.write("- **Status**: Continuing current stage 📚\\n")
                f.write("\\n")
            
            # Reward function analysis
            f.write("## Enhanced Reward Function Results\\n\\n")
            if summary.get('best_hit_rate', 0) > 0.8:
                f.write("✅ **Excellent Performance**: High hit rate achieved\\n")
            elif summary.get('best_hit_rate', 0) > 0.6:
                f.write("✅ **Good Performance**: Solid hit rate with room for improvement\\n")
            else:
                f.write("⚠️ **Needs Improvement**: Consider adjusting reward function\\n")
            
            f.write("\\n### Key Improvements:\\n")
            f.write("- **Time Penalty**: Encourages faster interception\\n")
            f.write("- **Oscillation Prevention**: Smoothed progress rewards\\n")
            f.write("- **Trajectory Efficiency**: Rewards direct paths to target\\n")
            f.write("- **Behavioral Analysis**: Detailed failure mode detection\\n\\n")
            
            # Latest recommendations
            if self.behavior_callback.evaluation_reports:
                latest_report = self.behavior_callback.evaluation_reports[-1]
                if latest_report.recommendations:
                    f.write("## Latest Recommendations\\n\\n")
                    for rec in latest_report.recommendations:
                        f.write(f"- {rec}\\n")
                    f.write("\\n")
            
            f.write("## Files Generated\\n\\n")
            f.write("- `evaluation_report_*.json`: Detailed evaluation data\\n")
            f.write("- `trajectory_analysis_*.png`: Trajectory visualizations\\n")
            f.write("- Training logs and model checkpoints\\n")
        
        logging.info(f"📊 Training report saved to {report_path}")
    
    def test_model(self, model_path: str, num_episodes: int = 20,
                   test_all_stages: bool = True) -> Dict[str, Any]:
        """
        Test trained model across different difficulty levels.
        
        Args:
            model_path: Path to saved model
            num_episodes: Episodes per test
            test_all_stages: Whether to test on all curriculum stages
            
        Returns:
            Test results dictionary
        """
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for testing")
        
        # Load model
        model = SAC.load(model_path)
        
        results = {}
        
        if test_all_stages and self.curriculum_manager:
            # Test on all curriculum stages
            for stage in self.curriculum_manager.stages:
                test_env = EnhancedMissileEnvironment(curriculum_stage=stage)
                stage_results = self._test_on_stage(model, test_env, num_episodes, stage.name)
                results[stage.name] = stage_results
        else:
            # Test on current/default stage
            test_env = EnhancedMissileEnvironment()
            results["standard"] = self._test_on_stage(model, test_env, num_episodes, "standard")
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def _test_on_stage(self, model, env, num_episodes: int, stage_name: str) -> Dict[str, Any]:
        """Test model on a specific stage."""
        rewards = []
        hit_counts = 0
        episode_lengths = []
        trajectory_efficiencies = []
        oscillation_counts = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    if info["success"]:
                        hit_counts += 1
                    
                    rewards.append(total_reward)
                    episode_lengths.append(steps)
                    trajectory_efficiencies.append(info["trajectory_efficiency"])
                    oscillation_counts.append(info["oscillation_count"])
                    break
        
        return {
            "avg_reward": np.mean(rewards),
            "hit_rate": hit_counts / num_episodes,
            "avg_episode_length": np.mean(episode_lengths),
            "avg_trajectory_efficiency": np.mean(trajectory_efficiencies),
            "avg_oscillation_count": np.mean(oscillation_counts),
            "reward_std": np.std(rewards)
        }
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        logging.info("🎯 Enhanced SAC Test Results:")
        
        for stage_name, metrics in results.items():
            logging.info(f"\\n📊 Stage: {stage_name}")
            logging.info(f"   Hit Rate: {metrics['hit_rate']:.2%}")
            logging.info(f"   Avg Reward: {metrics['avg_reward']:.2f}")
            logging.info(f"   Avg Episode Length: {metrics['avg_episode_length']:.1f}")
            logging.info(f"   Trajectory Efficiency: {metrics['avg_trajectory_efficiency']:.2f}")
            logging.info(f"   Oscillation Count: {metrics['avg_oscillation_count']:.1f}")


class CurriculumProgressCallback:
    """Callback to handle curriculum progression during training."""
    
    def __init__(self, curriculum_manager: CurriculumManager,
                 env: EnhancedMissileEnvironment,
                 eval_env: EnhancedMissileEnvironment):
        self.curriculum_manager = curriculum_manager
        self.env = env
        self.eval_env = eval_env
    
    def __call__(self, timestep: int, total_timesteps: int):
        """Called during training to check curriculum advancement."""
        # This would be integrated with the behavior analysis callback
        # to update environments when curriculum advances
        pass


def run_enhanced_training():
    """Run enhanced missile training with all improvements."""
    if not STABLE_BASELINES_AVAILABLE:
        print("stable_baselines3 is not available. Please install it for enhanced training.")
        return
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create enhanced trainer
    trainer = EnhancedSACTrainer(
        save_path="models/enhanced_missile_sac",
        use_curriculum=True,
        analysis_dir="enhanced_missile_analysis/"
    )
    
    print("🚀 Starting Enhanced Missile SAC Training")
    print("   Features:")
    print("   - Time penalty to prevent stalling")
    print("   - Oscillation prevention")
    print("   - Curriculum learning (6 progressive stages)")
    print("   - Detailed behavior analysis")
    print("   - Automatic reward tuning recommendations")
    print()
    
    # Train model
    model = trainer.train(total_timesteps=30000, eval_freq=2000)
    
    # Test on all curriculum stages
    results = trainer.test_model("models/enhanced_missile_sac_final.zip", num_episodes=15)
    
    print("\\n✅ Enhanced training completed!")
    print("Check 'enhanced_missile_analysis/' for detailed reports and trajectory visualizations")
    
    return model, results


if __name__ == "__main__":
    run_enhanced_training()