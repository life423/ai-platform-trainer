"""
Training script for missile avoidance using GPU-accelerated environment.

This script trains an enemy AI agent to avoid missiles using reinforcement learning
with the GPU-accelerated environment.
"""
import os
import time
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import torch

# Import our GPU environment wrapper
from gpu_environment import make_env, create_vectorized_env, native_env, HAS_GPU_ENV

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def make_env_fn(env_id, rank, seed=0):
    """
    Create a function that will create an environment with appropriate wrappers.

    Args:
        env_id: Identifier for the environment
        rank: Process rank for seeding
        seed: Base seed for RNG

    Returns:
        Function that creates an environment
    """
    def _init():
        # Create configuration with enhanced missile avoidance settings
        config = native_env.EnvironmentConfig()
        config.screen_width = 800
        config.screen_height = 600
        config.max_missiles = 5
        config.player_size = 50.0
        config.enemy_size = 50.0
        config.missile_size = 10.0
        config.player_speed = 5.0
        config.enemy_speed = 5.0  # Same as player for fair challenge
        config.missile_speed = 5.0  
        config.missile_lifespan = 10000.0  # 10 seconds, longer missile life
        config.respawn_delay = 500.0
        config.max_steps = 1000
        
        # Missile avoidance specific settings
        config.enable_missile_avoidance = True  # Initially enabled for better baseline
        config.missile_prediction_steps = 30    # Look ahead 30 frames
        config.missile_detection_radius = 250.0 # Increased detection radius
        config.missile_danger_radius = 150.0    # Increased danger radius
        config.evasion_strength = 2.5           # Higher evasion strength
        
        env = make_env(config)
        env.seed(seed + rank)
        return env
    return _init


def train_model(
    num_envs=4,
    total_timesteps=1000000,
    save_dir="models/gpu_rl",
    log_dir="logs/gpu_rl",
    save_freq=10000,
):
    """
    Train a PPO model for missile avoidance using the GPU-accelerated environment.

    Args:
        num_envs: Number of parallel environments
        total_timesteps: Total number of timesteps to train for
        save_dir: Directory to save model checkpoints
        log_dir: Directory to save tensorboard logs
        save_freq: Frequency to save checkpoints
    """
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if not HAS_GPU_ENV:
        raise ImportError("GPU environment not available. Build the extension first.")

    # Create a vectorized environment
    if num_envs == 1:
        env = DummyVecEnv([make_env_fn("GPUGameEnv", 0)])
    else:
        env = SubprocVecEnv([make_env_fn("GPUGameEnv", i) for i in range(num_envs)])

    # Normalize observation and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Configure policy network
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[
            dict(
                pi=[128, 128, 64],  # Policy network
                vf=[128, 128, 64]   # Value function network
            )
        ]
    )

    # Create a PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Slightly higher entropy for better exploration
        policy_kwargs=policy_kwargs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Set up checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // num_envs,
        save_path=save_dir,
        name_prefix="missile_avoidance_model",
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    # Record start time
    start_time = time.time()
    
    # Train the model
    try:
        logger.info(f"Starting training with {num_envs} environments for {total_timesteps} timesteps")
        logger.info(f"Using device: {model.device}")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            tb_log_name="ppo_missile_avoidance",
            reset_num_timesteps=True
        )
        
        # Calculate training duration
        duration = time.time() - start_time
        logger.info(f"Training completed in {duration:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(save_dir, "final_model")
        model.save(final_model_path)
        env.save(os.path.join(save_dir, "vec_normalize.pkl"))
        logger.info(f"Final model saved to {final_model_path}")
        
        return model, env
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # Save model on error
        if os.path.exists(save_dir):
            error_model_path = os.path.join(save_dir, "error_model")
            try:
                model.save(error_model_path)
                logger.info(f"Saved model at error to {error_model_path}")
            except Exception:
                logger.error("Failed to save model at error")
        return None, None


def evaluate_model(
    model_path,
    normalize_path=None,
    num_episodes=10
):
    """
    Evaluate a trained model.

    Args:
        model_path: Path to the trained model
        normalize_path: Path to normalization statistics (optional)
        num_episodes: Number of episodes to evaluate
    """
    if not HAS_GPU_ENV:
        raise ImportError("GPU environment not available. Build the extension first.")

    # Create a single environment for evaluation
    env = DummyVecEnv([make_env_fn("GPUGameEnv", 0)])
    
    # Load normalization statistics if available
    if normalize_path and os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        # Don't update normalization statistics during evaluation
        env.training = False
        env.norm_reward = False
    
    # Load the model
    model = PPO.load(model_path, env=env)
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    avoidance_counts = []
    
    logger.info(f"Evaluating model for {num_episodes} episodes")
    
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        avoidances = 0
        
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward.item()
            steps += 1
            
            # Extract info about missile avoidances from info dict
            if len(info) > 0 and 'missile_avoidance' in info[0]:
                avoidances += info[0]['missile_avoidance']
            
            done = done[0]
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        avoidance_counts.append(avoidances)
        
        logger.info(f"Episode {i+1}: reward={episode_reward:.2f}, length={steps}, avoidances={avoidances}")
    
    # Calculate average statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_avoidances = np.mean(avoidance_counts)
    
    logger.info(f"Evaluation complete:")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Average episode length: {avg_length:.2f}")
    logger.info(f"Average missile avoidances: {avg_avoidances:.2f}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'avg_avoidances': avg_avoidances,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'avoidance_counts': avoidance_counts
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train missile avoidance using GPU-accelerated environment")
    parser.add_argument('--num-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total timesteps to train for')
    parser.add_argument('--save-dir', type=str, default='models/gpu_rl', help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/gpu_rl', help='Directory to save tensorboard logs')
    parser.add_argument('--save-freq', type=int, default=10000, help='Frequency to save checkpoints')
    parser.add_argument('--eval', action='store_true', help='Evaluate model instead of training')
    parser.add_argument('--model-path', type=str, help='Path to the model for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=10, help='Number of episodes for evaluation')
    
    args = parser.parse_args()
    
    if args.eval:
        if args.model_path is None:
            parser.error("--eval requires --model-path")
        
        # Evaluate existing model
        normalize_path = os.path.join(os.path.dirname(args.model_path), "vec_normalize.pkl")
        evaluate_model(args.model_path, normalize_path, args.eval_episodes)
    else:
        # Train new model
        train_model(
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            save_freq=args.save_freq
        )
