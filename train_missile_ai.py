#!/usr/bin/env python3
"""
Headless Missile AI Training Script

This script trains a reinforcement learning model for missile homing behavior
in the background without any GUI. The trained model can then be loaded
by the game to provide intelligent homing missiles to players.

Usage:
python train_missile_ai.py [--timesteps 100000] [--test]
"""
import argparse
import logging
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_platform_trainer.ai.training.train_missile_rl import MissileRLTrainer


def setup_logging():
    """Setup logging for training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('missile_training.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train missile AI using reinforcement learning')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Number of training timesteps (default: 100000)')
    parser.add_argument('--test', action='store_true',
                       help='Test existing trained model instead of training')
    parser.add_argument('--model-path', type=str, default='models/missile_rl_model',
                       help='Path to save/load the model')
    
    args = parser.parse_args()
    
    setup_logging()
    
    try:
        # Check if stable_baselines3 is available
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logging.error("stable_baselines3 is not installed. Please install it with:")
            logging.error("pip install stable-baselines3[extra]")
            return 1
        
        # Create trainer
        trainer = MissileRLTrainer(save_path=args.model_path)
        
        if args.test:
            # Test existing model
            logging.info("Testing existing missile AI model...")
            model_file = f"{args.model_path}_final.zip"
            if not os.path.exists(model_file):
                logging.error(f"Model file not found: {model_file}")
                logging.error("Please train a model first by running without --test flag")
                return 1
            
            avg_reward, hit_rate = trainer.test_model(model_file, num_episodes=20)
            
            print(f"\n{'='*50}")
            print(f"MISSILE AI PERFORMANCE TEST RESULTS")
            print(f"{'='*50}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Hit Rate: {hit_rate:.1%}")
            print(f"{'='*50}")
            
            if hit_rate > 0.7:
                print("✅ Missile AI is performing excellently!")
            elif hit_rate > 0.5:
                print("🟡 Missile AI is performing well, but could be better.")
            else:
                print("❌ Missile AI needs more training.")
                
        else:
            # Train new model
            logging.info(f"Starting headless missile AI training...")
            logging.info(f"Training for {args.timesteps:,} timesteps...")
            logging.info(f"This will take approximately {args.timesteps / 1000:.0f} minutes")
            
            print(f"\n{'='*60}")
            print(f"🚀 MISSILE AI TRAINING STARTED")
            print(f"{'='*60}")
            print(f"Timesteps: {args.timesteps:,}")
            print(f"Model will be saved to: {args.model_path}")
            print(f"Training logs: missile_training.log")
            print(f"{'='*60}\n")
            
            # Train the model
            model = trainer.train(total_timesteps=args.timesteps)
            
            print(f"\n{'='*60}")
            print(f"✅ MISSILE AI TRAINING COMPLETED!")
            print(f"{'='*60}")
            print(f"Model saved to: {args.model_path}_final.zip")
            print(f"You can now test it with: python train_missile_ai.py --test")
            print(f"{'='*60}\n")
            
            # Run a quick test
            logging.info("Running quick performance test...")
            avg_reward, hit_rate = trainer.test_model(f"{args.model_path}_final.zip", num_episodes=5)
            
            print(f"Quick Test Results:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Hit Rate: {hit_rate:.1%}")
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())