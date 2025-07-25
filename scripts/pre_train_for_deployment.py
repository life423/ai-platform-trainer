#!/usr/bin/env python3
"""
Pre-train AI Models for Deployment

This script pre-trains the missile AI models so they can be bundled
with the executable, eliminating the need for first-time training.
"""
import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup logging for deployment training."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('deployment_training.log'),
            logging.StreamHandler()
        ]
    )

def pre_train_missile_ai():
    """Pre-train missile AI for deployment."""
    try:
        from ai_platform_trainer.ai.training.train_missile_rl import MissileRLTrainer
        
        # Ensure models directory exists
        models_dir = project_root / 'models'
        models_dir.mkdir(exist_ok=True)
        
        logging.info("🚀 Starting missile AI pre-training for deployment...")
        
        # Create trainer
        trainer = MissileRLTrainer(save_path=str(models_dir / 'missile_rl_model'))
        
        # Train with optimized settings for deployment
        # Balance between training time and performance
        total_timesteps = 75000  # Good performance, reasonable training time
        
        start_time = time.time()
        logging.info(f"Training missile AI with {total_timesteps:,} timesteps...")
        
        model = trainer.train(total_timesteps=total_timesteps)
        
        training_time = time.time() - start_time
        logging.info(f"✅ Missile AI training completed in {training_time:.1f} seconds")
        
        # Test the model performance
        logging.info("🧪 Testing trained model performance...")
        model_path = str(models_dir / 'missile_rl_model_final.zip')
        
        if os.path.exists(model_path):
            avg_reward, hit_rate = trainer.test_model(model_path, num_episodes=10)
            
            logging.info(f"📊 Model Performance Results:")
            logging.info(f"   Average Reward: {avg_reward:.2f}")
            logging.info(f"   Hit Rate: {hit_rate:.1%}")
            
            # Performance thresholds
            if hit_rate >= 0.8:
                logging.info("🎯 Excellent performance - model ready for deployment!")
                return True
            elif hit_rate >= 0.6:
                logging.info("✅ Good performance - model acceptable for deployment")
                return True
            else:
                logging.warning(f"⚠️  Performance below target (hit rate: {hit_rate:.1%})")
                logging.warning("Model will still be bundled, but users may experience suboptimal missile AI")
                return True
        else:
            logging.error("❌ Model file not found after training!")
            return False
            
    except ImportError as e:
        logging.error(f"❌ Missing dependencies for AI training: {e}")
        logging.error("Please install: pip install 'stable-baselines3[extra]'")
        return False
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        logging.exception("Exception details:")
        return False

def verify_models():
    """Verify that all required models are present and valid."""
    models_dir = project_root / 'models'
    
    required_models = [
        'missile_rl_model_final.zip',
        'missile_model.pth',  # Fallback neural network model
    ]
    
    logging.info("🔍 Verifying AI models for deployment...")
    
    all_present = True
    for model_file in required_models:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logging.info(f"✅ {model_file}: {size_mb:.1f} MB")
        else:
            logging.warning(f"⚠️  Missing: {model_file}")
            if model_file == 'missile_rl_model_final.zip':
                all_present = False
    
    if all_present:
        logging.info("🎯 All required AI models present and ready for bundling!")
    else:
        logging.warning("⚠️  Some models missing - executable may fall back to basic homing")
    
    return all_present

def create_deployment_info():
    """Create a file with deployment information."""
    info_file = project_root / 'models' / 'deployment_info.json'
    
    import json
    from datetime import datetime
    
    deployment_info = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "training_timesteps": 75000,
        "python_version": sys.version,
        "purpose": "Pre-trained AI models for executable deployment",
        "models": {
            "missile_rl_model_final.zip": "Advanced RL missile AI for intelligent homing",
            "missile_model.pth": "Fallback neural network missile AI"
        }
    }
    
    with open(info_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    logging.info(f"📝 Created deployment info: {info_file}")

def main():
    """Main deployment preparation function."""
    setup_logging()
    
    logging.info("🚀 AI Platform Trainer - Deployment Preparation")
    logging.info("=" * 60)
    
    success = True
    
    # Pre-train missile AI
    if not pre_train_missile_ai():
        success = False
    
    # Verify all models
    if not verify_models():
        logging.warning("Not all models verified, but continuing...")
    
    # Create deployment info
    create_deployment_info()
    
    if success:
        logging.info("✅ Deployment preparation completed successfully!")
        logging.info("🎮 AI models are ready for bundling with executable")
        return 0
    else:
        logging.error("❌ Deployment preparation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())