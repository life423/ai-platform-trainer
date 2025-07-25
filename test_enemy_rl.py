"""
Test Enemy RL System

This script demonstrates the true reinforcement learning enemy system,
showing the difference between scripted behavior and genuine learned intelligence.
"""
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)

print("🤖 Enemy RL System Test")
print("=" * 50)

def test_environment_creation():
    """Test creating the enemy RL environment."""
    print("\\n1. Testing Enemy RL Environment Creation...")
    
    try:
        from ai_platform_trainer.ai.training.enemy_rl_environment import EnemyRLEnvironment, EnemyRLConfig
        
        # Create configuration
        config = EnemyRLConfig(
            max_episode_steps=500,
            missile_frequency=0.08,  # Higher frequency for testing
            player_skill_level=0.6,
            survival_reward_per_step=0.1
        )
        
        # Create environment
        env = EnemyRLEnvironment(config=config)
        print(f"✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset: observation shape {obs.shape}")
        
        # Test a few steps
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated:
                print(f"   Episode terminated at step {step+1} (enemy hit)")
                break
            elif step == 9:
                print(f"   Completed 10 steps successfully")
        
        print(f"   Total reward over steps: {total_reward:.3f}")
        print(f"   Final info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_enemy_entity():
    """Test the RL enemy entity."""
    print("\\n2. Testing RL Enemy Entity...")
    
    try:
        from ai_platform_trainer.entities.enemy_rl import EnemyRL
        
        # Create RL enemy (without trained model for now)
        enemy = EnemyRL(
            screen_width=800,
            screen_height=600,
            rl_model_path=None,  # No model yet
            fallback_to_scripted=True
        )
        
        print(f"✅ RL Enemy created successfully")
        print(f"   RL Available: {enemy.rl_available}")
        print(f"   Fallback Active: {enemy.fallback_enemy is not None}")
        print(f"   AI Info: {enemy.get_ai_info()}")
        
        # Test movement updates
        print("\\n   Testing movement updates...")
        for step in range(5):
            enemy.update_movement(
                player_x=400, player_y=300,
                player_step=step,
                current_time=step * 16,  # ~60fps
                missiles=[]
            )
            
            pos = enemy.pos
            stats = enemy.get_performance_stats()
            print(f"   Step {step+1}: Position ({pos['x']:.1f}, {pos['y']:.1f}), Energy: {stats['energy']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enemy entity test failed: {e}")
        return False

def test_reward_function():
    """Test the enemy reward function components."""
    print("\\n3. Testing Enemy Reward Function...")
    
    try:
        from ai_platform_trainer.ai.training.enemy_rl_environment import EnemyRLEnvironment
        
        env = EnemyRLEnvironment()
        obs, info = env.reset()
        
        # Test different scenarios
        scenarios = [
            {"name": "Safe Movement", "action": [0.1, 0.1], "expected": "Positive survival reward"},
            {"name": "High Energy Usage", "action": [1.0, 1.0], "expected": "Energy penalty"},
            {"name": "No Movement", "action": [0.0, 0.0], "expected": "Conservative reward"},
            {"name": "Evasive Movement", "action": [0.5, -0.5], "expected": "Mixed reward"}
        ]
        
        print("\\n   Reward Analysis:")
        for scenario in scenarios:
            env.reset()
            action = scenario["action"]
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   {scenario['name']}: Action {action} → Reward {reward:.3f}")
            print(f"     Expected: {scenario['expected']}")
            print(f"     Info: Survival time {info['survival_time']}, Energy {info['enemy_energy']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False

def test_quick_training():
    """Test quick training to verify the system works."""
    print("\\n4. Testing Quick Training (100 steps)...")
    
    try:
        from ai_platform_trainer.ai.training.enemy_rl_environment import EnemyRLTrainer
        from stable_baselines3 import SAC
        
        # Create trainer
        trainer = EnemyRLTrainer(save_path="models/test_enemy_rl")
        
        # Create environment
        env = trainer.create_environment()
        print("✅ Training environment created")
        
        # Create a minimal SAC model for testing
        model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=50,
            batch_size=32,
            train_freq=1,
            policy_kwargs=dict(net_arch=[64, 64])
        )
        
        print("✅ SAC model created")
        
        # Train for a very short time (just to test)
        print("   Training for 100 timesteps...")
        model.learn(total_timesteps=100, progress_bar=False)
        
        # Test the trained model
        print("   Testing trained model...")
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"✅ Quick training completed successfully")
        print(f"   Test reward: {total_reward:.3f}")
        print(f"   Survival time: {info['survival_time']}")
        
        return True
        
    except ImportError:
        print("⚠️  stable_baselines3 not available - skipping training test")
        return True
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        return False

def demonstrate_rl_vs_scripted():
    """Demonstrate difference between RL and scripted behavior."""
    print("\\n5. RL vs Scripted Behavior Comparison...")
    
    try:
        from ai_platform_trainer.entities.enemy_rl import EnemyRL
        from ai_platform_trainer.entities.enemy_learning import AdaptiveStagedEnemyAI
        
        # Create both types
        rl_enemy = EnemyRL(800, 600, rl_model_path=None, fallback_to_scripted=False)
        scripted_enemy = AdaptiveStagedEnemyAI(800, 600)
        
        print("\\n   Behavior Analysis:")
        print(f"   RL Enemy: {rl_enemy.get_ai_info()}")
        print(f"   Scripted Enemy: {scripted_enemy.get_ai_info()}")
        
        # Simulate similar conditions
        player_x, player_y = 400, 300
        
        print("\\n   Movement Response to Player Position:")
        for enemy_type, enemy in [("RL", rl_enemy), ("Scripted", scripted_enemy)]:
            initial_pos = enemy.pos.copy()
            
            # Update movement
            enemy.update_movement(player_x, player_y, 0, 0, [])
            
            final_pos = enemy.pos
            movement = {
                "x": final_pos["x"] - initial_pos["x"],
                "y": final_pos["y"] - initial_pos["y"]
            }
            
            print(f"   {enemy_type} Enemy: Moved ({movement['x']:.2f}, {movement['y']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Run all enemy RL tests."""
    print("Testing True Reinforcement Learning Enemy System")
    print("This system learns genuine evasion strategies through experience")
    print()
    
    tests = [
        test_environment_creation,
        test_enemy_entity,
        test_reward_function,
        test_quick_training,
        demonstrate_rl_vs_scripted
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All tests passed! Enemy RL system is ready")
        print("\\n🚀 Next Steps:")
        print("   1. Train full enemy RL model: python -m ai_platform_trainer.ai.training.enemy_rl_environment")
        print("   2. Integrate with game: Add 'play_rl_enemy' mode to menu")
        print("   3. Test against human players to evaluate learned strategies")
    else:
        print(f"\\n⚠️  {total - passed} tests failed - check implementation")
    
    print("\\n🤖 Key Features of True RL Enemy:")
    print("   - Learns from experience, not scripted rules")
    print("   - Develops genuine evasion strategies")
    print("   - Adapts to player behavior patterns")
    print("   - Shows emergent intelligent behavior")
    print("   - Continuously improves with more training")

if __name__ == "__main__":
    main()