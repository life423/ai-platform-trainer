#!/usr/bin/env python3
"""
Debug Enemy RL Collision Detection

This script tests the specific collision detection issue in the enemy RL environment.
"""
import logging
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_missile_collision():
    """Test missile collision detection in isolation."""
    print("🔍 Testing Missile Collision Detection...")
    
    try:
        from ai_platform_trainer.ai.training.enemy_rl_environment import EnemyRLEnvironment
        
        # Create environment
        env = EnemyRLEnvironment()
        obs, info = env.reset()
        
        print(f"✅ Environment created and reset successfully")
        
        # Try a few steps to see when collision detection fails
        for step in range(5):
            print(f"\n--- Step {step + 1} ---")
            action = env.action_space.sample()
            print(f"Action: {action}")
            
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"✅ Step completed - Reward: {reward:.3f}, Terminated: {terminated}")
                print(f"   Active missiles: {len(env.missiles)}")
                print(f"   Enemy position: ({env.enemy_pos['x']:.1f}, {env.enemy_pos['y']:.1f})")
                
                # Manual missile inspection
                for i, missile in enumerate(env.missiles):
                    print(f"   Missile {i}: pos=({missile.pos['x']:.1f}, {missile.pos['y']:.1f}), vx={missile.vx:.1f}, vy={missile.vy:.1f}")
                    
                    # Test missile attributes
                    if hasattr(missile, 'size'):
                        print(f"     Size: {missile.size}")
                    else:
                        print(f"     ❌ Missing size attribute!")
                    
                    # Test get_rect() method directly
                    try:
                        rect = missile.get_rect()
                        print(f"     ✅ get_rect() works: {rect}")
                    except Exception as e:
                        print(f"     ❌ get_rect() failed: {e}")
                        print(f"     Missile type: {type(missile)}")
                        print(f"     Missile attributes: {dir(missile)}")
                
                if terminated:
                    print(f"Episode terminated (enemy hit)")
                    break
                    
            except Exception as e:
                print(f"❌ Step failed with error: {e}")
                traceback.print_exc()
                
                # Debug the missiles at the time of failure
                print(f"\n🔍 Debug info at failure:")
                print(f"   Number of missiles: {len(env.missiles)}")
                for i, missile in enumerate(env.missiles):
                    print(f"   Missile {i} type: {type(missile)}")
                    print(f"   Missile {i} dir: {[attr for attr in dir(missile) if not attr.startswith('_')]}")
                break
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

def test_missile_creation():
    """Test missile creation directly."""
    print("\n🔍 Testing Missile Creation...")
    
    try:
        from ai_platform_trainer.entities.missile import Missile
        import pygame
        
        # Initialize pygame
        pygame.init()
        
        # Create missile
        missile = Missile(
            x=100, y=100,
            speed=8.0,
            vx=5.0, vy=3.0,
            birth_time=pygame.time.get_ticks(),
            lifespan=5000
        )
        
        print(f"✅ Missile created successfully")
        print(f"   Type: {type(missile)}")
        print(f"   Position: {missile.pos}")
        print(f"   Velocity: ({missile.vx}, {missile.vy})")
        print(f"   Size: {missile.size}")
        
        # Test get_rect()
        rect = missile.get_rect()
        print(f"   ✅ get_rect() works: {rect}")
        
        return True
        
    except Exception as e:
        print(f"❌ Missile creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run debug tests."""
    print("🔍 Enemy RL Collision Detection Debug")
    print("=" * 50)
    
    tests = [
        test_missile_creation,
        test_missile_collision
    ]
    
    for test in tests:
        test()
        print()

if __name__ == "__main__":
    main()