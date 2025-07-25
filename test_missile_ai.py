#!/usr/bin/env python3
"""Test script to verify missile AI functionality."""

import pygame
import logging
from ai_platform_trainer.ai.missile_ai_loader import create_smart_missile, get_missile_ai_status

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_missile_creation():
    """Test if smart missiles are created with AI."""
    pygame.init()
    
    print("Testing Missile AI System...")
    print("-" * 50)
    
    # Check AI status
    ai_status = get_missile_ai_status()
    print(f"Missile AI Status: {ai_status}")
    
    # Create a test missile
    missile = create_smart_missile(
        x=100, y=100,
        target_x=500, target_y=500,
        speed=5.0,
        vx=5.0, vy=0.0,
        birth_time=0,
        lifespan=10000
    )
    
    print(f"Missile created: {type(missile).__name__}")
    print(f"Has AI model: {missile.ai_model is not None}")
    print(f"Has RL model: {missile.rl_model is not None}")
    print(f"Using RL: {missile.use_rl}")
    print(f"Has update_with_ai method: {hasattr(missile, 'update_with_ai')}")
    
    # Test update
    player_pos = {"x": 100, "y": 100}
    target_pos = {"x": 500, "y": 500}
    
    print("\nTesting missile update...")
    initial_pos = missile.pos.copy()
    missile.update_with_ai(player_pos, target_pos)
    final_pos = missile.pos.copy()
    
    print(f"Initial position: ({initial_pos['x']:.1f}, {initial_pos['y']:.1f})")
    print(f"Final position: ({final_pos['x']:.1f}, {final_pos['y']:.1f})")
    print(f"Position changed: {initial_pos != final_pos}")
    
    pygame.quit()

if __name__ == "__main__":
    test_missile_creation()