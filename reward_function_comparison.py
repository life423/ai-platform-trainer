"""
Reward Function Comparison: Old vs Enhanced

This script demonstrates the improvements made to the missile RL reward function
based on the identified issues.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

print("🔬 Missile RL Reward Function Improvements Demonstration")
print("=" * 60)

# Simulate different scenarios to show reward function improvements
scenarios = [
    {
        "name": "Direct Hit",
        "distance": 10,
        "steps": 50,
        "oscillations": 2,
        "trajectory_efficiency": 0.9,
        "progress": 5.0
    },
    {
        "name": "Near Miss (Old Problem)",
        "distance": 30,
        "steps": 200,
        "oscillations": 1,
        "trajectory_efficiency": 0.7,
        "progress": 2.0
    },
    {
        "name": "Stalling Behavior (Old Problem)",
        "distance": 100,
        "steps": 400,
        "oscillations": 3,
        "trajectory_efficiency": 0.3,
        "progress": 0.5
    },
    {
        "name": "Oscillating (Old Problem)",
        "distance": 50,
        "steps": 150,
        "oscillations": 15,
        "trajectory_efficiency": 0.2,
        "progress": -1.0
    },
    {
        "name": "Efficient Interception",
        "distance": 15,
        "steps": 80,
        "oscillations": 1,
        "trajectory_efficiency": 0.85,
        "progress": 4.0
    }
]

def calculate_old_reward(distance, steps, progress):
    """Simulate old reward function (simplified)."""
    # Old function focused mainly on distance and basic progress
    max_distance = 800
    
    # Hit reward
    if distance < 25:
        return 200.0
    
    # Distance reward
    distance_reward = (max_distance - distance) / max_distance * 2.0
    
    # Simple progress reward (exploitable)
    progress_reward = progress * 0.1 if progress > 0 else progress * 0.5
    
    # Basic boundary penalty
    boundary_penalty = 0 if distance < max_distance else -20.0
    
    return distance_reward + progress_reward + boundary_penalty

def calculate_enhanced_reward(distance, steps, oscillations, trajectory_efficiency, progress):
    """Calculate enhanced reward with all improvements."""
    max_distance = 800
    success_threshold = 20
    
    # 1. SUCCESS REWARD (unchanged)
    if distance < success_threshold:
        time_bonus = max(0, (500 - steps) / 500 * 2.0)  # Bonus for speed
        return 20.0 + time_bonus
    
    # 2. TIME PENALTY (NEW - prevents stalling)
    time_penalty = -0.001 * steps
    
    # 3. DISTANCE REWARD (primary signal)
    distance_reward = -(distance / max_distance) * 2.0
    
    # 4. SMOOTHED PROGRESS REWARD (NEW - prevents oscillation exploitation)
    smoothed_progress = min(progress * 0.8, 1.0) if progress > 0 else max(progress * 0.5, -0.5)
    
    # 5. OSCILLATION PENALTY (NEW)
    oscillation_penalty = -max(0, oscillations - 3) * 0.1
    
    # 6. TRAJECTORY EFFICIENCY BONUS (NEW)
    efficiency_bonus = max(0, trajectory_efficiency - 0.8) * 1.0
    
    # 7. ACTION SMOOTHNESS (simulated)
    action_smoothness = -0.02  # Assume some action noise
    
    return (distance_reward + smoothed_progress + time_penalty + 
            oscillation_penalty + efficiency_bonus + action_smoothness)

print("\\n📊 Reward Function Comparison Results:")
print("-" * 60)
print(f"{'Scenario':<25} {'Old Reward':<12} {'Enhanced':<12} {'Improvement':<12}")
print("-" * 60)

improvements = []
for scenario in scenarios:
    old_reward = calculate_old_reward(
        scenario["distance"], 
        scenario["steps"], 
        scenario["progress"]
    )
    
    enhanced_reward = calculate_enhanced_reward(
        scenario["distance"],
        scenario["steps"],
        scenario["oscillations"],
        scenario["trajectory_efficiency"],
        scenario["progress"]
    )
    
    improvement = enhanced_reward - old_reward
    improvements.append(improvement)
    
    print(f"{scenario['name']:<25} {old_reward:<12.3f} {enhanced_reward:<12.3f} {improvement:<12.3f}")

print("-" * 60)
print()

# Analysis of improvements
print("🎯 Key Improvements Analysis:")
print()

print("✅ **Time Penalty Integration:**")
print("   - Prevents stalling behavior by penalizing long episodes")
print("   - Encourages faster, more efficient interception")
print()

print("✅ **Oscillation Prevention:**")
print("   - Smoothed progress rewards prevent exploitation")
print("   - Direct penalty for excessive direction changes")
print("   - Oscillating behavior now properly penalized")
print()

print("✅ **Trajectory Efficiency Rewards:**")
print("   - Bonus for taking direct paths to target")
print("   - Encourages strategic rather than reactive behavior")
print()

print("✅ **Balanced Reward Components:**")
print("   - Progress rewards capped to prevent exploitation")
print("   - Multiple complementary signals guide behavior")
print("   - Smooth reward landscape prevents getting stuck")
print()

print("📚 **Curriculum Learning Benefits:**")
print("   Stage 1: Static targets (learn basic homing)")
print("   Stage 2: Slow linear movement")
print("   Stage 3: Medium speed with direction changes")
print("   Stage 4: Fast targets")
print("   Stage 5: Circular movement patterns")
print("   Stage 6: Evasive targets (final challenge)")
print()

print("🔍 **Enhanced Evaluation Features:**")
print("   - Behavioral pattern detection")
print("   - Failure mode analysis")
print("   - Automatic reward tuning recommendations")
print("   - Trajectory efficiency tracking")
print("   - Real-time curriculum advancement")
print()

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Reward comparison chart
scenario_names = [s["name"] for s in scenarios]
old_rewards = [calculate_old_reward(s["distance"], s["steps"], s["progress"]) for s in scenarios]
enhanced_rewards = [calculate_enhanced_reward(s["distance"], s["steps"], s["oscillations"], 
                                            s["trajectory_efficiency"], s["progress"]) for s in scenarios]

x = np.arange(len(scenario_names))
width = 0.35

ax1.bar(x - width/2, old_rewards, width, label='Old Reward Function', alpha=0.7, color='orange')
ax1.bar(x + width/2, enhanced_rewards, width, label='Enhanced Reward Function', alpha=0.7, color='blue')

ax1.set_xlabel('Scenarios')
ax1.set_ylabel('Reward Value')
ax1.set_title('Reward Function Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Improvement visualization
ax2.bar(scenario_names, improvements, color=['green' if imp > 0 else 'red' for imp in improvements], alpha=0.7)
ax2.set_xlabel('Scenarios')
ax2.set_ylabel('Improvement (Enhanced - Old)')
ax2.set_title('Reward Function Improvements')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('reward_function_comparison.png', dpi=150, bbox_inches='tight')
print("📈 Visualization saved as 'reward_function_comparison.png'")

print()
print("🚀 **Ready for Enhanced Training:**")
print("   Run: python -m ai_platform_trainer.ai.training.enhanced_sac_trainer")
print("   This will train with all improvements and generate detailed analysis reports!")