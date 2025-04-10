#include "../include/environment.h"
#include <cmath>
#include <algorithm>

namespace gpu_env {

// Define a reward class that will handle the calculation of rewards
// for our RL environment
class RewardCalculator {
public:
    RewardCalculator(const EnvironmentConfig& config)
        : config_(config)
        , player_hit_reward_(-10.0f)
        , enemy_hit_reward_(-5.0f)
        , missile_avoid_reward_(2.0f)
        , maintain_distance_reward_(0.1f)
        , optimal_distance_(200.0f)
        , distance_tolerance_(50.0f)
        , last_avoidance_time_(0.0f)
        , avoidance_cooldown_(30.0f)  // frames
    {
    }

    float calculate_reward(
        const Player& player,
        const Enemy& enemy,
        const MissileBatch& missiles,
        float current_time,
        float last_distance,
        bool enemy_hit_player,
        bool player_missile_hit_enemy
    ) {
        float reward = 0.0f;
        
        // Penalize the enemy for being hit by a missile
        if (player_missile_hit_enemy) {
            reward += enemy_hit_reward_;
        }
        
        // Penalize the enemy for hitting the player
        if (enemy_hit_player) {
            reward += player_hit_reward_;
        }
        
        // Calculate current distance between player and enemy
        float current_distance = calculate_distance(
            player.x, player.y,
            enemy.x, enemy.y
        );
        
        // Reward for maintaining an optimal distance from player
        // Closer to optimal = higher reward
        float distance_reward = calculate_distance_reward(current_distance);
        reward += distance_reward;
        
        // Reward for actively pursuing player (closing distance)
        // Only if we're too far away
        if (current_distance > optimal_distance_ + distance_tolerance_) {
            float distance_change = last_distance - current_distance;
            reward += distance_change * maintain_distance_reward_;
        }
        
        // Reward for missile avoidance
        float avoidance_reward = calculate_avoidance_reward(
            enemy, missiles, current_time
        );
        reward += avoidance_reward;
        
        return reward;
    }
    
    // Check if a missile was successfully avoided
    bool check_missile_avoidance(
        const Enemy& enemy,
        const MissileBatch& missiles,
        float current_time
    ) {
        if (current_time - last_avoidance_time_ < avoidance_cooldown_) {
            return false;  // Still in cooldown from last avoidance
        }
        
        for (size_t i = 0; i < missiles.size(); ++i) {
            // Skip if missile is collided or not visible
            if (missiles.collided[i] || !missiles.visible[i]) {
                continue;
            }
            
            // Calculate distance to missile
            float dx = missiles.x[i] - enemy.x;
            float dy = missiles.y[i] - enemy.y;
            float distance = std::sqrt(dx * dx + dy * dy);
            
            // Calculate missile velocity magnitude
            float vx = missiles.vx[i];
            float vy = missiles.vy[i];
            float v_mag = std::sqrt(vx * vx + vy * vy);
            
            // Calculate normalized direction to missile
            float nx = dx / distance;
            float ny = dy / distance;
            
            // Calculate dot product (projection of velocity onto direction)
            float dot_product = vx * nx + vy * ny;
            
            // Check if missile is moving toward the enemy and is close
            if (dot_product < 0 && distance < config_.missile_danger_radius) {
                // Calculate perpendicular distance to missile path
                float perp_distance = std::abs(dx * vy - dy * vx) / v_mag;
                
                // If we're close to the path but not hit, we avoided it!
                if (perp_distance < config_.enemy_size * 1.5f && 
                    perp_distance > config_.enemy_size * 0.5f) {
                    last_avoidance_time_ = current_time;
                    return true;
                }
            }
        }
        
        return false;
    }

private:
    const EnvironmentConfig& config_;
    
    // Reward values
    float player_hit_reward_;
    float enemy_hit_reward_;
    float missile_avoid_reward_;
    float maintain_distance_reward_;
    
    // Distance parameters
    float optimal_distance_;
    float distance_tolerance_;
    
    // Avoidance tracking
    float last_avoidance_time_;
    float avoidance_cooldown_;
    
    // Helper methods
    float calculate_distance(float x1, float y1, float x2, float y2) const {
        float dx = x1 - x2;
        float dy = y1 - y2;
        return std::sqrt(dx * dx + dy * dy);
    }
    
    float calculate_distance_reward(float distance) const {
        // Optimize to stay within a certain range of the player
        float distance_diff = std::abs(distance - optimal_distance_);
        if (distance_diff < distance_tolerance_) {
            return 0.1f;  // Small reward for being at optimal distance
        } else {
            // Penalty scales with distance from optimal range
            return -0.02f * (distance_diff - distance_tolerance_) / 100.0f;
        }
    }
    
    float calculate_avoidance_reward(
        const Enemy& enemy,
        const MissileBatch& missiles,
        float current_time
    ) {
        // Check for successful missile avoidance
        if (check_missile_avoidance(enemy, missiles, current_time)) {
            return missile_avoid_reward_;  // Reward for avoiding missiles
        }
        
        return 0.0f;
    }
};

// These functions will be used by the Environment class
float calculate_reward(
    const Player& player,
    const Enemy& enemy,
    const MissileBatch& missiles,
    float current_time,
    float last_distance,
    bool enemy_hit_player,
    bool player_missile_hit_enemy,
    const EnvironmentConfig& config
) {
    static RewardCalculator calculator(config);
    
    return calculator.calculate_reward(
        player, enemy, missiles, current_time,
        last_distance, enemy_hit_player, player_missile_hit_enemy
    );
}

bool check_missile_avoidance(
    const Enemy& enemy,
    const MissileBatch& missiles,
    float current_time,
    const EnvironmentConfig& config
) {
    static RewardCalculator calculator(config);
    
    return calculator.check_missile_avoidance(enemy, missiles, current_time);
}

} // namespace gpu_env
