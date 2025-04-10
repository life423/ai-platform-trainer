#pragma once

#include <vector>
#include <memory>
#include <random>
#include <string>
#include <array>

#include "entity.h"
#include "physics.h"

namespace gpu_env {

// Forward declarations for Python bindings
class PyEnvironment;

// Environment configuration
struct EnvironmentConfig {
    int screen_width = 800;
    int screen_height = 600;
    int max_missiles = 5;
    float player_size = 50.0f;
    float enemy_size = 50.0f;
    float missile_size = 10.0f;
    float player_speed = 5.0f;
    float enemy_speed = 5.0f;
    float missile_speed = 5.0f;
    float missile_lifespan = 10000.0f; // 10 seconds
    float respawn_delay = 500.0f;
    int max_steps = 1000;
    bool enable_missile_avoidance = true;
    int missile_prediction_steps = 30;
    float missile_detection_radius = 250.0f;
    float missile_danger_radius = 150.0f;
    float evasion_strength = 2.5f;
};

// The main game environment
class Environment {
public:
    // Constructor/destructor
    Environment(const EnvironmentConfig& config = EnvironmentConfig());
    ~Environment();

    // Reset the environment
    std::vector<float> reset(unsigned int seed = 0);

    // Step the environment
    std::tuple<std::vector<float>, float, bool, bool, std::unordered_map<std::string, float>> step(
        const std::vector<float>& action
    );

    // Get observation space shape
    std::vector<int> get_observation_shape() const;

    // Get action space shape
    std::vector<int> get_action_shape() const;

    // Get environment configuration
    const EnvironmentConfig& get_config() const;

    // Batch processing interface (for vectorized environments)
    std::vector<std::vector<float>> batch_reset(int batch_size, const std::vector<unsigned int>& seeds);
    std::vector<std::tuple<std::vector<float>, float, bool, bool, std::unordered_map<std::string, float>>> batch_step(
        const std::vector<std::vector<float>>& actions
    );

    // Debug visualization helpers
    std::unordered_map<std::string, std::vector<float>> get_debug_data() const;

private:
    // Game state
    EnvironmentConfig config_;
    std::unique_ptr<Player> player_;
    std::unique_ptr<Enemy> enemy_;
    bool enemy_visible_;
    int steps_since_reset_;
    bool done_;

    // Physics engine
    std::unique_ptr<PhysicsEngine> physics_engine_;

    // Random number generation
    std::mt19937 rng_;
    
    // Observation space handling
    static constexpr int observation_size_ = 10;  // player_x, player_y, enemy_x, enemy_y, 
                                                  // missile data (3 closest missiles x position, y position, velocity)
    std::array<float, observation_size_> observation_buffer_;

    // Reward calculation
    float last_enemy_player_distance_;
    float last_hit_time_;
    bool has_enemy_hit_player_since_reset_;
    int missile_avoidance_count_;
    
    // Danger map for missile avoidance
    static constexpr int danger_map_width_ = 20;
    static constexpr int danger_map_height_ = 15;
    std::vector<float> danger_map_;
    
    // Helper methods
    void update_player(const std::vector<float>& player_action = {});
    void update_enemy(const std::vector<float>& enemy_action);
    void update_missiles();
    void check_collisions();
    std::vector<float> get_observation() const;
    float calculate_reward();
    void spawn_enemy();
    void spawn_player();
    std::pair<float, float> calculate_evasion_vector();

    // Utility methods
    float random_float(float min, float max);
    int random_int(int min, int max);
    float calculate_distance(float x1, float y1, float x2, float y2) const;

    // Friend classes
    friend class PyEnvironment;
};

} // namespace gpu_env
