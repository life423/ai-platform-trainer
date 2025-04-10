#include "../include/environment.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <unordered_map>

namespace gpu_env {

// Forward declaration of reward functions defined in reward.cpp
float calculate_reward(
    const Player& player,
    const Enemy& enemy,
    const MissileBatch& missiles,
    float current_time,
    float last_distance,
    bool enemy_hit_player,
    bool player_missile_hit_enemy,
    const EnvironmentConfig& config
);

bool check_missile_avoidance(
    const Enemy& enemy,
    const MissileBatch& missiles,
    float current_time,
    const EnvironmentConfig& config
);

Environment::Environment(const EnvironmentConfig& config)
    : config_(config)
    , enemy_visible_(true)
    , steps_since_reset_(0)
    , done_(false)
    , last_enemy_player_distance_(0.0f)
    , last_hit_time_(0.0f)
    , has_enemy_hit_player_since_reset_(false)
    , missile_avoidance_count_(0)
{
    // Initialize physics engine
    physics_engine_ = std::make_unique<PhysicsEngine>(config_.screen_width, config_.screen_height);
    
    // Initialize random number generator
    unsigned int seed = std::random_device{}();
    rng_.seed(seed);
    
    // Initialize entities
    spawn_player();
    spawn_enemy();
    
    // Initialize observation buffer
    std::fill(observation_buffer_.begin(), observation_buffer_.end(), 0.0f);
    
    // Initialize danger map
    danger_map_.resize(danger_map_width_ * danger_map_height_, 0.0f);
}

Environment::~Environment() {
    // Cleanup handled by smart pointers
}

std::vector<float> Environment::reset(unsigned int seed) {
    // Seed random generator
    if (seed != 0) {
        rng_.seed(seed);
    } else {
        rng_.seed(std::random_device{}());
    }
    
    // Reset state variables
    steps_since_reset_ = 0;
    done_ = false;
    has_enemy_hit_player_since_reset_ = false;
    missile_avoidance_count_ = 0;
    
    // Respawn entities
    spawn_player();
    spawn_enemy();
    
    // Clear any existing missiles
    if (player_) {
        player_->missiles.clear();
    }
    
    // Calculate initial distance
    last_enemy_player_distance_ = calculate_distance(
        player_->x, player_->y, enemy_->x, enemy_->y
    );
    
    // Reset last hit time
    last_hit_time_ = 0;
    
    // Get initial observation
    return get_observation();
}

std::tuple<std::vector<float>, float, bool, bool, std::unordered_map<std::string, float>> 
Environment::step(const std::vector<float>& action) {
    // Check if already done
    if (done_) {
        return std::make_tuple(
            get_observation(),            // observation
            0.0f,                         // reward
            true,                         // done
            false,                        // truncated
            std::unordered_map<std::string, float>{}  // info
        );
    }
    
    // Increment step counter
    steps_since_reset_++;
    
    // Check for max steps
    bool truncated = steps_since_reset_ >= config_.max_steps;
    
    // Store pre-step distance for reward calculation
    float pre_step_distance = calculate_distance(
        player_->x, player_->y, enemy_->x, enemy_->y
    );
    
    // Simulate player action (random movement for now)
    std::vector<float> player_action = {
        random_float(-1.0f, 1.0f),
        random_float(-1.0f, 1.0f)
    };
    update_player(player_action);
    
    // Update enemy with the provided action
    update_enemy(action);
    
    // Update missiles
    update_missiles();
    
    // Check for collisions
    bool enemy_hit_player = false;
    bool player_missile_hit_enemy = false;
    check_collisions(enemy_hit_player, player_missile_hit_enemy);
    
    // Calculate reward
    float current_time = static_cast<float>(steps_since_reset_);
    MissileBatch missile_batch;
    for (const auto& missile : player_->missiles) {
        missile_batch.add(*missile);
    }
    
    float reward = calculate_reward(
        *player_, *enemy_, missile_batch, current_time,
        pre_step_distance, enemy_hit_player, player_missile_hit_enemy,
        config_
    );
    
    // Check for missile avoidance
    if (check_missile_avoidance(*enemy_, missile_batch, current_time, config_)) {
        missile_avoidance_count_++;
    }
    
    // Update danger map for visualization
    if (config_.enable_missile_avoidance && !player_->missiles.empty()) {
        danger_map_ = physics_engine_->calculate_danger_map(
            missile_batch, danger_map_width_, danger_map_height_
        );
    }
    
    // Update distance for next step
    last_enemy_player_distance_ = calculate_distance(
        player_->x, player_->y, enemy_->x, enemy_->y
    );
    
    // Check if done
    done_ = enemy_hit_player || player_missile_hit_enemy || truncated;
    
    // Get observation
    std::vector<float> observation = get_observation();
    
    // Prepare info dict
    std::unordered_map<std::string, float> info;
    info["steps"] = static_cast<float>(steps_since_reset_);
    info["player_x"] = player_->x / config_.screen_width;
    info["player_y"] = player_->y / config_.screen_height;
    info["enemy_x"] = enemy_->x / config_.screen_width;
    info["enemy_y"] = enemy_->y / config_.screen_height;
    info["distance"] = last_enemy_player_distance_;
    info["missile_count"] = static_cast<float>(player_->missiles.size());
    info["missile_avoidance"] = static_cast<float>(missile_avoidance_count_);
    info["enemy_hit_player"] = enemy_hit_player ? 1.0f : 0.0f;
    info["player_missile_hit_enemy"] = player_missile_hit_enemy ? 1.0f : 0.0f;
    
    return std::make_tuple(observation, reward, done_, truncated, info);
}

std::vector<int> Environment::get_observation_shape() const {
    return {observation_size_};
}

std::vector<int> Environment::get_action_shape() const {
    return {2};  // (dx, dy) for enemy movement
}

const EnvironmentConfig& Environment::get_config() const {
    return config_;
}

std::vector<std::vector<float>> Environment::batch_reset(
    int batch_size, const std::vector<unsigned int>& seeds
) {
    std::vector<std::vector<float>> observations;
    observations.reserve(batch_size);
    
    for (int i = 0; i < batch_size; ++i) {
        unsigned int seed = (i < static_cast<int>(seeds.size())) ? seeds[i] : 0;
        observations.push_back(reset(seed));
    }
    
    return observations;
}

std::vector<std::tuple<std::vector<float>, float, bool, bool, std::unordered_map<std::string, float>>>
Environment::batch_step(const std::vector<std::vector<float>>& actions) {
    std::vector<std::tuple<std::vector<float>, float, bool, bool, std::unordered_map<std::string, float>>> results;
    results.reserve(actions.size());
    
    for (const auto& action : actions) {
        results.push_back(step(action));
    }
    
    return results;
}

std::unordered_map<std::string, std::vector<float>> Environment::get_debug_data() const {
    std::unordered_map<std::string, std::vector<float>> debug_data;
    
    // Add basic entity data
    debug_data["player_pos"] = {player_->x, player_->y};
    debug_data["enemy_pos"] = {enemy_->x, enemy_->y};
    
    // Add missile data
    std::vector<float> missile_data;
    for (const auto& missile : player_->missiles) {
        missile_data.push_back(missile->x);
        missile_data.push_back(missile->y);
        missile_data.push_back(missile->vx);
        missile_data.push_back(missile->vy);
        missile_data.push_back(missile->angle);
    }
    debug_data["missiles"] = missile_data;
    
    // Add danger map
    debug_data["danger_map"] = danger_map_;
    debug_data["danger_map_width"] = {static_cast<float>(danger_map_width_)};
    debug_data["danger_map_height"] = {static_cast<float>(danger_map_height_)};
    
    // Add evasion vector if available
    if (config_.enable_missile_avoidance && !player_->missiles.empty()) {
        MissileBatch missile_batch;
        for (const auto& missile : player_->missiles) {
            missile_batch.add(*missile);
        }
        
        auto evasion_vector = physics_engine_->calculate_evasion_vector(
            enemy_->x, enemy_->y, missile_batch, config_.missile_prediction_steps
        );
        
        debug_data["evasion_vector"] = {evasion_vector.first, evasion_vector.second};
    }
    
    return debug_data;
}

void Environment::update_player(const std::vector<float>& player_action) {
    if (!player_) return;
    
    // Apply player movement
    if (player_action.size() >= 2) {
        player_->handle_input(player_action[0], player_action[1]);
    }
    
    // Wrap position
    player_->wrap_position(config_.screen_width, config_.screen_height);
    
    // Maybe fire a missile
    if (random_float(0.0f, 1.0f) < 0.01f && player_->missiles.size() < static_cast<size_t>(config_.max_missiles)) {
        player_->shoot_missile(enemy_->x, enemy_->y);
    }
}

void Environment::update_enemy(const std::vector<float>& enemy_action) {
    if (!enemy_ || !enemy_->visible) return;
    
    // Normalize action values to [-1, 1]
    float dx = std::max(-1.0f, std::min(1.0f, enemy_action[0]));
    float dy = std::max(-1.0f, std::min(1.0f, enemy_action[1]));
    
    // Apply enemy movement
    enemy_->apply_action(dx, dy);
    enemy_->update();
    
    // Wrap position
    enemy_->wrap_position(config_.screen_width, config_.screen_height);
}

void Environment::update_missiles() {
    if (!player_) return;
    
    // Update missile positions
    player_->update_missiles(config_.screen_width, config_.screen_height);
    
    // Update missile physics if we have any
    if (!player_->missiles.empty()) {
        // Prepare velocity vectors
        std::vector<float> velocities_x;
        std::vector<float> velocities_y;
        EntityBatch missile_entities;
        
        for (const auto& missile : player_->missiles) {
            missile_entities.add(*missile);
            velocities_x.push_back(missile->vx);
            velocities_y.push_back(missile->vy);
        }
        
        // Use physics engine to update positions
        physics_engine_->update_positions(
            missile_entities, velocities_x, velocities_y
        );
        
        // Update missile objects with new positions from batch
        for (size_t i = 0; i < player_->missiles.size(); ++i) {
            player_->missiles[i]->x = missile_entities.x[i];
            player_->missiles[i]->y = missile_entities.y[i];
        }
    }
}

void Environment::check_collisions(bool& enemy_hit_player, bool& player_missile_hit_enemy) {
    enemy_hit_player = false;
    player_missile_hit_enemy = false;
    
    if (!player_ || !enemy_ || !enemy_->visible) return;
    
    // Check for direct collision between player and enemy
    if (player_->collides_with(*enemy_)) {
        enemy_hit_player = true;
        has_enemy_hit_player_since_reset_ = true;
        last_hit_time_ = static_cast<float>(steps_since_reset_);
        return;
    }
    
    // Check for missile collisions
    for (auto& missile : player_->missiles) {
        if (missile->collides_with(*enemy_)) {
            player_missile_hit_enemy = true;
            enemy_->hide();
            missile->has_collided = true;
            missile->visible = false;
            return;
        }
    }
}

std::vector<float> Environment::get_observation() const {
    std::vector<float> observation(observation_size_, 0.0f);
    
    if (!player_ || !enemy_) return observation;
    
    // Normalize positions to [0,1] range
    float screen_width = static_cast<float>(config_.screen_width);
    float screen_height = static_cast<float>(config_.screen_height);
    
    // Player and enemy positions
    observation[0] = player_->x / screen_width;
    observation[1] = player_->y / screen_height;
    observation[2] = enemy_->x / screen_width;
    observation[3] = enemy_->y / screen_height;
    
    // Enemy velocity
    observation[4] = enemy_->last_action_dx;
    observation[5] = enemy_->last_action_dy;
    
    // Distance between player and enemy
    observation[6] = last_enemy_player_distance_ / std::max(screen_width, screen_height);
    
    // Time since last hit
    float current_time = static_cast<float>(steps_since_reset_);
    observation[7] = (current_time - last_hit_time_) / 100.0f;
    
    // Missile information (up to the closest 2 missiles)
    int missile_idx = 8;
    if (!player_->missiles.empty()) {
        // Sort missiles by distance to enemy
        std::vector<std::pair<float, size_t>> missile_distances;
        for (size_t i = 0; i < player_->missiles.size(); ++i) {
            const auto& missile = player_->missiles[i];
            float distance = calculate_distance(
                missile->x, missile->y, enemy_->x, enemy_->y
            );
            missile_distances.emplace_back(distance, i);
        }
        
        // Sort by distance (closest first)
        std::sort(missile_distances.begin(), missile_distances.end());
        
        // Add closest missile info to observation
        size_t num_missiles = std::min(player_->missiles.size(), static_cast<size_t>(2));
        for (size_t i = 0; i < num_missiles; ++i) {
            size_t idx = missile_distances[i].second;
            const auto& missile = player_->missiles[idx];
            
            // Normalized position
            observation[missile_idx++] = missile->x / screen_width;
            observation[missile_idx++] = missile->y / screen_height;
            
            // Normalized velocity
            observation[missile_idx++] = missile->vx / 10.0f;
            observation[missile_idx++] = missile->vy / 10.0f;
            
            // Distance to enemy (normalized)
            float missile_distance = missile_distances[i].first;
            observation[missile_idx++] = missile_distance / std::max(screen_width, screen_height);
        }
    }
    
    return observation;
}

float Environment::calculate_reward() {
    // This is a stub - the actual implementation is in reward.cpp
    return 0.0f;
}

void Environment::spawn_enemy() {
    // Initialize enemy at a random position
    float x = random_float(0, config_.screen_width - config_.enemy_size);
    float y = random_float(0, config_.screen_height - config_.enemy_size);
    
    // Keep enemy away from player at start
    if (player_) {
        float min_distance = std::max(config_.screen_width, config_.screen_height) * 0.3f;
        int max_attempts = 10;
        
        for (int i = 0; i < max_attempts; ++i) {
            float distance = calculate_distance(player_->x, player_->y, x, y);
            if (distance >= min_distance) {
                break;
            }
            
            // Try new position
            x = random_float(0, config_.screen_width - config_.enemy_size);
            y = random_float(0, config_.screen_height - config_.enemy_size);
        }
    }
    
    enemy_ = std::make_unique<Enemy>(x, y, config_.enemy_size, config_.enemy_speed);
    enemy_visible_ = true;
}

void Environment::spawn_player() {
    // Initialize player at a random position
    float x = random_float(0, config_.screen_width - config_.player_size);
    float y = random_float(0, config_.screen_height - config_.player_size);
    
    player_ = std::make_unique<Player>(x, y, config_.player_size, config_.player_speed);
}

std::pair<float, float> Environment::calculate_evasion_vector() {
    // Skip if no missiles or avoidance disabled
    if (!player_ || player_->missiles.empty() || !config_.enable_missile_avoidance) {
        return std::make_pair(0.0f, 0.0f);
    }
    
    // Prepare missile batch
    MissileBatch missile_batch;
    for (const auto& missile : player_->missiles) {
        missile_batch.add(*missile);
    }
    
    // Use physics engine to calculate evasion vector
    return physics_engine_->calculate_evasion_vector(
        enemy_->x, enemy_->y, missile_batch, config_.missile_prediction_steps
    );
}

float Environment::random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng_);
}

int Environment::random_int(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng_);
}

float Environment::calculate_distance(float x1, float y1, float x2, float y2) const {
    float dx = x1 - x2;
    float dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

} // namespace gpu_env
