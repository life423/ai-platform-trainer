#pragma once

#include <vector>
#include <memory>
#include <string>

namespace gpu_env {

// Forward declarations
class Entity;
class Player;
class Enemy;
class Missile;

// Base entity class
class Entity {
public:
    Entity(float x, float y, float size);
    virtual ~Entity() = default;

    // Common properties
    float x, y;
    float size;
    bool visible;

    // Virtual methods
    virtual void update() = 0;
    virtual bool collides_with(const Entity& other) const;
    
    // Utility methods
    void set_position(float new_x, float new_y);
    void wrap_position(float screen_width, float screen_height);
};

// Player entity
class Player : public Entity {
public:
    Player(float x, float y, float size, float step);
    ~Player() override = default;

    // Player-specific properties
    float step;
    std::vector<std::shared_ptr<Missile>> missiles;

    // Methods
    void update() override;
    void handle_input(float dx, float dy);
    void shoot_missile(float enemy_x, float enemy_y);
    void update_missiles(float screen_width, float screen_height);
};

// Enemy entity
class Enemy : public Entity {
public:
    Enemy(float x, float y, float size, float step);
    ~Enemy() override = default;

    // Enemy-specific properties
    float step;
    float last_action_dx;
    float last_action_dy;

    // Methods
    void update() override;
    void apply_action(float dx, float dy);
    void hide();
    void reset(float x, float y);
};

// Missile entity
class Missile : public Entity {
public:
    Missile(float x, float y, float speed, float vx, float vy, float lifespan);
    ~Missile() override = default;

    // Missile-specific properties
    float speed;
    float vx, vy;
    float birth_time;
    float lifespan;
    float angle;
    bool has_collided;

    // Methods
    void update() override;
    bool is_expired(float current_time) const;
};

// Batch data structures for CUDA processing
struct EntityBatch {
    // Struct-of-arrays layout for better GPU memory access patterns
    std::vector<float> x;          // X positions
    std::vector<float> y;          // Y positions
    std::vector<float> sizes;      // Entity sizes
    std::vector<bool> visible;     // Visibility flags

    void add(const Entity& entity);
    void clear();
    size_t size() const;
};

struct MissileBatch {
    // Struct-of-arrays layout optimized for GPU
    std::vector<float> x;          // X positions
    std::vector<float> y;          // Y positions
    std::vector<float> vx;         // X velocities
    std::vector<float> vy;         // Y velocities
    std::vector<float> sizes;      // Missile sizes
    std::vector<float> birth_times;// Birth times
    std::vector<float> lifespans;  // Lifespans
    std::vector<bool> collided;    // Collision flags
    std::vector<bool> visible;     // Visibility flags

    void add(const Missile& missile);
    void clear();
    size_t size() const;
};

} // namespace gpu_env
