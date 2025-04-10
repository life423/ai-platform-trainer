#include "../include/entity.h"
#include <cmath>

namespace gpu_env {

Entity::Entity(float x, float y, float size)
    : x(x)
    , y(y)
    , size(size)
    , visible(true)
{
}

bool Entity::collides_with(const Entity& other) const {
    // Calculate Euclidean distance between entity centers
    float dx = this->x - other.x;
    float dy = this->y - other.y;
    float distance_squared = dx * dx + dy * dy;
    
    // Calculate minimum distance for collision
    float min_distance = (this->size + other.size) * 0.5f;
    float min_distance_squared = min_distance * min_distance;
    
    // Return true if entities are overlapping
    return distance_squared <= min_distance_squared;
}

void Entity::set_position(float new_x, float new_y) {
    x = new_x;
    y = new_y;
}

void Entity::wrap_position(float screen_width, float screen_height) {
    // Wrap horizontally
    if (x < 0) {
        x += screen_width;
    } else if (x >= screen_width) {
        x -= screen_width;
    }
    
    // Wrap vertically
    if (y < 0) {
        y += screen_height;
    } else if (y >= screen_height) {
        y -= screen_height;
    }
}

// Player implementation
Player::Player(float x, float y, float size, float step)
    : Entity(x, y, size)
    , step(step)
{
}

void Player::update() {
    // No automatic movement for player
}

void Player::handle_input(float dx, float dy) {
    // Normalize direction if needed
    float length = std::sqrt(dx * dx + dy * dy);
    if (length > 0.001f) {
        dx /= length;
        dy /= length;
    }
    
    // Update position based on input
    x += dx * step;
    y += dy * step;
}

void Player::shoot_missile(float enemy_x, float enemy_y) {
    // Calculate center position for missile
    float missile_x = x + size * 0.5f;
    float missile_y = y + size * 0.5f;
    
    // Calculate direction to enemy
    float dx = enemy_x - missile_x;
    float dy = enemy_y - missile_y;
    
    // Normalize direction
    float length = std::sqrt(dx * dx + dy * dy);
    if (length > 0.001f) {
        dx /= length;
        dy /= length;
    }
    
    // Create missile with calculated velocity
    constexpr float kMissileSpeed = 5.0f;
    constexpr float kMissileSize = 10.0f;
    constexpr float kMissileLifespan = 10000.0f;  // 10 seconds
    
    missiles.push_back(
        std::make_shared<Missile>(
            missile_x, missile_y, kMissileSpeed, 
            dx * kMissileSpeed, dy * kMissileSpeed, 
            kMissileLifespan
        )
    );
}

void Player::update_missiles(float screen_width, float screen_height) {
    for (auto& missile : missiles) {
        missile->update();
        
        // Wrap missile position to screen boundaries
        missile->wrap_position(screen_width, screen_height);
    }
    
    // Remove expired missiles
    missiles.erase(
        std::remove_if(
            missiles.begin(), 
            missiles.end(), 
            [](const std::shared_ptr<Missile>& m) {
                return !m->visible;
            }
        ),
        missiles.end()
    );
}

// Enemy implementation
Enemy::Enemy(float x, float y, float size, float step)
    : Entity(x, y, size)
    , step(step)
    , last_action_dx(0.0f)
    , last_action_dy(0.0f)
{
}

void Enemy::update() {
    // Move using the last action
    x += last_action_dx * step;
    y += last_action_dy * step;
}

void Enemy::apply_action(float dx, float dy) {
    // Store action for next update
    last_action_dx = dx;
    last_action_dy = dy;
}

void Enemy::hide() {
    visible = false;
}

void Enemy::reset(float new_x, float new_y) {
    x = new_x;
    y = new_y;
    visible = true;
    last_action_dx = 0.0f;
    last_action_dy = 0.0f;
}

// Missile implementation
Missile::Missile(float x, float y, float speed, float vx, float vy, float lifespan)
    : Entity(x, y, 10.0f)  // Fixed size for missiles
    , speed(speed)
    , vx(vx)
    , vy(vy)
    , birth_time(0.0f)     // Will be set by environment
    , lifespan(lifespan)
    , angle(std::atan2(vy, vx))
    , has_collided(false)
{
}

void Missile::update() {
    // Update position based on velocity
    x += vx;
    y += vy;
    
    // Update angle based on velocity
    angle = std::atan2(vy, vx);
}

bool Missile::is_expired(float current_time) const {
    return current_time >= birth_time + lifespan;
}

// Batch data structure implementations

void EntityBatch::add(const Entity& entity) {
    x.push_back(entity.x);
    y.push_back(entity.y);
    sizes.push_back(entity.size);
    visible.push_back(entity.visible);
}

void EntityBatch::clear() {
    x.clear();
    y.clear();
    sizes.clear();
    visible.clear();
}

size_t EntityBatch::size() const {
    return x.size();
}

void MissileBatch::add(const Missile& missile) {
    x.push_back(missile.x);
    y.push_back(missile.y);
    vx.push_back(missile.vx);
    vy.push_back(missile.vy);
    sizes.push_back(missile.size);
    birth_times.push_back(missile.birth_time);
    lifespans.push_back(missile.lifespan);
    collided.push_back(missile.has_collided);
    visible.push_back(missile.visible);
}

void MissileBatch::clear() {
    x.clear();
    y.clear();
    vx.clear();
    vy.clear();
    sizes.clear();
    birth_times.clear();
    lifespans.clear();
    collided.clear();
    visible.clear();
}

size_t MissileBatch::size() const {
    return x.size();
}

} // namespace gpu_env
