#pragma once

#include <vector>
#include <memory>
#include "entity.h"

// Conditionally include CUDA headers
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace gpu_env {

// Vector structure (usable with or without CUDA)
struct Vector2 {
    float x;
    float y;
};

// Forward declaration of CUDA kernel launcher functions
// These functions will be implemented in physics.cu

/**
 * Update positions of entities in parallel on the GPU
 * 
 * @param entities_x X-coordinates of entities
 * @param entities_y Y-coordinates of entities
 * @param velocities_x X velocities
 * @param velocities_y Y velocities
 * @param count Number of entities to process
 * @param screen_width Width of the screen for wrapping
 * @param screen_height Height of the screen for wrapping
 */
void cuda_update_positions(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
);

/**
 * Detect collisions between entities in parallel on the GPU
 * 
 * @param entities_a_x X-coordinates of first entity set
 * @param entities_a_y Y-coordinates of first entity set
 * @param entities_a_sizes Sizes of first entity set
 * @param entities_a_count Number of entities in first set
 * @param entities_b_x X-coordinates of second entity set
 * @param entities_b_y Y-coordinates of second entity set
 * @param entities_b_sizes Sizes of second entity set
 * @param entities_b_count Number of entities in second set
 * @param collision_matrix Output collision matrix (a_count x b_count)
 */
void cuda_detect_collisions(
    const float* entities_a_x,
    const float* entities_a_y,
    const float* entities_a_sizes,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    const float* entities_b_sizes,
    int entities_b_count,
    bool* collision_matrix
);

/**
 * Predict missile trajectories and calculate the danger level for enemy at each position
 * 
 * @param missiles_x X-coordinates of missiles
 * @param missiles_y Y-coordinates of missiles
 * @param missiles_vx X velocities of missiles
 * @param missiles_vy Y velocities of missiles
 * @param missile_count Number of missiles
 * @param grid_positions_x X-coordinates of grid points
 * @param grid_positions_y Y-coordinates of grid points
 * @param grid_width Width of the grid
 * @param grid_height Height of the grid
 * @param danger_values Output danger values for each grid point
 */
void cuda_calculate_danger_map(
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    const float* grid_positions_x,
    const float* grid_positions_y,
    int grid_width,
    int grid_height,
    float* danger_values
);

/**
 * Calculate distances between entities in parallel on the GPU
 * 
 * @param entities_a_x X-coordinates of first entity set
 * @param entities_a_y Y-coordinates of first entity set
 * @param entities_a_count Number of entities in first set
 * @param entities_b_x X-coordinates of second entity set
 * @param entities_b_y Y-coordinates of second entity set
 * @param entities_b_count Number of entities in second set
 * @param distance_matrix Output distance matrix (a_count x b_count)
 */
void cuda_calculate_distances(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
);

/**
 * Calculate optimal evasion vectors for enemies based on missile positions and velocities
 * 
 * @param enemy_x X-coordinate of enemy
 * @param enemy_y Y-coordinate of enemy
 * @param missiles_x X-coordinates of missiles
 * @param missiles_y Y-coordinates of missiles
 * @param missiles_vx X velocities of missiles
 * @param missiles_vy Y velocities of missiles
 * @param missile_count Number of missiles
 * @param prediction_steps Number of steps to predict missile trajectories
 * @param evasion_vector_x Output X component of evasion vector
 * @param evasion_vector_y Output Y component of evasion vector
 */
#pragma once

#include <vector>
#include <memory>
#include "entity.h"

namespace gpu_env {

// Vector structure
struct Vector2 {
    float x;
    float y;
};

// C++ interface for physics operations
class PhysicsEngine {
public:
    PhysicsEngine(float screen_width, float screen_height);
    ~PhysicsEngine();

    // Initialize the physics engine
    void initialize();

    // Update positions of all entities
    void update_positions(
        EntityBatch& entities,
        const std::vector<float>& velocities_x,
        const std::vector<float>& velocities_y
    );

    // Detect collisions between two entity batches
    std::vector<std::vector<bool>> detect_collisions(
        const EntityBatch& entities_a,
        const EntityBatch& entities_b
    );

    // Calculate danger map for the entire screen
    std::vector<float> calculate_danger_map(
        const MissileBatch& missiles,
        int grid_width,
        int grid_height
    );

    // Calculate optimal evasion vector for an enemy
    std::pair<float, float> calculate_evasion_vector(
        float enemy_x,
        float enemy_y,
        const MissileBatch& missiles,
        int prediction_steps = 30
    );

private:
    float screen_width_;
    float screen_height_;
    bool initialized_;
};

} // namespace gpu_env


// C++ interface for physics operations
class PhysicsEngine {
public:
    PhysicsEngine(float screen_width, float screen_height);
    ~PhysicsEngine();

    // Initialize the physics engine
    void initialize();

    // Update positions of all entities
    void update_positions(
        EntityBatch& entities,
        const std::vector<float>& velocities_x,
        const std::vector<float>& velocities_y
    );

    // Detect collisions between two entity batches
    std::vector<std::vector<bool>> detect_collisions(
        const EntityBatch& entities_a,
        const EntityBatch& entities_b
    );

    // Calculate danger map for the entire screen
    std::vector<float> calculate_danger_map(
        const MissileBatch& missiles,
        int grid_width,
        int grid_height
    );

    // Calculate optimal evasion vector for an enemy
    std::pair<float, float> calculate_evasion_vector(
        float enemy_x,
        float enemy_y,
        const MissileBatch& missiles,
        int prediction_steps = 30
    );

private:
    float screen_width_;
    float screen_height_;
    bool initialized_;

    // CUDA memory management
    void allocate_device_memory(size_t size);
    void free_device_memory();

    // Device memory pointers
    float* d_temp_float_array1_;
    float* d_temp_float_array2_;
    bool* d_temp_bool_array_;
    size_t d_temp_array_size_;
};

} // namespace gpu_env
