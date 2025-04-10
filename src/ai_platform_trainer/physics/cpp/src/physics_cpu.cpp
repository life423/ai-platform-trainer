#include "../include/physics.h"
#include <cmath>
#include <algorithm>

namespace gpu_env {

// CPU implementations of physics functions that match the CUDA interface
// These will be used when CUDA is not available

void update_positions_cpu(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    for (int i = 0; i < count; ++i) {
        // Update position
        entities_x[i] += velocities_x[i];
        entities_y[i] += velocities_y[i];

        // Wrap around screen boundaries
        if (entities_x[i] < 0) {
            entities_x[i] += screen_width;
        } else if (entities_x[i] >= screen_width) {
            entities_x[i] -= screen_width;
        }

        if (entities_y[i] < 0) {
            entities_y[i] += screen_height;
        } else if (entities_y[i] >= screen_height) {
            entities_y[i] -= screen_height;
        }
    }
}

void detect_collisions_cpu(
    const float* entities_a_x,
    const float* entities_a_y,
    const float* entities_a_sizes,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    const float* entities_b_sizes,
    int entities_b_count,
    bool* collision_matrix
) {
    for (int a = 0; a < entities_a_count; ++a) {
        for (int b = 0; b < entities_b_count; ++b) {
            // Calculate Euclidean distance between entity centers
            const float dx = entities_a_x[a] - entities_b_x[b];
            const float dy = entities_a_y[a] - entities_b_y[b];
            const float distance_squared = dx * dx + dy * dy;
            
            // Calculate minimum distance for collision
            const float min_distance = (entities_a_sizes[a] + entities_b_sizes[b]) * 0.5f;
            const float min_distance_squared = min_distance * min_distance;
            
            // Check if entities are overlapping
            collision_matrix[a * entities_b_count + b] = distance_squared <= min_distance_squared;
        }
    }
}

void calculate_distances_cpu(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
) {
    for (int a = 0; a < entities_a_count; ++a) {
        for (int b = 0; b < entities_b_count; ++b) {
            // Calculate Euclidean distance between entity centers
            const float dx = entities_a_x[a] - entities_b_x[b];
            const float dy = entities_a_y[a] - entities_b_y[b];
            const float distance = std::sqrt(dx * dx + dy * dy);
            
            // Store distance in matrix
            distance_matrix[a * entities_b_count + b] = distance;
        }
    }
}

void calculate_danger_map_cpu(
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    const float* grid_positions_x,
    const float* grid_positions_y,
    int grid_width,
    int grid_height,
    float* danger_values,
    int prediction_steps = 20,
    float danger_radius = 150.0f
) {
    // For each grid position
    for (int y = 0; y < grid_height; ++y) {
        for (int x = 0; x < grid_width; ++x) {
            const int grid_idx = y * grid_width + x;
            const float grid_x = grid_positions_x[x];
            const float grid_y = grid_positions_y[y];
            
            float total_danger = 0.0f;
            
            // For each missile
            for (int m = 0; m < missile_count; ++m) {
                // Start with missile's current position
                float missile_x = missiles_x[m];
                float missile_y = missiles_y[m];
                const float missile_vx = missiles_vx[m];
                const float missile_vy = missiles_vy[m];
                
                // Predict trajectory and find closest approach
                float min_distance_squared = std::numeric_limits<float>::max();
                
                for (int step = 0; step < prediction_steps; ++step) {
                    // Move missile forward
                    missile_x += missile_vx;
                    missile_y += missile_vy;
                    
                    // Calculate distance to grid position
                    const float dx = grid_x - missile_x;
                    const float dy = grid_y - missile_y;
                    const float distance_squared = dx * dx + dy * dy;
                    
                    // Track minimum distance
                    min_distance_squared = std::min(min_distance_squared, distance_squared);
                }
                
                // Convert to danger value
                const float min_distance = std::sqrt(min_distance_squared);
                if (min_distance < danger_radius) {
                    // Danger increases as distance decreases
                    const float danger = 1.0f - (min_distance / danger_radius);
                    total_danger += danger;
                }
            }
            
            // Store calculated danger value
            danger_values[grid_idx] = total_danger;
        }
    }
}

void calculate_evasion_vector_cpu(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float* evasion_vector_x,
    float* evasion_vector_y,
    float missile_danger_radius = 150.0f
) {
    float total_evasion_x = 0.0f;
    float total_evasion_y = 0.0f;
    float total_weight = 0.0f;
    
    // For each missile
    for (int m = 0; m < missile_count; ++m) {
        // Start with missile's current position
        float missile_x = missiles_x[m];
        float missile_y = missiles_y[m];
        const float missile_vx = missiles_vx[m];
        const float missile_vy = missiles_vy[m];
        
        // Calculate initial distance and direction to missile
        float dx = missile_x - enemy_x;
        float dy = missile_y - enemy_y;
        float distance = std::sqrt(dx * dx + dy * dy);
        
        // Find closest predicted position
        float min_distance = distance;
        float closest_x = missile_x;
        float closest_y = missile_y;
        bool will_hit = false;
        
        for (int step = 0; step < prediction_steps; ++step) {
            // Move missile forward
            missile_x += missile_vx;
            missile_y += missile_vy;
            
            // Calculate new distance
            dx = missile_x - enemy_x;
            dy = missile_y - enemy_y;
            distance = std::sqrt(dx * dx + dy * dy);
            
            // Check if this is the closest position
            if (distance < min_distance) {
                min_distance = distance;
                closest_x = missile_x;
                closest_y = missile_y;
                
                // Check if missile will hit enemy
                if (distance < missile_danger_radius * 0.5f) {
                    will_hit = true;
                }
            }
        }
        
        // If missile is close enough to be dangerous
        if (min_distance < missile_danger_radius) {
            // Calculate danger weight (higher for closer missiles)
            float danger = 1.0f - (min_distance / missile_danger_radius);
            
            // Apply higher weight if missile will hit
            if (will_hit) {
                danger *= 2.0f;
            }
            
            // Calculate evasion direction (away from closest predicted position)
            dx = enemy_x - closest_x;
            dy = enemy_y - closest_y;
            
            // Normalize direction
            const float length = std::sqrt(dx * dx + dy * dy);
            if (length > 0.0001f) {
                dx /= length;
                dy /= length;
            }
            
            // Add weighted contribution to total evasion vector
            total_evasion_x += dx * danger;
            total_evasion_y += dy * danger;
            total_weight += danger;
        }
    }
    
    // Calculate final evasion vector
    if (total_weight > 0.0001f) {
        // Normalize by total weight
        *evasion_vector_x = total_evasion_x / total_weight;
        *evasion_vector_y = total_evasion_y / total_weight;
    } else {
        // No danger, no evasion
        *evasion_vector_x = 0.0f;
        *evasion_vector_y = 0.0f;
    }
}

// CUDA launcher functions that dispatch to either CUDA or CPU implementations
// These are linked in depending on whether CUDA support is available

#ifdef USE_CUDA

// If CUDA is available, these are just pass-through to the CUDA implementations
// (the actual CUDA implementations are in physics.cu)

void cuda_update_positions(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
);

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

void cuda_calculate_distances(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
);

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

void cuda_calculate_evasion_vector(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float* evasion_vector_x,
    float* evasion_vector_y
);

#else

// If CUDA is not available, these call the CPU implementations

void cuda_update_positions(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    update_positions_cpu(
        entities_x, entities_y,
        velocities_x, velocities_y,
        count, screen_width, screen_height
    );
}

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
) {
    detect_collisions_cpu(
        entities_a_x, entities_a_y, entities_a_sizes, entities_a_count,
        entities_b_x, entities_b_y, entities_b_sizes, entities_b_count,
        collision_matrix
    );
}

void cuda_calculate_distances(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
) {
    calculate_distances_cpu(
        entities_a_x, entities_a_y, entities_a_count,
        entities_b_x, entities_b_y, entities_b_count,
        distance_matrix
    );
}

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
) {
    calculate_danger_map_cpu(
        missiles_x, missiles_y, missiles_vx, missiles_vy, missile_count,
        grid_positions_x, grid_positions_y, grid_width, grid_height,
        danger_values
    );
}

void cuda_calculate_evasion_vector(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float* evasion_vector_x,
    float* evasion_vector_y
) {
    calculate_evasion_vector_cpu(
        enemy_x, enemy_y,
        missiles_x, missiles_y, missiles_vx, missiles_vy,
        missile_count, prediction_steps,
        evasion_vector_x, evasion_vector_y
    );
}

#endif

// PhysicsEngine implementation

PhysicsEngine::PhysicsEngine(float screen_width, float screen_height)
    : screen_width_(screen_width)
    , screen_height_(screen_height)
    , initialized_(false)
    , d_temp_float_array1_(nullptr)
    , d_temp_float_array2_(nullptr)
    , d_temp_bool_array_(nullptr)
    , d_temp_array_size_(0)
{
}

PhysicsEngine::~PhysicsEngine() {
    if (initialized_) {
        free_device_memory();
    }
}

void PhysicsEngine::initialize() {
    if (!initialized_) {
        #ifdef USE_CUDA
        // Initialize CUDA if available
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            // Choose the best GPU device
            int device_id = 0;
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, device_id);
            
            printf("Using GPU device %d: %s\n", device_id, device_prop.name);
            cudaSetDevice(device_id);
            
            // Allocate some initial device memory
            allocate_device_memory(1024);
        } else {
            printf("No CUDA-capable devices found. Using CPU implementation.\n");
        }
        #else
        printf("Built without CUDA support. Using CPU implementation.\n");
        #endif
        
        initialized_ = true;
    }
}

void PhysicsEngine::update_positions(
    EntityBatch& entities,
    const std::vector<float>& velocities_x,
    const std::vector<float>& velocities_y
) {
    if (!initialized_) {
        initialize();
    }
    
    const size_t count = entities.size();
    if (count == 0) {
        return;
    }

    #ifdef USE_CUDA
    // Ensure we have enough device memory
    if (count > d_temp_array_size_) {
        allocate_device_memory(count);
    }
    
    // Copy entity data to device
    cudaMemcpy(d_temp_float_array1_, entities.x.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_temp_float_array2_, entities.y.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy velocity data to device
    float* d_velocities_x;
    float* d_velocities_y;
    cudaMalloc(&d_velocities_x, count * sizeof(float));
    cudaMalloc(&d_velocities_y, count * sizeof(float));
    cudaMemcpy(d_velocities_x, velocities_x.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocities_y, velocities_y.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call the CUDA kernel
    cuda_update_positions(
        d_temp_float_array1_,
        d_temp_float_array2_,
        d_velocities_x,
        d_velocities_y,
        static_cast<int>(count),
        screen_width_,
        screen_height_
    );
    
    // Copy updated positions back to host
    cudaMemcpy(entities.x.data(), d_temp_float_array1_, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(entities.y.data(), d_temp_float_array2_, count * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_velocities_x);
    cudaFree(d_velocities_y);
    #else
    // Use CPU implementation
    cuda_update_positions(
        entities.x.data(), entities.y.data(),
        velocities_x.data(), velocities_y.data(),
        static_cast<int>(count),
        screen_width_,
        screen_height_
    );
    #endif
}

std::vector<std::vector<bool>> PhysicsEngine::detect_collisions(
    const EntityBatch& entities_a,
    const EntityBatch& entities_b
) {
    if (!initialized_) {
        initialize();
    }
    
    const size_t a_count = entities_a.size();
    const size_t b_count = entities_b.size();
    
    // Prepare result matrix
    std::vector<std::vector<bool>> collision_matrix(a_count, std::vector<bool>(b_count, false));
    
    if (a_count == 0 || b_count == 0) {
        return collision_matrix;
    }
    
    #ifdef USE_CUDA
    // Ensure we have enough device memory
    const size_t required_memory = std::max(a_count, b_count) * 3 + a_count * b_count;
    if (required_memory > d_temp_array_size_) {
        allocate_device_memory(required_memory);
    }
    
    // Allocate and copy entity data to device
    float* d_a_x; float* d_a_y; float* d_a_sizes;
    float* d_b_x; float* d_b_y; float* d_b_sizes;
    bool* d_collision_matrix;
    
    cudaMalloc(&d_a_x, a_count * sizeof(float));
    cudaMalloc(&d_a_y, a_count * sizeof(float));
    cudaMalloc(&d_a_sizes, a_count * sizeof(float));
    cudaMalloc(&d_b_x, b_count * sizeof(float));
    cudaMalloc(&d_b_y, b_count * sizeof(float));
    cudaMalloc(&d_b_sizes, b_count * sizeof(float));
    cudaMalloc(&d_collision_matrix, a_count * b_count * sizeof(bool));
    
    cudaMemcpy(d_a_x, entities_a.x.data(), a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_y, entities_a.y.data(), a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_sizes, entities_a.sizes.data(), a_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_x, entities_b.x.data(), b_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_y, entities_b.y.data(), b_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sizes, entities_b.sizes.data(), b_count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Clear the collision matrix
    cudaMemset(d_collision_matrix, 0, a_count * b_count * sizeof(bool));
    
    // Call the CUDA kernel
    cuda_detect_collisions(
        d_a_x, d_a_y, d_a_sizes, static_cast<int>(a_count),
        d_b_x, d_b_y, d_b_sizes, static_cast<int>(b_count),
        d_collision_matrix
    );
    
    // Allocate host memory for the collision matrix
    std::vector<bool> flat_collision_matrix(a_count * b_count);
    
    // Copy collision matrix back to host
    cudaMemcpy(flat_collision_matrix.data(), d_collision_matrix, 
              a_count * b_count * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // Convert flat matrix to 2D vector
    for (size_t a = 0; a < a_count; ++a) {
        for (size_t b = 0; b < b_count; ++b) {
            collision_matrix[a][b] = flat_collision_matrix[a * b_count + b];
        }
    }
    
    // Clean up
    cudaFree(d_a_x);
    cudaFree(d_a_y);
    cudaFree(d_a_sizes);
    cudaFree(d_b_x);
    cudaFree(d_b_y);
    cudaFree(d_b_sizes);
    cudaFree(d_collision_matrix);
    #else
    // Allocate flat collision matrix for the CPU implementation
    // Note: std::vector<bool> doesn't have a data() method, so we use a regular array
    bool* flat_collision_matrix = new bool[a_count * b_count]();
    
    // Use CPU implementation
    cuda_detect_collisions(
        entities_a.x.data(), entities_a.y.data(), entities_a.sizes.data(), static_cast<int>(a_count),
        entities_b.x.data(), entities_b.y.data(), entities_b.sizes.data(), static_cast<int>(b_count),
        flat_collision_matrix
    );
    
    // Convert flat matrix to 2D vector
    for (size_t a = 0; a < a_count; ++a) {
        for (size_t b = 0; b < b_count; ++b) {
            collision_matrix[a][b] = flat_collision_matrix[a * b_count + b];
        }
    }
    #endif
    
    return collision_matrix;
}

std::vector<float> PhysicsEngine::calculate_danger_map(
    const MissileBatch& missiles,
    int grid_width,
    int grid_height
) {
    if (!initialized_) {
        initialize();
    }
    
    const size_t missile_count = missiles.size();
    const size_t grid_size = grid_width * grid_height;
    
    // Prepare result vector
    std::vector<float> danger_map(grid_size, 0.0f);
    
    if (missile_count == 0 || grid_size == 0) {
        return danger_map;
    }
    
    // Calculate grid positions
    std::vector<float> grid_positions_x(grid_width);
    std::vector<float> grid_positions_y(grid_height);
    
    const float cell_width = screen_width_ / static_cast<float>(grid_width);
    const float cell_height = screen_height_ / static_cast<float>(grid_height);
    
    for (int x = 0; x < grid_width; ++x) {
        grid_positions_x[x] = x * cell_width + cell_width * 0.5f;
    }
    
    for (int y = 0; y < grid_height; ++y) {
        grid_positions_y[y] = y * cell_height + cell_height * 0.5f;
    }
    
    #ifdef USE_CUDA
    // Allocate device memory
    float* d_missiles_x; float* d_missiles_y; 
    float* d_missiles_vx; float* d_missiles_vy;
    float* d_grid_positions_x; float* d_grid_positions_y;
    float* d_danger_values;
    
    cudaMalloc(&d_missiles_x, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_y, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_vx, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_vy, missile_count * sizeof(float));
    cudaMalloc(&d_grid_positions_x, grid_width * sizeof(float));
    cudaMalloc(&d_grid_positions_y, grid_height * sizeof(float));
    cudaMalloc(&d_danger_values, grid_size * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_missiles_x, missiles.x.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_y, missiles.y.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_vx, missiles.vx.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_vy, missiles.vy.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_positions_x, grid_positions_x.data(), grid_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_positions_y, grid_positions_y.data(), grid_height * sizeof(float), cudaMemcpyHostToDevice);
    
    // Clear the danger values
    cudaMemset(d_danger_values, 0, grid_size * sizeof(float));
    
    // Call the CUDA kernel
    cuda_calculate_danger_map(
        d_missiles_x, d_missiles_y, d_missiles_vx, d_missiles_vy, static_cast<int>(missile_count),
        d_grid_positions_x, d_grid_positions_y, grid_width, grid_height,
        d_danger_values
    );
    
    // Copy danger values back to host
    cudaMemcpy(danger_map.data(), d_danger_values, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_missiles_x);
    cudaFree(d_missiles_y);
    cudaFree(d_missiles_vx);
    cudaFree(d_missiles_vy);
    cudaFree(d_grid_positions_x);
    cudaFree(d_grid_positions_y);
    cudaFree(d_danger_values);
    #else
    // Use CPU implementation
    cuda_calculate_danger_map(
        missiles.x.data(), missiles.y.data(), missiles.vx.data(), missiles.vy.data(), 
        static_cast<int>(missile_count),
        grid_positions_x.data(), grid_positions_y.data(), grid_width, grid_height,
        danger_map.data()
    );
    #endif
    
    return danger_map;
}

std::pair<float, float> PhysicsEngine::calculate_evasion_vector(
    float enemy_x,
    float enemy_y,
    const MissileBatch& missiles,
    int prediction_steps
) {
    if (!initialized_) {
        initialize();
    }
    
    float evasion_x = 0.0f;
    float evasion_y = 0.0f;
    
    const size_t missile_count = missiles.size();
    if (missile_count == 0) {
        return std::make_pair(evasion_x, evasion_y);
    }
    
    #ifdef USE_CUDA
    // Allocate device memory
    float* d_missiles_x; float* d_missiles_y; 
    float* d_missiles_vx; float* d_missiles_vy;
    
    cudaMalloc(&d_missiles_x, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_y, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_vx, missile_count * sizeof(float));
    cudaMalloc(&d_missiles_vy, missile_count * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_missiles_x, missiles.x.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_y, missiles.y.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_vx, missiles.vx.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_missiles_vy, missiles.vy.data(), missile_count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call the CUDA kernel
    cuda_calculate_evasion_vector(
        enemy_x, enemy_y,
        d_missiles_x, d_missiles_y, d_missiles_vx, d_missiles_vy,
        static_cast<int>(missile_count), prediction_steps,
        &evasion_x, &evasion_y
    );
    
    // Clean up
    cudaFree(d_missiles_x);
    cudaFree(d_missiles_y);
    cudaFree(d_missiles_vx);
    cudaFree(d_missiles_vy);
    #else
    // Use CPU implementation
    cuda_calculate_evasion_vector(
        enemy_x, enemy_y,
        missiles.x.data(), missiles.y.data(), missiles.vx.data(), missiles.vy.data(),
        static_cast<int>(missile_count), prediction_steps,
        &evasion_x, &evasion_y
    );
    #endif
    
    return std::make_pair(evasion_x, evasion_y);
}

void PhysicsEngine::allocate_device_memory(size_t size) {
    #ifdef USE_CUDA
    // Free existing memory if any
    free_device_memory();
    
    // Allocate new memory
    cudaMalloc(&d_temp_float_array1_, size * sizeof(float));
    cudaMalloc(&d_temp_float_array2_, size * sizeof(float));
    cudaMalloc(&d_temp_bool_array_, size * sizeof(bool));
    
    d_temp_array_size_ = size;
    #endif
}

void PhysicsEngine::free_device_memory() {
    #ifdef USE_CUDA
    if (d_temp_float_array1_ != nullptr) {
        cudaFree(d_temp_float_array1_);
        d_temp_float_array1_ = nullptr;
    }
    
    if (d_temp_float_array2_ != nullptr) {
        cudaFree(d_temp_float_array2_);
        d_temp_float_array2_ = nullptr;
    }
    
    if (d_temp_bool_array_ != nullptr) {
        cudaFree(d_temp_bool_array_);
        d_temp_bool_array_ = nullptr;
    }
    
    d_temp_array_size_ = 0;
    #endif
}

} // namespace gpu_env
