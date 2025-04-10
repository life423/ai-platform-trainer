#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../include/physics.h"

namespace gpu_env {

// CUDA error checking helper macro
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// CUDA kernels

/**
 * CUDA kernel for updating entity positions based on velocities and wrapping
 * around screen boundaries.
 */
__global__ void update_positions_kernel(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we're within bounds
    if (idx < count) {
        // Update position
        entities_x[idx] += velocities_x[idx];
        entities_y[idx] += velocities_y[idx];

        // Wrap around screen boundaries
        if (entities_x[idx] < 0) {
            entities_x[idx] += screen_width;
        } else if (entities_x[idx] >= screen_width) {
            entities_x[idx] -= screen_width;
        }

        if (entities_y[idx] < 0) {
            entities_y[idx] += screen_height;
        } else if (entities_y[idx] >= screen_height) {
            entities_y[idx] -= screen_height;
        }
    }
}

/**
 * CUDA kernel for detecting collisions between two sets of entities.
 * Each thread handles one entity pair comparison.
 */
__global__ void detect_collisions_kernel(
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
    // Calculate global thread ID for entity pair (a_idx, b_idx)
    const int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we're within bounds
    if (a_idx < entities_a_count && b_idx < entities_b_count) {
        // Get entity A position and size
        const float a_x = entities_a_x[a_idx];
        const float a_y = entities_a_y[a_idx];
        const float a_size = entities_a_sizes[a_idx];

        // Get entity B position and size
        const float b_x = entities_b_x[b_idx];
        const float b_y = entities_b_y[b_idx];
        const float b_size = entities_b_sizes[b_idx];

        // Calculate distance between entities
        const float dx = a_x - b_x;
        const float dy = a_y - b_y;
        const float distance_squared = dx * dx + dy * dy;

        // Calculate minimum distance for collision
        const float min_distance = (a_size + b_size) * 0.5f;
        const float min_distance_squared = min_distance * min_distance;

        // Check if collision occurred
        const bool collision = distance_squared <= min_distance_squared;

        // Store collision result in matrix
        collision_matrix[a_idx * entities_b_count + b_idx] = collision;
    }
}

/**
 * CUDA kernel for calculating distances between two sets of entities.
 * Each thread handles one entity pair comparison.
 */
__global__ void calculate_distances_kernel(
    const float* entities_a_x,
    const float* entities_a_y,
    int entities_a_count,
    const float* entities_b_x,
    const float* entities_b_y,
    int entities_b_count,
    float* distance_matrix
) {
    // Calculate global thread ID for entity pair (a_idx, b_idx)
    const int a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int b_idx = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we're within bounds
    if (a_idx < entities_a_count && b_idx < entities_b_count) {
        // Get entity A position
        const float a_x = entities_a_x[a_idx];
        const float a_y = entities_a_y[a_idx];

        // Get entity B position
        const float b_x = entities_b_x[b_idx];
        const float b_y = entities_b_y[b_idx];

        // Calculate distance between entities
        const float dx = a_x - b_x;
        const float dy = a_y - b_y;
        const float distance = sqrtf(dx * dx + dy * dy);

        // Store distance in matrix
        distance_matrix[a_idx * entities_b_count + b_idx] = distance;
    }
}

/**
 * CUDA kernel for predicting missile trajectories and calculating danger values
 * for each position on a grid.
 */
__global__ void calculate_danger_map_kernel(
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
    int prediction_steps,
    float danger_radius
) {
    // Calculate grid position index
    const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int grid_idx = y_idx * grid_width + x_idx;

    // Ensure we're within grid bounds
    if (x_idx < grid_width && y_idx < grid_height) {
        // Get grid position
        const float grid_x = grid_positions_x[x_idx];
        const float grid_y = grid_positions_y[y_idx];
        
        // Calculate danger value for this grid position
        float total_danger = 0.0f;
        
        // For each missile
        for (int m = 0; m < missile_count; ++m) {
            float missile_x = missiles_x[m];
            float missile_y = missiles_y[m];
            const float missile_vx = missiles_vx[m];
            const float missile_vy = missiles_vy[m];
            
            // Simulate missile trajectory for several steps
            float min_distance_squared = FLT_MAX;
            
            for (int step = 0; step < prediction_steps; ++step) {
                // Move missile
                missile_x += missile_vx;
                missile_y += missile_vy;
                
                // Calculate distance to grid position
                const float dx = grid_x - missile_x;
                const float dy = grid_y - missile_y;
                const float distance_squared = dx * dx + dy * dy;
                
                // Keep track of minimum distance
                min_distance_squared = min(min_distance_squared, distance_squared);
            }
            
            // Convert minimum distance to danger value
            const float min_distance = sqrt(min_distance_squared);
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

/**
 * CUDA kernel for calculating optimal evasion vector for an enemy based on
 * missile positions and velocities.
 * This kernel is designed for a single enemy but can process multiple missiles in parallel.
 */
__global__ void calculate_evasion_vector_kernel(
    float enemy_x,
    float enemy_y,
    const float* missiles_x,
    const float* missiles_y,
    const float* missiles_vx,
    const float* missiles_vy,
    int missile_count,
    int prediction_steps,
    float missile_danger_radius,
    float* missile_dangers,
    float2* evasion_vectors
) {
    // Each thread handles one missile
    const int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory for reduction
    __shared__ float s_evasion_x[256];
    __shared__ float s_evasion_y[256];
    __shared__ float s_danger_weights[256];
    
    // Initialize this thread's values
    float evasion_x = 0.0f;
    float evasion_y = 0.0f;
    float danger = 0.0f;
    
    // Calculate evasion vector and danger for this missile
    if (m_idx < missile_count) {
        // Get missile data
        float missile_x = missiles_x[m_idx];
        float missile_y = missiles_y[m_idx];
        const float missile_vx = missiles_vx[m_idx];
        const float missile_vy = missiles_vy[m_idx];
        
        // Calculate initial distance to missile
        float dx = missile_x - enemy_x;
        float dy = missile_y - enemy_y;
        float distance = sqrtf(dx * dx + dy * dy);
        
        // Find the closest predicted position
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
            distance = sqrtf(dx * dx + dy * dy);
            
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
        
        // Calculate danger based on minimum distance
        if (min_distance < missile_danger_radius) {
            danger = 1.0f - (min_distance / missile_danger_radius);
            
            // Apply higher weight if missile will hit
            if (will_hit) {
                danger *= 2.0f;
            }
            
            // Calculate evasion vector (away from closest predicted position)
            dx = enemy_x - closest_x;
            dy = enemy_y - closest_y;
            
            // Normalize
            const float length = sqrtf(dx * dx + dy * dy);
            if (length > 0.0001f) {
                evasion_x = dx / length;
                evasion_y = dy / length;
            }
            
            // Weight by danger
            evasion_x *= danger;
            evasion_y *= danger;
        }
        
        // Store results for reduction
        missile_dangers[m_idx] = danger;
        evasion_vectors[m_idx] = make_float2(evasion_x, evasion_y);
    }
    
    // Store in shared memory for reduction
    const int tid = threadIdx.x;
    s_evasion_x[tid] = evasion_x;
    s_evasion_y[tid] = evasion_y;
    s_danger_weights[tid] = danger;
    __syncthreads();
    
    // Reduction to calculate final evasion vector
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_evasion_x[tid] += s_evasion_x[tid + stride];
            s_evasion_y[tid] += s_evasion_y[tid + stride];
            s_danger_weights[tid] += s_danger_weights[tid + stride];
        }
        __syncthreads();
    }
    
    // Write final result from the first thread
    if (tid == 0) {
        const float total_danger = s_danger_weights[0];
        if (total_danger > 0.0001f) {
            // Normalize the final evasion vector
            const float final_evx = s_evasion_x[0] / total_danger;
            const float final_evy = s_evasion_y[0] / total_danger;
            
            evasion_vectors[missile_count] = make_float2(final_evx, final_evy);
        } else {
            evasion_vectors[missile_count] = make_float2(0.0f, 0.0f);
        }
    }
}

// Host-side launcher functions that interface with the kernels

void cuda_update_positions(
    float* entities_x,
    float* entities_y,
    const float* velocities_x,
    const float* velocities_y,
    int count,
    float screen_width,
    float screen_height
) {
    // Configure kernel launch
    const int block_size = 256;
    const int grid_size = (count + block_size - 1) / block_size;
    
    // Launch kernel
    update_positions_kernel<<<grid_size, block_size>>>(
        entities_x, entities_y,
        velocities_x, velocities_y,
        count, screen_width, screen_height
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Configure kernel launch
    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (entities_a_count + block_size.x - 1) / block_size.x,
        (entities_b_count + block_size.y - 1) / block_size.y
    );
    
    // Launch kernel
    detect_collisions_kernel<<<grid_size, block_size>>>(
        entities_a_x, entities_a_y, entities_a_sizes, entities_a_count,
        entities_b_x, entities_b_y, entities_b_sizes, entities_b_count,
        collision_matrix
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Configure kernel launch
    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (entities_a_count + block_size.x - 1) / block_size.x,
        (entities_b_count + block_size.y - 1) / block_size.y
    );
    
    // Launch kernel
    calculate_distances_kernel<<<grid_size, block_size>>>(
        entities_a_x, entities_a_y, entities_a_count,
        entities_b_x, entities_b_y, entities_b_count,
        distance_matrix
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Configure kernel launch
    const dim3 block_size(16, 16);
    const dim3 grid_size(
        (grid_width + block_size.x - 1) / block_size.x,
        (grid_height + block_size.y - 1) / block_size.y
    );
    
    // Define danger radius and prediction steps
    const float danger_radius = 150.0f;
    const int prediction_steps = 20;
    
    // Launch kernel
    calculate_danger_map_kernel<<<grid_size, block_size>>>(
        missiles_x, missiles_y, missiles_vx, missiles_vy, missile_count,
        grid_positions_x, grid_positions_y, grid_width, grid_height,
        danger_values, prediction_steps, danger_radius
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    // Allocate device memory for intermediate results
    float* d_missile_dangers;
    float2* d_evasion_vectors;
    CUDA_CHECK(cudaMalloc(&d_missile_dangers, missile_count * sizeof(float)));
    // Allocate one extra slot for the final result
    CUDA_CHECK(cudaMalloc(&d_evasion_vectors, (missile_count + 1) * sizeof(float2)));
    
    // Configure kernel launch
    const int block_size = min(256, missile_count);
    const int grid_size = (missile_count + block_size - 1) / block_size;
    
    // Define danger radius
    const float missile_danger_radius = 150.0f;
    
    // Launch kernel
    calculate_evasion_vector_kernel<<<grid_size, block_size>>>(
        enemy_x, enemy_y,
        missiles_x, missiles_y, missiles_vx, missiles_vy,
        missile_count, prediction_steps, missile_danger_radius,
        d_missile_dangers, d_evasion_vectors
    );
    
    // Check for errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy final evasion vector back to host
    float2 result;
    CUDA_CHECK(cudaMemcpy(&result, &d_evasion_vectors[missile_count], sizeof(float2), cudaMemcpyDeviceToHost));
    
    // Clean up device memory
    CUDA_CHECK(cudaFree(d_missile_dangers));
    CUDA_CHECK(cudaFree(d_evasion_vectors));
    
    // Set the output values
    *evasion_vector_x = result.x;
    *evasion_vector_y = result.y;
}

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
        // Initialize CUDA
        int device_count;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            fprintf(stderr, "No CUDA-capable device found.\n");
            exit(EXIT_FAILURE);
        }
        
        // Choose the best GPU device
        int device_id = 0;
        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        
        printf("Using GPU device %d: %s\n", device_id, device_prop.name);
        CUDA_CHECK(cudaSetDevice(device_id));
        
        // Allocate some initial device memory
        allocate_device_memory(1024);
        
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
    
    // Ensure we have enough device memory
    if (count > d_temp_array_size_) {
        allocate_device_memory(count);
    }
    
    // Copy entity data to device
    CUDA_CHECK(cudaMemcpy(d_temp_float_array1_, entities.x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temp_float_array2_, entities.y.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create device vectors for velocities
    thrust::device_vector<float> d_velocities_x(velocities_x.begin(), velocities_x.end());
    thrust::device_vector<float> d_velocities_y(velocities_y.begin(), velocities_y.end());
    
    // Call the CUDA kernel launcher
    cuda_update_positions(
        d_temp_float_array1_,
        d_temp_float_array2_,
        thrust::raw_pointer_cast(d_velocities_x.data()),
        thrust::raw_pointer_cast(d_velocities_y.data()),
        static_cast<int>(count),
        screen_width_,
        screen_height_
    );
    
    // Copy updated positions back to host
    CUDA_CHECK(cudaMemcpy(entities.x.data(), d_temp_float_array1_, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(entities.y.data(), d_temp_float_array2_, count * sizeof(float), cudaMemcpyDeviceToHost));
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
    
    // Ensure we have enough device memory for all temporary arrays
    const size_t required_memory = std::max(a_count, b_count) * 3 + a_count * b_count;
    if (required_memory > d_temp_array_size_) {
        allocate_device_memory(required_memory);
    }
    
    // Create device vectors for entity data
    thrust::device_vector<float> d_a_x(entities_a.x.begin(), entities_a.x.end());
    thrust::device_vector<float> d_a_y(entities_a.y.begin(), entities_a.y.end());
    thrust::device_vector<float> d_a_sizes(entities_a.sizes.begin(), entities_a.sizes.end());
    
    thrust::device_vector<float> d_b_x(entities_b.x.begin(), entities_b.x.end());
    thrust::device_vector<float> d_b_y(entities_b.y.begin(), entities_b.y.end());
    thrust::device_vector<float> d_b_sizes(entities_b.sizes.begin(), entities_b.sizes.end());
    
    // Allocate device memory for collision matrix
    const size_t collision_matrix_size = a_count * b_count;
    thrust::device_vector<bool> d_collision_matrix(collision_matrix_size, false);
    
    // Call the CUDA kernel launcher
    cuda_detect_collisions(
        thrust::raw_pointer_cast(d_a_x.data()),
        thrust::raw_pointer_cast(d_a_y.data()),
        thrust::raw_pointer_cast(d_a_sizes.data()),
        static_cast<int>(a_count),
        thrust::raw_pointer_cast(d_b_x.data()),
        thrust::raw_pointer_cast(d_b_y.data()),
        thrust::raw_pointer_cast(d_b_sizes.data()),
        static_cast<int>(b_count),
        thrust::raw_pointer_cast(d_collision_matrix.data())
    );
    
    // Copy collision matrix back to host and convert to 2D vector
    thrust::host_vector<bool> h_collision_matrix = d_collision_matrix;
    
    for (size_t a = 0; a < a_count; ++a) {
        for (size_t b = 0; b < b_count; ++b) {
            collision_matrix[a][b] = h_collision_matrix[a * b_count + b];
        }
    }
    
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
    
    // Ensure we have enough device memory
    const size_t required_memory = missile_count * 4 + grid_width + grid_height + grid_size;
    if (required_memory > d_temp_array_size_) {
        allocate_device_memory(required_memory);
    }
    
    // Create device vectors for missile data
    thrust::device_vector<float> d_missiles_x(missiles.x.begin(), missiles.x.end());
    thrust::device_vector<float> d_missiles_y(missiles.y.begin(), missiles.y.end());
    thrust::device_vector<float> d_missiles_vx(missiles.vx.begin(), missiles.vx.end());
    thrust::device_vector<float> d_missiles_vy(missiles.vy.begin(), missiles.vy.end());
    
    // Create device vectors for grid positions
    thrust::device_vector<float> d_grid_positions_x(grid_positions_x.begin(), grid_positions_x.end());
    thrust::device_vector<float> d_grid_positions_y(grid_positions_y.begin(), grid_positions_y.end());
    
    // Create device vector for danger values
    thrust::device_vector<float> d_danger_values(grid_size, 0.0f);
    
    // Call the CUDA kernel launcher
    cuda_calculate_danger_map(
        thrust::raw_pointer_cast(d_missiles_x.data()),
        thrust::raw_pointer_cast(d_missiles_y.data()),
        thrust::raw_pointer_cast(d_missiles_vx.data()),
        thrust::raw_pointer_cast(d_missiles_vy.data()),
        static_cast<int>(missile_count),
        thrust::raw_pointer_cast(d_grid_positions_x.data()),
        thrust::raw_pointer_cast(d_grid_positions_y.data()),
        grid_width,
        grid_height,
        thrust::raw_pointer_cast(d_danger_values.data())
    );
    
    // Copy danger values back to host
    thrust::host_vector<float> h_danger_values = d_danger_values;
    danger_map.assign(h_danger_values.begin(), h_danger_values.end());
    
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
    
    const size_t missile_count = missiles.size();
    
    // Default result if no missiles
    if (missile_count == 0) {
        return std::make_pair(0.0f, 0.0f);
    }
    
    // Ensure we have enough device memory
    const size_t required_memory = missile_count * 4;
    if (required_memory > d_temp_array_size_) {
        allocate_device_memory(required_memory);
    }
    
    // Create device vectors for missile data
    thrust::device_vector<float> d_missiles_x(missiles.x.begin(), missiles.x.end());
    thrust::device_vector<float> d_missiles_y(missiles.y.begin(), missiles.y.end());
