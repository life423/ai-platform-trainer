#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <tuple>
#include <string>

#include "../include/environment.h"
#include "../include/entity.h"

namespace py = pybind11;
using namespace gpu_env;

// Wrapper class for Python with helpful convenience methods
class PyEnvironment {
public:
    PyEnvironment(const EnvironmentConfig& config = EnvironmentConfig())
        : env_(std::make_unique<Environment>(config)) {}

    // Reset the environment and return the initial observation
    py::array_t<float> reset(unsigned int seed = 0) {
        std::vector<float> obs = env_->reset(seed);
        return vector_to_numpy(obs);
    }

    // Step the environment and return (observation, reward, done, info)
    py::tuple step(py::array_t<float> action) {
        // Convert numpy action to vector
        auto action_vec = numpy_to_vector<float>(action);
        
        // Call the C++ step method
        auto [obs, reward, done, truncated, info] = env_->step(action_vec);
        
        // Convert observation to numpy array
        py::array_t<float> obs_array = vector_to_numpy(obs);
        
        // Convert info map to Python dict
        py::dict info_dict;
        for (const auto& [key, value] : info) {
            info_dict[key.c_str()] = value;
        }
        
        return py::make_tuple(obs_array, reward, done, truncated, info_dict);
    }

    // Get observation space shape as a tuple
    py::tuple get_observation_shape() const {
        auto shape = env_->get_observation_shape();
        return vector_to_tuple(shape);
    }

    // Get action space shape as a tuple
    py::tuple get_action_shape() const {
        auto shape = env_->get_action_shape();
        return vector_to_tuple(shape);
    }

    // Get environment configuration
    EnvironmentConfig get_config() const {
        return env_->get_config();
    }

    // Get debug visualization data
    py::dict get_debug_data() const {
        auto debug_data = env_->get_debug_data();
        py::dict result;
        
        for (const auto& [key, values] : debug_data) {
            result[key.c_str()] = vector_to_numpy(values);
        }
        
        return result;
    }

    // Vectorized environment methods for batch processing
    py::list batch_reset(int batch_size, py::array_t<unsigned int> seeds) {
        std::vector<unsigned int> seed_vec;
        
        // If seeds provided, convert to vector
        if (seeds.size() > 0) {
            seed_vec = numpy_to_vector<unsigned int>(seeds);
        } else {
            // Create default seeds
            seed_vec.resize(batch_size);
            for (int i = 0; i < batch_size; i++) {
                seed_vec[i] = static_cast<unsigned int>(i);
            }
        }
        
        // Call batch reset
        auto observations = env_->batch_reset(batch_size, seed_vec);
        
        // Convert to list of numpy arrays
        py::list result;
        for (const auto& obs : observations) {
            result.append(vector_to_numpy(obs));
        }
        
        return result;
    }

    py::list batch_step(py::list actions) {
        // Convert Python list of actions to vector of vectors
        std::vector<std::vector<float>> action_vecs;
        for (const auto& action : actions) {
            action_vecs.push_back(numpy_to_vector<float>(action.cast<py::array_t<float>>()));
        }
        
        // Call batch step
        auto results = env_->batch_step(action_vecs);
        
        // Convert results to Python list of tuples
        py::list result_list;
        for (const auto& [obs, reward, done, truncated, info] : results) {
            // Convert observation to numpy array
            py::array_t<float> obs_array = vector_to_numpy(obs);
            
            // Convert info map to Python dict
            py::dict info_dict;
            for (const auto& [key, value] : info) {
                info_dict[key.c_str()] = value;
            }
            
            // Create and append the result tuple
            result_list.append(
                py::make_tuple(obs_array, reward, done, truncated, info_dict)
            );
        }
        
        return result_list;
    }

private:
    std::unique_ptr<Environment> env_;
    
    // Helper methods for numpy conversions
    template<typename T>
    py::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
        auto result = py::array_t<T>(vec.size());
        auto buf = result.mutable_data();
        std::copy(vec.begin(), vec.end(), buf);
        return result;
    }
    
    template<typename T>
    std::vector<T> numpy_to_vector(py::array_t<T> array) {
        auto buf = array.request();
        T* ptr = static_cast<T*>(buf.ptr);
        return std::vector<T>(ptr, ptr + buf.size);
    }
    
    template<typename T>
    py::tuple vector_to_tuple(const std::vector<T>& vec) {
        py::tuple result(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            PyTuple_SET_ITEM(result.ptr(), i, py::int_(vec[i]).release().ptr());
        }
        return result;
    }
};

PYBIND11_MODULE(gpu_environment, m) {
    m.doc() = "GPU-accelerated game environment for reinforcement learning";

    // Bind the environment configuration
    py::class_<EnvironmentConfig>(m, "EnvironmentConfig")
        .def(py::init<>())
        .def_readwrite("screen_width", &EnvironmentConfig::screen_width)
        .def_readwrite("screen_height", &EnvironmentConfig::screen_height)
        .def_readwrite("max_missiles", &EnvironmentConfig::max_missiles)
        .def_readwrite("player_size", &EnvironmentConfig::player_size)
        .def_readwrite("enemy_size", &EnvironmentConfig::enemy_size)
        .def_readwrite("missile_size", &EnvironmentConfig::missile_size)
        .def_readwrite("player_speed", &EnvironmentConfig::player_speed)
        .def_readwrite("enemy_speed", &EnvironmentConfig::enemy_speed)
        .def_readwrite("missile_speed", &EnvironmentConfig::missile_speed)
        .def_readwrite("missile_lifespan", &EnvironmentConfig::missile_lifespan)
        .def_readwrite("respawn_delay", &EnvironmentConfig::respawn_delay)
        .def_readwrite("max_steps", &EnvironmentConfig::max_steps)
        .def_readwrite("enable_missile_avoidance", &EnvironmentConfig::enable_missile_avoidance)
        .def_readwrite("missile_prediction_steps", &EnvironmentConfig::missile_prediction_steps)
        .def_readwrite("missile_detection_radius", &EnvironmentConfig::missile_detection_radius)
        .def_readwrite("missile_danger_radius", &EnvironmentConfig::missile_danger_radius)
        .def_readwrite("evasion_strength", &EnvironmentConfig::evasion_strength);

    // Bind the Python environment wrapper
    py::class_<PyEnvironment>(m, "Environment")
        .def(py::init<const EnvironmentConfig&>(), py::arg("config") = EnvironmentConfig())
        .def("reset", &PyEnvironment::reset, py::arg("seed") = 0)
        .def("step", &PyEnvironment::step)
        .def("get_observation_shape", &PyEnvironment::get_observation_shape)
        .def("get_action_shape", &PyEnvironment::get_action_shape)
        .def("get_config", &PyEnvironment::get_config)
        .def("get_debug_data", &PyEnvironment::get_debug_data)
        .def("batch_reset", &PyEnvironment::batch_reset, 
             py::arg("batch_size"), py::arg("seeds") = py::array_t<unsigned int>())
        .def("batch_step", &PyEnvironment::batch_step);

    // Custom convenience methods for creating gym-compatible environments
    m.def("make_gym_env", []() {
        // This function will be implemented in Python
        // using the environment we've exposed
        return py::none();
    });
    
    m.def("create_vectorized_env", [](int num_envs, const EnvironmentConfig& config) {
        // This function will be implemented in Python
        // to create a vectorized environment
        return py::none();
    }, py::arg("num_envs") = 4, py::arg("config") = EnvironmentConfig());
}
