cmake_minimum_required(VERSION 3.18)
project(gpu_game_environment LANGUAGES CXX)

# Modern CUDA detection (instead of deprecated FindCUDA)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CUDA_ENABLED TRUE)
  message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")

  # Set CUDA architecture specifically for RTX 5070 (Ada Lovelace)
  set(CMAKE_CUDA_ARCHITECTURES 89)
  message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(CUDA_ENABLED FALSE)
  message(STATUS "CUDA not found, using CPU-only build")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CUDA_ENABLED)
  set(CMAKE_CUDA_STANDARD 17)  # Updated to C++17 for better compatibility
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set pybind11 directory manually
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKE_DIR)
  set(pybind11_DIR "${PYBIND11_CMAKE_DIR}")
  message(STATUS "Found pybind11 via Python: ${pybind11_DIR}")
else()
  message(STATUS "Could not find pybind11 via Python, trying default paths")
  set(pybind11_DIR "${Python_SITELIB}/pybind11/share/cmake/pybind11")
  message(STATUS "Setting pybind11_DIR to: ${pybind11_DIR}")
endif()

# Find PyBind11
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  message(FATAL_ERROR "Could not find pybind11. Please install it with: pip install pybind11")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/entity.cpp
    src/environment.cpp
    src/physics_cpu.cpp
    src/reward.cpp
)

# Conditionally add CUDA-specific files
if(CUDA_ENABLED)
    set(CUDA_SOURCES
        src/physics.cu  # Using the original file name
    )
    list(APPEND SOURCES ${CUDA_SOURCES})
    add_definitions(-DWITH_CUDA)
endif()

# Pybind11 module
pybind11_add_module(gpu_environment
    pybind/bindings.cpp
    ${SOURCES}
)

# Link libraries
if(CUDA_ENABLED)
    target_link_libraries(gpu_environment PRIVATE ${CMAKE_CUDA_LIBRARIES})
endif()

# Set output directory
set_target_properties(gpu_environment PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)
