cmake_minimum_required(VERSION 4.0.2)
set(CMAKE_MODULE_PATH "")
project(CheckCUDA CUDA)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_CUDA_COMPILER \"${CMAKE_CUDA_COMPILER}\")\n"
  
  )