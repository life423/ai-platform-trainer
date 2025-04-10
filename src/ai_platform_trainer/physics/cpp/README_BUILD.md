# Building the CUDA-accelerated Environment

This document provides instructions for building the CUDA-accelerated training environment for AI Platform Trainer.

## Prerequisites

Before building, ensure you have the following installed:

1. **CUDA Toolkit** (version 11.0 or higher)
2. **CMake** (version 3.18 or higher)
3. **Python dependencies**:
   - PyTorch
   - PyBind11
   - Stable-Baselines3
   - NumPy
4. **C++ compiler**:
   - Windows: Visual Studio Build Tools with C++ support
   - Linux/macOS: GCC/Clang

## Build Process

### Step 1: Locate CMake

Run the helper script to locate CMake on your system:

```bash
python find_cmake.py
```

This script will:
- Search for CMake in common locations
- Set the CMAKE_EXECUTABLE environment variable
- Check other dependencies (CUDA, PyTorch, PyBind11)
- Optionally start the build process

### Step 2: Build the Extension

If you didn't choose to build in the previous step, run:

```bash
python setup.py build_ext --inplace
```

This will:
1. Compile the C++/CUDA code
2. Create a Python extension module
3. Place the built module where Python can find it

### Step 3: Verify the Build

To verify that the CUDA acceleration is working properly, run:

```bash
python verify_gpu_training.py
```

This will perform several checks to ensure your GPU is being used:
1. CUDA capability check
2. PyBind11 extension verification
3. CUDA kernel execution test
4. Performance benchmark (GPU vs CPU)

## Troubleshooting

### CMake Not Found

If CMake isn't found automatically:

1. Install CMake:

   - Windows: `winget install Kitware.CMake` or `choco install cmake`
   - Linux: `apt install cmake` or `yum install cmake`
   - macOS: `brew install cmake`


2. Set the CMAKE_EXECUTABLE environment variable:
   - Windows: `set CMAKE_EXECUTABLE=C:\path\to\cmake.exe`
   - Linux/macOS: `export CMAKE_EXECUTABLE=/path/to/cmake`

### Missing CUDA

If CUDA isn't detected:

1. Ensure the CUDA Toolkit is installed
2. Make sure PyTorch was installed with CUDA support
3. Check that your GPU driver is up to date

### Compiler Errors

If you encounter compiler errors:

1. On Windows, make sure you're using a Developer Command Prompt for Visual Studio
2. Install required C++ compiler tools
3. Make sure your CUDA version is compatible with your compiler version

## Using the Environment

After successfully building the extension, you can use it for training:

```python
from ai_platform_trainer.cpp.gpu_environment import make_env
from stable_baselines3 import PPO

# Create environment
env = make_env()

# Create and train model on GPU
model = PPO("MlpPolicy", env, device="cuda")
model.learn(total_timesteps=100000)
```

For more advanced usage and benchmarking, refer to the `train_missile_avoidance.py` script.
