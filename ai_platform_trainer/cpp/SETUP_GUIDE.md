# Setting Up Development Environment for AI Platform Trainer

This guide will help you install and configure all the required dependencies for building the CUDA-accelerated environment for AI Platform Trainer.

## Prerequisites Overview

You need the following components:

1. **Python 3.7+** with development headers
2. **C++ compiler** (Visual Studio Build Tools on Windows)
3. **CMake** (version 3.18 or higher)
4. **CUDA Toolkit** (version 11.0 or higher)
5. **Python dependencies**: PyTorch, PyBind11, etc.

## Step-by-Step Installation Guide

### 1. Install Visual Studio Build Tools (Windows)

The C++ compiler comes from Visual Studio Build Tools, which is required even if you don't use Visual Studio IDE.

1. Download the Visual Studio Build Tools installer:
   - Go to [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
   - Scroll down to "Tools for Visual Studio" section
   - Click "Download" for "Build Tools for Visual Studio 2022"

2. Run the installer and select these components:
   - Workload: "Desktop development with C++"
   - Individual components:
     - "MSVC C++ build tools"
     - "Windows 10/11 SDK"
     - "C++ CMake tools for Windows"

3. Complete the installation (this may take a while)

4. Verify installation by opening a command prompt and running:

   ```bash
   cl
   ```

   If you see "Microsoft (R) C/C++ Optimizing Compiler" message, it's working.

### 2. Install CMake

CMake is required to generate the build system.

#### Windows:

1. **Option 1 - Using Winget (recommended)**:

   ```bash
   winget install Kitware.CMake
   ```

2. **Option 2 - Manual Installation**:
   - Download the installer from [CMake Downloads](https://cmake.org/download/)
   - Run the installer and select "Add CMake to PATH" option
   - Complete the installation

3. Verify installation:

   ```bash
   cmake --version
   ```

   You should see "cmake version 3.18" or higher.

#### Linux:

```bash
sudo apt update
sudo apt install cmake
cmake --version
```

#### macOS:

```bash
brew install cmake
cmake --version
```

### 3. Install CUDA Toolkit

The CUDA Toolkit is required for GPU acceleration.

#### Windows:

1. Check if your GPU supports CUDA:
   - NVIDIA GPUs from the GeForce 600 series or newer generally support CUDA
   - Visit [NVIDIA's CUDA GPUs list](https://developer.nvidia.com/cuda-gpus) to confirm

2. Download CUDA Toolkit:
   - Go to [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Select your operating system and version
   - Follow the download and installation instructions

3. During installation:
   - Choose "Custom installation"
   - Ensure "CUDA" and "Development" components are selected
   - Complete the installation

4. Verify installation:

   ```bash
   nvcc --version
   ```

   You should see "Cuda compilation tools" followed by the version.

#### Linux:

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version
```

#### macOS:

CUDA support on macOS is limited to older versions. Consider using a Windows or Linux environment for CUDA development.

### 4. Install Python Dependencies

Install the required Python packages:

```bash
pip install torch torchvision numpy pybind11 pytest
```

Verify PyTorch CUDA support:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Visual Studio Build Tools Issues

**Problem: `cl` command not found**
- Open the "Developer Command Prompt for VS 2022" from the Start menu instead of a regular command prompt
- Or run `"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"` in your command prompt

**Problem: CMake can't find the compiler**
- Make sure you're using a "Developer Command Prompt for VS 2022" 
- Verify that the compiler is properly installed: `where cl`

### CUDA Issues

**Problem: `nvcc` command not found**
- Add CUDA to your PATH: Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your PATH environment variable
- Replace `v12.x` with your actual CUDA version

**Problem: Incompatible CUDA and Visual Studio versions**
- Each CUDA version supports specific Visual Studio versions
- Consult the [CUDA Toolkit documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) for compatibility information

### Python Issues

**Problem: PyBind11 not found**
- Install it manually: `pip install pybind11`
- Verify installation: `python -c "import pybind11; print(pybind11.__version__)"`

## Verifying Your Environment

Run our automated environment checker to verify everything is correctly set up:

```bash
cd ai_platform_trainer/cpp
python find_cmake.py
```

This will check for:
- CMake installation
- CUDA installation
- PyTorch and CUDA compatibility
- PyBind11 installation
- C++ compiler availability

## Next Steps

Once your environment is set up, proceed to [README_BUILD.md](README_BUILD.md) for instructions on building the project.
