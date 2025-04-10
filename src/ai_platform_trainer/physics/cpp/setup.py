from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
import platform

# Based on the PyBind11 example setup
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def find_cmake_executable():
    """Find the CMake executable path"""
    # First try environment variable
    cmake_env = os.environ.get("CMAKE_EXECUTABLE")
    if cmake_env and os.path.isfile(cmake_env) and os.access(cmake_env, os.X_OK):
        return cmake_env
    
    # Try direct command (for when it's in PATH)
    try:
        subprocess.check_output(['cmake', '--version'], stderr=subprocess.STDOUT)
        return 'cmake'
    except (subprocess.SubprocessError, OSError):
        pass
    
    # Common locations on Windows
    if platform.system() == "Windows":
        common_locations = [
            r"C:\Program Files\CMake\bin\cmake.exe",
            r"C:\Program Files (x86)\CMake\bin\cmake.exe",
            # For CMake installed via winget, scoop, or chocolatey
            os.path.expanduser(r"~\scoop\apps\cmake\current\bin\cmake.exe"),
            os.path.expanduser(r"~\chocolatey\bin\cmake.exe"),
            # VS 2019/2022 bundled CMake
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Community"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
            r"\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
        ]
        
        for location in common_locations:
            if os.path.isfile(location) and os.access(location, os.X_OK):
                return location
    
    # Common locations on Linux/macOS
    elif platform.system() in ["Linux", "Darwin"]:
        common_locations = [
            "/usr/bin/cmake",
            "/usr/local/bin/cmake",
            "/opt/homebrew/bin/cmake",  # Homebrew on M1 Macs
        ]
        
        for location in common_locations:
            if os.path.isfile(location) and os.access(location, os.X_OK):
                return location
    
    return None


class CMakeBuild(build_ext):
    def run(self):
        # Find CMake executable
        self.cmake_executable = find_cmake_executable()
        
        if not self.cmake_executable:
            raise RuntimeError(
                "\n\nCMake must be installed to build the C++/CUDA extension.\n"
                "CMake could not be found automatically in common locations.\n"
                "Please install CMake or set the CMAKE_EXECUTABLE environment variable\n"
                "to the path of the cmake executable.\n"
                "On Windows, you can install CMake with:\n"
                "    winget install Kitware.CMake  \n"
                "    or\n"
                "    choco install cmake  \n"
            )
        
        print(f"Using CMake: {self.cmake_executable}")
        
        # Check if CMake is working
        try:
            version_output = subprocess.check_output(
                [self.cmake_executable, '--version'], 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"CMake version: {version_output.strip()}")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Error running CMake: {e}")
            
        # Check for CUDA
        try:
            import torch
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                print(f"CUDA is available. PyTorch CUDA version: {cuda_version}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("WARNING: CUDA is not available! Training will run on CPU only.")
        except ImportError:
            print("WARNING: PyTorch not found, skipping CUDA check")
            
        # Check for PyBind11
        try:
            import pybind11
            print(f"Found PyBind11 version: {pybind11.__version__}")
        except ImportError:
            print("WARNING: PyBind11 not found. Installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pybind11'])
        
        # Build extensions
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Get pybind11 cmake directory
        pybind11_cmake_dir = None
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "pybind11", "--cmakedir"],
                capture_output=True,
                text=True,
                check=True
            )
            pybind11_cmake_dir = result.stdout.strip()
            print(f"Found pybind11 cmake directory: {pybind11_cmake_dir}")
        except Exception as e:
            print(f"Warning: Could not get pybind11 cmake directory: {e}")
            print("Will try to use default pybind11 location.")
            
        # Set up cmake arguments
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]
        
        # Add pybind11 directory if found
        if pybind11_cmake_dir:
            cmake_args.append(f'-Dpybind11_DIR="{pybind11_cmake_dir}"')

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        
        # Use the located CMake executable
        subprocess.check_call(
            [self.cmake_executable, ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )
        subprocess.check_call(
            [self.cmake_executable, '--build', '.'] + build_args,
            cwd=self.build_temp
        )


setup(
    name='gpu_environment',
    version='0.1.0',
    author='AI Platform Trainer Team',
    description='GPU-accelerated game environment for reinforcement learning',
    long_description='',
    ext_modules=[CMakeExtension('gpu_environment')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'stable-baselines3>=1.0.0',
        'gym>=0.17.0',
        'numpy>=1.19.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)
