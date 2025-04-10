#!/usr/bin/env python
"""
Setup Environment Script for AI Platform Trainer CUDA Acceleration

This script checks for required dependencies and helps install them if missing:
- Visual Studio Build Tools with C++ components
- CMake
- CUDA Toolkit
- Python dependencies

Usage:
    python setup_environment.py [--install] [--cuda-only]

Options:
    --install     Attempt to install missing dependencies
    --cuda-only   Only check/install CUDA-related components
"""

import os
import sys
import platform
import subprocess
import argparse
import shutil
import tempfile
import ctypes
import urllib.request
# import time  # Unused import
# from pathlib import Path  # Unused import


# Define color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def colorize(text, color):
        if platform.system() == "Windows":
            # Enable VT100 escape sequences on Windows
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return f"{color}{text}{Colors.ENDC}"


def print_header(text):
    """Print a section header with formatting."""
    print("\n" + "=" * 80)
    print(Colors.colorize(text, Colors.HEADER + Colors.BOLD))
    print("=" * 80)


def print_status(component, status, message=""):
    """Print the status of a component check."""
    if status == "ok":
        status_text = Colors.colorize("✓ Found", Colors.GREEN + Colors.BOLD)
    elif status == "missing":
        status_text = Colors.colorize("✗ Missing", Colors.RED + Colors.BOLD)
    elif status == "warning":
        status_text = Colors.colorize("⚠ Warning", Colors.YELLOW + Colors.BOLD)
    else:
        status_text = status

    print(f"{component:25} {status_text:20} {message}")


def run_command(command, shell=False, env=None, capture_output=True):
    """Run a command and return its output."""
    try:
        if capture_output:
            result = subprocess.run(
                command, 
                shell=shell, 
                check=False, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                env=env,
                universal_newlines=True
            )
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(
                command,
                shell=shell,
                check=False,
                env=env
            )
            return result.returncode, "", ""
    except Exception as e:
        return 1, "", str(e)


def is_admin():
    """Check if the script is running with administrator privileges."""
    try:
        if platform.system() == "Windows":
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            return os.geteuid() == 0
    except Exception:
        return False


def check_python():
    """Check Python version and development headers."""
    print_header("Checking Python Environment")
    
    # Check Python version
    version = sys.version.split()[0]
    major, minor, patch = map(int, version.split('.'))
    
    if major >= 3 and minor >= 7:
        print_status("Python Version", "ok", f"Python {version}")
    else:
        print_status("Python Version", "warning", 
                    f"Python {version} (3.7+ recommended)")
    
    # Check for development headers
    def has_python_dev():
        if platform.system() == "Windows":
            # On Windows, check if Python.h exists in the include directory
            include_dir = os.path.join(sys.base_prefix, "include")
            return os.path.exists(os.path.join(include_dir, "Python.h"))
        else:
            # On Unix-like systems, try to compile a small C program
            with tempfile.NamedTemporaryFile(suffix='.c') as tmp:
                tmp.write(b'#include <Python.h>\nint main() { return 0; }')
                tmp.flush()
                code, _, _ = run_command(
                    ['gcc', '-c', tmp.name, '-o', os.devnull,
                        f'-I{sys.base_prefix}/include/python{major}.{minor}']
                )
                return code == 0
    
    if has_python_dev():
        print_status("Python Development Headers", "ok")
    else:
        print_status("Python Development Headers", "missing", 
                    "Required for building C++ extensions")


def check_cpp_compiler():
    """Check for a C++ compiler."""
    print_header("Checking C++ Compiler")
    
    if platform.system() == "Windows":
        # Check for Visual Studio (cl.exe)
        cl_path = shutil.which("cl")
        if cl_path:
            code, out, _ = run_command(["cl"], shell=True)
            if "Microsoft" in out:
                print_status("Visual C++ Compiler", "ok", f"Found at {cl_path}")
                return True
        
        # Check for Visual Studio installation
        vswhere_path = (
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\vswhere.exe"
        )
        if os.path.exists(vswhere_path):
            code, out, _ = run_command([vswhere_path, "-latest", "-property", "installationPath"])
            if code == 0 and out.strip():
                vs_path = out.strip()
                print_status("Visual Studio", "ok", f"Found at {vs_path}")
                vcvars_path = os.path.join(
                    vs_path, 
                    "VC\\Auxiliary\\Build\\vcvars64.bat"
                )
                if os.path.exists(vcvars_path):
                    print_status("VC Environment Script", "ok", f"Found at {vcvars_path}")
                    print("\nTo use the Visual C++ compiler, run:")
                    print(f'  "{vcvars_path}"')
                    print("  OR")
                    print("  Open 'Developer Command Prompt for VS' from the Start Menu")
                    return True
        
        print_status("Visual C++ Compiler", "missing", 
                    "Required for building C++ extensions")
        return False
    else:
        # Check for GCC/Clang on Unix-like systems
        for compiler in ["g++", "clang++"]:
            compiler_path = shutil.which(compiler)
            if compiler_path:
                code, out, _ = run_command([compiler_path, "--version"])
                if code == 0:
                    print_status(f"{compiler} Compiler", "ok", 
                                f"Found at {compiler_path}")
                    return True
        
        print_status("C++ Compiler", "missing", 
                    "Required for building C++ extensions")
        return False


def check_cmake():
    """Check for CMake installation."""
    print_header("Checking CMake")
    
    cmake_path = shutil.which("cmake")
    if cmake_path:
        code, out, _ = run_command([cmake_path, "--version"])
        if code == 0:
            version = out.split('\n')[0].split()[-1]
            print_status("CMake", "ok", f"Version {version} at {cmake_path}")
            
            # Check for minimum version requirement
            major, minor, _ = map(int, version.split('.'))
            if major < 3 or (major == 3 and minor < 18):
                print_status("CMake Version", "warning", 
                            "CMake 3.18+ is recommended")
            return True
    
    print_status("CMake", "missing", "Required for building C++ extensions")
    return False


def check_cuda():
    """Check for CUDA installation."""
    print_header("Checking CUDA")
    
    # Check for NVCC
    nvcc_path = shutil.which("nvcc")
    cuda_found = False
    
    if nvcc_path:
        code, out, _ = run_command([nvcc_path, "--version"])
        if code == 0:
            version = out.split("release")[-1].split(",")[0].strip()
            print_status("CUDA Toolkit", "ok", f"Version {version} at {nvcc_path}")
            cuda_found = True
        else:
            print_status("CUDA Toolkit", "warning", "nvcc found but not working")
    else:
        # Try to find CUDA in common locations
        if platform.system() == "Windows":
            cuda_paths = [
                os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                             "NVIDIA GPU Computing Toolkit\\CUDA")
            ]
            
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    versions = [d for d in os.listdir(cuda_path) if d.startswith('v')]
                    if versions:
                        latest = sorted(versions)[-1]
                        nvcc_path = os.path.join(cuda_path, latest, "bin", "nvcc.exe")
                        if os.path.exists(nvcc_path):
                            print_status("CUDA Toolkit", "warning", 
                                        f"Found at {nvcc_path} but not in PATH")
                            print(f"Add {os.path.dirname(nvcc_path)} to your PATH")
                            cuda_found = True
        
        if not cuda_found:
            print_status("CUDA Toolkit", "missing", 
                        "Required for GPU acceleration")
    
    # Check for CUDA capabilities via PyTorch
    try:
        import torch
        print_status("PyTorch", "ok", f"Version {torch.__version__}")
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            
            print_status("PyTorch CUDA Support", "ok",
                         f"CUDA {cuda_version}, {device_count} device(s)")
            print_status("GPU Device", "ok", device_name)
            cuda_found = True
        else:
            print_status("PyTorch CUDA Support", "warning", 
                        "CUDA not available through PyTorch")
    except ImportError:
        print_status("PyTorch", "missing", 
                    "Required for AI training and inference")
    
    return cuda_found


def check_python_dependencies():
    """Check for required Python packages."""
    print_header("Checking Python Dependencies")
    
    dependencies = {
        "torch": "1.7.0",
        "numpy": "1.19.0",
        "pybind11": "2.6.0",
        "pytest": "6.0.0"
    }
    
    all_found = True
    for package, min_version in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "unknown")
            
            print_status(f"Python {package}", "ok", f"Version {version}")
        except ImportError:
            print_status(f"Python {package}", "missing", f"Minimum version: {min_version}")
            all_found = False
    
    return all_found


def check_all_dependencies(cuda_only=False):
    """Check all required dependencies."""
    results = {}
    
    if not cuda_only:
        results["python"] = check_python()
        results["cpp"] = check_cpp_compiler()
        results["cmake"] = check_cmake()
        results["python_deps"] = check_python_dependencies()
    
    results["cuda"] = check_cuda()
    
    return results


def install_vs_build_tools():
    """Download and install Visual Studio Build Tools."""
    print_header("Installing Visual Studio Build Tools")
    
    if platform.system() != "Windows":
        print("Visual Studio Build Tools can only be installed on Windows.")
        return False
    
    if not is_admin():
        print(Colors.colorize(
            "Administrator privileges required for VS Build Tools installation.",
            Colors.RED
        ))
        print("Please run this script as administrator and try again.")
        return False
    
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            installer_path = os.path.join(temp_dir, "vs_buildtools.exe")
            
            print("Downloading Visual Studio Build Tools installer...")
            url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
            urllib.request.urlretrieve(url, installer_path)
            
            print("Running installer. This may take a while...")
            # Install the required components
            command = [
                installer_path,
                "--quiet", "--norestart", "--wait",
                "--add", "Microsoft.VisualStudio.Workload.VCTools",
                "--add", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "--add", "Microsoft.VisualStudio.Component.Windows10SDK.19041"
            ]
            
            returncode, _, _ = run_command(command, capture_output=False)
            
            if returncode == 0:
                print("Visual Studio Build Tools installed successfully.")
                return True
            else:
                print(f"Installation failed with code {returncode}.")
                print("You may need to download and install Visual Studio Build Tools manually.")
                return False
    except Exception as e:
        print(f"Error during installation: {e}")
        return False


def install_cmake():
    """Download and install CMake."""
    print_header("Installing CMake")
    
    if platform.system() == "Windows":
        try:
            # Try using winget first
            print("Attempting to install CMake using winget...")
            returncode, out, err = run_command(["winget", "install", "Kitware.CMake"], capture_output=True)
            
            if returncode == 0:
                print("CMake installed successfully using winget.")
                print("Please restart your terminal to use CMake.")
                return True
            
            # If winget fails, try downloading the installer
            with tempfile.TemporaryDirectory() as temp_dir:
                installer_path = os.path.join(temp_dir, "cmake-installer.msi")
                
                print("Downloading CMake installer...")
                url = "https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-windows-x86_64.msi"
                urllib.request.urlretrieve(url, installer_path)
                
                print("Running installer...")
                command = ["msiexec", "/i", installer_path, "/quiet", "/norestart", "ADD_CMAKE_TO_PATH=System"]
                
                returncode, _, _ = run_command(command, capture_output=False)
                
                if returncode == 0:
                    print("CMake installed successfully.")
                    print("Please restart your terminal to use CMake.")
                    return True
                else:
                    print(f"Installation failed with code {returncode}.")
                    return False
        except Exception as e:
            print(f"Error during installation: {e}")
            return False
    elif platform.system() == "Linux":
        try:
            # Try using apt for Debian/Ubuntu
            print("Attempting to install CMake using apt...")
            command = ["sudo", "apt", "update"]
            run_command(command, capture_output=False)
            
            command = ["sudo", "apt", "install", "-y", "cmake"]
            returncode, _, _ = run_command(command, capture_output=False)
            
            if returncode == 0:
                print("CMake installed successfully.")
                return True
            else:
                print(f"Installation failed with code {returncode}.")
                return False
        except Exception as e:
            print(f"Error during installation: {e}")
            return False
    elif platform.system() == "Darwin":  # macOS
        try:
            # Try using brew
            print("Attempting to install CMake using brew...")
            command = ["brew", "install", "cmake"]
            returncode, _, _ = run_command(command, capture_output=False)
            
            if returncode == 0:
                print("CMake installed successfully.")
                return True
            else:
                print(f"Installation failed with code {returncode}.")
                return False
        except Exception as e:
            print(f"Error during installation: {e}")
            return False
    
    print("Automatic installation not supported for this platform.")
    print("Please install CMake manually.")
    return False


def install_cuda():
    """Download and install CUDA Toolkit."""
    print_header("Installing CUDA Toolkit")
    
    print(Colors.colorize(
        "CUDA Toolkit installation requires manual steps.",
        Colors.YELLOW
    ))
    
    print("Please follow these steps to install CUDA Toolkit:")
    print("1. Visit: https://developer.nvidia.com/cuda-downloads")
    print("2. Select your operating system and download the installer")
    print("3. Run the installer and follow the instructions")
    print("4. After installation, add CUDA bin directory to your PATH environment variable")
    
    if platform.system() == "Windows":
        print("\nFor Windows, typically add:")
        print("  C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin")
        print("to your PATH (replace X.Y with your CUDA version)")
    
    print("\nWould you like to open the CUDA download page in your web browser? (y/n)")
    choice = input().strip().lower()
    
    if choice == 'y':
        import webbrowser
        webbrowser.open("https://developer.nvidia.com/cuda-downloads")
    
    return False  # Return False since we can't verify the installation


def install_python_dependencies():
    """Install required Python packages."""
    print_header("Installing Python Dependencies")
    
    dependencies = ["torch", "numpy", "pybind11", "pytest"]
    
    try:
        for package in dependencies:
            print(f"Installing {package}...")
            command = [sys.executable, "-m", "pip", "install", package]
            returncode, _, _ = run_command(command, capture_output=False)
            
            if returncode != 0:
                print(f"Failed to install {package}.")
                return False
        
        print("All Python dependencies installed successfully.")
        return True
    except Exception as e:
        print(f"Error during installation: {e}")
        return False


def install_missing_dependencies(results, cuda_only=False):
    """Install missing dependencies."""
    if not is_admin() and platform.system() == "Windows":
        print(Colors.colorize(
            "\nWarning: Some installations may require administrator privileges.",
            Colors.YELLOW
        ))
        print("Consider running this script as administrator.")
    
    success = True
    
    if not cuda_only:
        if "cpp" in results and not results["cpp"]:
            if platform.system() == "Windows":
                success = install_vs_build_tools() and success
            else:
                print("Please install a C++ compiler manually (GCC or Clang).")
                success = False
        
        if "cmake" in results and not results["cmake"]:
            success = install_cmake() and success
        
        if "python_deps" in results and not results["python_deps"]:
            success = install_python_dependencies() and success
    
    if "cuda" in results and not results["cuda"]:
        install_cuda()  # Don't affect success as this is a manual process
    
    return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check and install dependencies for AI Platform Trainer"
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Attempt to install missing dependencies"
    )
    parser.add_argument(
        "--cuda-only", action="store_true",
        help="Only check/install CUDA-related components"
    )
    
    args = parser.parse_args()
    
    print_header("AI Platform Trainer - Environment Setup")
    print(f"System: {platform.system()} {platform.release()} {platform.machine()}")
    
    # Check dependencies
    results = check_all_dependencies(args.cuda_only)
    
    # Print summary
    print_header("Summary")
    
    all_ok = all(results.values())
    
    if all_ok:
        print(Colors.colorize(
            "All required dependencies are installed and configured correctly!",
            Colors.GREEN + Colors.BOLD
        ))
        print("\nYou can proceed with building the AI Platform Trainer.")
        print("Run: python find_cmake.py")
    else:
        print(Colors.colorize(
            "Some dependencies are missing or not configured correctly.",
            Colors.RED + Colors.BOLD
        ))
        
        if args.install:
            print("\nAttempting to install missing dependencies...")
            install_missing_dependencies(results, args.cuda_only)
            
            # Re-check dependencies
            print("\nRe-checking dependencies after installation...")
            results = check_all_dependencies(args.cuda_only)
            
            all_ok = all(results.values())
            
            if all_ok:
                print(Colors.colorize(
                    "\nAll required dependencies are now installed and configured correctly!",
                    Colors.GREEN + Colors.BOLD
                ))
                print("\nYou can proceed with building the AI Platform Trainer.")
                print("Run: python find_cmake.py")
            else:
                print(Colors.colorize(
                    "\nSome dependencies could not be installed automatically.",
                    Colors.RED + Colors.BOLD
                ))
                print("Please refer to SETUP_GUIDE.md for manual installation instructions.")
        else:
            print("\nUse --install flag to attempt automatic installation of missing dependencies:")
            print("python setup_environment.py --install")
            print("\nOr refer to SETUP_GUIDE.md for manual installation instructions.")
    
    print("\nFor more information, see SETUP_GUIDE.md and README_BUILD.md")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
