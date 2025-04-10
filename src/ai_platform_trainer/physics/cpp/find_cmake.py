"""
Helper script to locate CMake on the system and set the CMAKE_EXECUTABLE environment variable.
"""
import os
import subprocess
import sys
import platform


def find_cmake_executable():
    """Find the CMake executable path"""
    # Try direct command (for when it's in PATH)
    try:
        subprocess.check_output(['cmake', '--version'], stderr=subprocess.STDOUT)
        cmake_path = 'cmake'
        print(f"Found CMake in PATH: {cmake_path}")
        return cmake_path
    except (subprocess.SubprocessError, OSError):
        print("CMake not found in PATH, searching common locations...")
    
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
                print(f"Found CMake at: {location}")
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
                print(f"Found CMake at: {location}")
                return location
    
    print("Unable to find CMake automatically.")
    return None


def verify_cmake(cmake_path):
    """Verify CMake works and show version"""
    try:
        output = subprocess.check_output(
            [cmake_path, '--version'], 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        print(f"CMake version: {output.strip()}")
        return True
    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error running CMake: {e}")
        return False


def check_other_deps():
    """Check other dependencies"""
    print("\nChecking other dependencies:")
    # Check for CUDA via PyTorch
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"✓ CUDA is available. PyTorch CUDA version: {cuda_version}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA is not available! Training will run on CPU only.")
    except ImportError:
        print("✗ PyTorch not found.")
    
    # Check for PyBind11
    try:
        import pybind11
        print(f"✓ PyBind11 version: {pybind11.__version__}")
    except ImportError:
        print("✗ PyBind11 not found. Will be installed during build.")
    
    # Check for Visual Studio tools (Windows only)
    if platform.system() == "Windows":
        try:
            cl_output = subprocess.check_output(
                ['where', 'cl'], 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"✓ Visual C++ compiler found: {cl_output.strip()}")
        except (subprocess.SubprocessError, OSError):
            print("✗ Visual C++ compiler (cl.exe) not found in PATH.")
            print("  You may need to install Visual Studio with C++ workload")
            print("  or run from a Developer Command Prompt for VS.")


def main():
    print("============================================")
    print("CMake Finder for AI Platform Trainer")
    print("============================================\n")
    # Find CMake
    cmake_path = find_cmake_executable()
    if cmake_path:
        print(f"\nFound CMake at: {cmake_path}")
        # Verify it works
        if verify_cmake(cmake_path):
            print("✓ CMake verification successful!")
            
            # Set environment variable
            os.environ["CMAKE_EXECUTABLE"] = cmake_path
            print(f"\nSet CMAKE_EXECUTABLE={cmake_path}")
            print("\nYou can now run the following command to build:")
            print("python setup.py build_ext --inplace")
            # Check other dependencies
            check_other_deps()
            # Offer to run the build
            choice = input("\nDo you want to run the build now? (y/n): ")
            if choice.lower().startswith('y'):
                print("\nRunning build...")
                os.chdir(os.path.dirname(os.path.abspath(__file__)))
                subprocess.call([sys.executable, 'setup.py', 'build_ext', '--inplace'])
        else:
            print("✗ CMake verification failed.")
    else:
        print("\n✗ CMake not found. Please install CMake and try again.")
        print("  On Windows, you can install CMake with:")
        print("      winget install Kitware.CMake")
        print("      or")
        print("      choco install cmake")
        # Ask for manual path
        print("\nIf you know where CMake is installed, you can enter the path manually:")
        manual_path = input("CMake path (or press Enter to skip): ")
        
        if manual_path and os.path.isfile(manual_path):
            if verify_cmake(manual_path):
                os.environ["CMAKE_EXECUTABLE"] = manual_path
                print(f"\nSet CMAKE_EXECUTABLE={manual_path}")
            else:
                print("✗ The provided path does not seem to be a valid CMake executable.")
    
    print("\n============================================")


if __name__ == "__main__":
    main()
