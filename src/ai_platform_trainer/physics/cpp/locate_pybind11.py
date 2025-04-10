import sys
import os
import pybind11
import site

print("Python version:", sys.version)
print("Executable:", sys.executable)
print("\nPybind11 Information:")
print("Version:", pybind11.__version__)
print("Path:", os.path.dirname(pybind11.__file__))

# Find all possible pybind11 cmake directories
site_packages = site.getsitepackages()
print("\nSite packages:")
for sp in site_packages:
    print(f"- {sp}")

user_site = site.getusersitepackages()
print(f"\nUser site packages: {user_site}")

# Check common locations for pybind11 cmake files
pybind11_cmake_locations = []

# Check main pybind11 package
main_cmake_dir = os.path.join(os.path.dirname(pybind11.__file__), "share", "cmake", "pybind11")
if os.path.exists(main_cmake_dir):
    pybind11_cmake_locations.append(main_cmake_dir)

# Check site packages
for sp in site_packages + [user_site]:
    # Check for pybind11
    cmake_dir = os.path.join(sp, "pybind11", "share", "cmake", "pybind11")
    if os.path.exists(cmake_dir):
        pybind11_cmake_locations.append(cmake_dir)
    
    # Check for pybind11-global
    global_cmake_dir = os.path.join(sp, "pybind11_global", "share", "cmake", "pybind11")
    if os.path.exists(global_cmake_dir):
        pybind11_cmake_locations.append(global_cmake_dir)

print("\nPotential pybind11 cmake directories:")
for i, location in enumerate(pybind11_cmake_locations, 1):
    print(f"{i}. {location}")
    # Check if it contains the key config files
    config_file = os.path.join(location, "pybind11Config.cmake")
    if os.path.exists(config_file):
        print("   ✓ Found pybind11Config.cmake")
    else:
        print("   ✗ Missing pybind11Config.cmake")

if not pybind11_cmake_locations:
    print("No pybind11 cmake directories found!")
else:
    print("\nRecommended command line argument:")
    print(f"-Dpybind11_DIR=\"{pybind11_cmake_locations[0]}\"")
