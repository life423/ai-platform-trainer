@echo off
echo Setting up build environment for AI Platform Trainer CUDA extension...

REM Setup Visual Studio environment
echo Setting up Visual Studio environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Find Python executable
echo Detecting Python...
FOR /F "tokens=* USEBACKQ" %%F IN (`where python`) DO (
    SET PYTHON_EXECUTABLE=%%F
    GOTO PYTHON_FOUND
)
:PYTHON_FOUND
echo Found Python at: %PYTHON_EXECUTABLE%

REM Get Python lib path for pybind11
echo Detecting pybind11...
FOR /F "tokens=* USEBACKQ" %%F IN (`python -c "import site, os; print(site.getsitepackages()[0])"`) DO (
    SET SITE_PACKAGES=%%F
)

IF EXIST "%SITE_PACKAGES%\pybind11-global\share\cmake\pybind11" (
    echo pybind11-global found at: %SITE_PACKAGES%\pybind11-global
    SET PYBIND11_CMAKE_DIR=%SITE_PACKAGES%\pybind11-global\share\cmake\pybind11
) ELSE IF EXIST "%SITE_PACKAGES%\pybind11\share\cmake\pybind11" (
    echo pybind11 found at: %SITE_PACKAGES%\pybind11
    SET PYBIND11_CMAKE_DIR=%SITE_PACKAGES%\pybind11\share\cmake\pybind11
) ELSE (
    echo WARNING: pybind11 cmake directory not found. Using default system directory.
)

echo Building with CMake...
cd %~dp0
if not exist build mkdir build
cd build

REM Configure with CMake
echo Running CMake configuration...
IF DEFINED PYBIND11_CMAKE_DIR (
    cmake .. -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" -Dpybind11_ROOT="%PYBIND11_CMAKE_DIR%" -Dpybind11_DIR="%PYBIND11_CMAKE_DIR%" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
) ELSE (
    cmake .. -DPYTHON_EXECUTABLE="%PYTHON_EXECUTABLE%" -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
)

REM Build using CMake
echo Building the extension...
cmake --build . --config Release

echo Done! The missiles will now travel much further on screen.
echo Press any key to exit...
pause
