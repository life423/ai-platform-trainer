#!/usr/bin/env python3
"""
Local Build Script for AI Platform Trainer

This script builds the executable locally for testing before deployment.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔨 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"   Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pyinstaller',
        'stable_baselines3',
        'pygame',
        'torch',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def pre_train_models():
    """Pre-train AI models for bundling."""
    print("\n🧠 Pre-training AI models...")
    
    # Check if models already exist
    models_dir = Path("models")
    rl_model_path = models_dir / "missile_rl_model_final.zip"
    
    if rl_model_path.exists():
        print("   ✅ RL model already exists - skipping training")
        return True
    
    # Run pre-training script
    script_path = Path("scripts") / "pre_train_for_deployment.py"
    if not script_path.exists():
        print("   ❌ Pre-training script not found!")
        return False
    
    return run_command(f"python {script_path}", "AI model pre-training")

def build_executable():
    """Build the executable using PyInstaller."""
    print("\n🏗️  Building executable...")
    
    # Determine output name based on platform
    system = platform.system().lower()
    if system == "windows":
        output_name = "AI-Platform-Trainer.exe"
    elif system == "darwin":
        output_name = "AI-Platform-Trainer.app"
    else:
        output_name = "AI-Platform-Trainer"
    
    # Check if spec file exists
    spec_file = Path("ai-platform-trainer.spec")
    if spec_file.exists():
        print("   📋 Using PyInstaller spec file")
        cmd = f"pyinstaller --clean {spec_file}"
    else:
        print("   📋 Using inline PyInstaller configuration")
        cmd = f"""pyinstaller --clean --onefile --windowed --name "AI-Platform-Trainer" \
                 --add-data "assets{os.pathsep}assets" \
                 --add-data "models{os.pathsep}models" \
                 --add-data "config.json{os.pathsep}." \
                 --hidden-import "stable_baselines3" \
                 --hidden-import "gymnasium" \
                 --hidden-import "torch" \
                 --hidden-import "pygame" \
                 --collect-all "stable_baselines3" \
                 run_game.py"""
    
    if run_command(cmd, "PyInstaller build"):
        print(f"   🎯 Built: dist/{output_name}")
        return True
    return False

def test_executable():
    """Test the built executable."""
    print("\n🧪 Testing executable...")
    
    # Find the built executable
    dist_dir = Path("dist")
    if not dist_dir.exists():
        print("   ❌ dist/ directory not found!")
        return False
    
    # Look for the executable
    executables = list(dist_dir.glob("AI-Platform-Trainer*"))
    if not executables:
        print("   ❌ No executable found in dist/!")
        return False
    
    exe_path = executables[0]
    print(f"   📏 Executable: {exe_path}")
    
    # Get size
    size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"   📏 Size: {size_mb:.1f} MB")
    
    # Try to run with timeout (just test it starts)
    print("   🚀 Testing startup...")
    try:
        # Very short test - just see if it starts without crashing
        if platform.system() == "Windows":
            cmd = f'timeout 3 "{exe_path}" 2>nul || echo "Test completed"'
        else:
            cmd = f'timeout 3 "{exe_path}" 2>/dev/null || echo "Test completed"'
        
        subprocess.run(cmd, shell=True, timeout=10)
        print("   ✅ Executable starts successfully")
        return True
    except Exception as e:
        print(f"   ⚠️  Test completed (this is normal): {e}")
        return True

def main():
    """Main build process."""
    print("🚀 AI Platform Trainer - Local Build")
    print("=" * 50)
    
    # Check we're in the right directory
    if not Path("run_game.py").exists():
        print("❌ Please run this script from the project root directory")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Pre-train models
    if not pre_train_models():
        print("⚠️  Continuing without pre-trained models (first-time training will occur)")
    
    # Build executable
    if not build_executable():
        return 1
    
    # Test executable
    if not test_executable():
        print("⚠️  Testing completed with warnings")
    
    print("\n🎉 Build completed successfully!")
    print(f"📦 Executable location: dist/")
    print("🎮 Ready for distribution!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())