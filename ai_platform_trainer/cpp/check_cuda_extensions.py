#!/usr/bin/env python
"""
Check if CUDA extensions are properly built and can be used.

This script verifies that the custom C++/CUDA extensions are built correctly
and can be imported and used from Python.
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CUDA-Check")

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def check_gpu_environment():
    """Check if gpu_environment module can be imported."""
    try:
        import gpu_environment

        logger.info("Successfully imported gpu_environment module.")
        return True
    except ImportError as e:
        logger.error(f"Failed to import gpu_environment module: {e}")
        logger.error(
            "Make sure you've built the extensions with 'python setup.py build_ext --inplace'"
        )
        return False


def test_environment():
    """Test creating and using the environment."""
    try:
        import gpu_environment

        # Create environment
        config = gpu_environment.EnvironmentConfig()
        env = gpu_environment.Environment(config)
        logger.info("Successfully created environment.")

        # Test reset
        obs = env.reset(42)
        logger.info(f"Reset successful. Observation shape: {obs.shape}")

        # Test step
        import numpy as np

        action = np.array([0.5, 0.5], dtype=np.float32)
        next_obs, reward, done, truncated, info = env.step(action)
        logger.info(f"Step successful. Reward: {reward}")

        return True
    except Exception as e:
        logger.error(f"Error testing environment: {e}")
        return False


def main():
    """Main function."""
    logger.info("=== Checking CUDA Extensions ===")

    # Step 1: Check if gpu_environment can be imported
    if not check_gpu_environment():
        logger.error("Failed to import gpu_environment module.")
        return False

    # Step 2: Test environment
    if not test_environment():
        logger.error("Failed to use environment.")
        return False

    logger.info("\n=== Check Successful! ===")
    logger.info("CUDA extensions are built and working correctly.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
