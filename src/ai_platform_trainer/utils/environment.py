"""
Environment utilities for AI Platform Trainer.

This module provides utilities for detecting and configuring execution environments,
such as CPU/GPU availability, version detection, and system compatibility checks.
"""
import logging
import platform
import os
from typing import Optional, Tuple, Dict, Any

import torch


def get_device() -> torch.device:
    """
    Get the best available PyTorch device (CUDA GPU if available, otherwise CPU).
    
    Returns:
        The best available torch.device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logging.info("GPU not available, using CPU")
    
    return device


def get_system_info() -> Dict[str, Any]:
    """
    Get detailed information about the current system environment.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    # Add CUDA details if available
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
        })
    
    return info


def check_compatibility() -> Tuple[bool, str]:
    """
    Check if the current environment is compatible with AI Platform Trainer.
    
    Returns:
        Tuple of (is_compatible, message)
    """
    # Check Python version
    python_version = tuple(map(int, platform.python_version_tuple()))
    if python_version < (3, 8):
        return False, (
            f"Python version {platform.python_version()} is not supported. "
            f"Please use Python 3.8 or newer."
        )
    
    # Check PyTorch version
    torch_version = torch.__version__.split('.')
    torch_major, torch_minor = int(torch_version[0]), int(torch_version[1])
    if torch_major < 1 or (torch_major == 1 and torch_minor < 8):
        return False, (
            f"PyTorch version {torch.__version__} is not supported. "
            f"Please use PyTorch 1.8.0 or newer."
        )
    
    # All checks passed
    return True, "Environment is compatible with AI Platform Trainer."


def setup_training_environment(
    seed: Optional[int] = None,
    deterministic: bool = False,
    benchmark: bool = True
) -> None:
    """
    Set up the PyTorch environment for training.
    
    Args:
        seed: Optional random seed for reproducibility
        deterministic: Whether to enable deterministic mode
        benchmark: Whether to enable cudnn benchmark mode
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    # Configure deterministic behavior if requested
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    elif benchmark and torch.cuda.is_available():
        # Use benchmark mode for faster training with variable input sizes
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Print system information when run directly
    info = get_system_info()
    for key, value in info.items():
        logging.info(f"{key}: {value}")
    
    # Check compatibility
    is_compatible, message = check_compatibility()
    logging.info(message)
    
    # Get device
    device = get_device()
    logging.info(f"Using device: {device}")
