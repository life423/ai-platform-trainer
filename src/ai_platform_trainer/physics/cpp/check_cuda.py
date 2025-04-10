"""
Simple script to check CUDA capability in PyTorch.
"""
import torch

# Check CUDA
print("\n=== CUDA Capability Check ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("WARNING: CUDA is not available! The training will run on CPU only.")
