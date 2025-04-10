"""
Verification script for GPU utilization during training.

This script performs a short training run and monitors GPU usage to verify
that the CUDA implementation is correctly utilizing the GPU cores.
"""
import os
import time
import argparse
import numpy as np
import torch
import psutil
import threading
import subprocess
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Import our GPU environment wrapper
from gpu_environment import make_env, HAS_GPU_ENV

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for monitoring
gpu_utilization = []
cpu_utilization = []
memory_usage = []
timestamps = []
stop_monitoring = False

def get_gpu_utilization():
    """Get current GPU utilization using nvidia-smi"""
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            universal_newlines=True
        )
        return float(result.strip())
    except Exception as e:
        logger.error(f"Error getting GPU utilization: {e}")
        return 0.0


def get_cpu_utilization():
    """Get current CPU utilization"""
    return psutil.cpu_percent()


def get_memory_usage():
    """Get current memory usage in MB"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def monitor_resources():
    """Monitor GPU, CPU and memory usage"""
    global stop_monitoring, gpu_utilization, cpu_utilization, memory_usage, timestamps
    
    start_time = time.time()
    while not stop_monitoring:
        current_time = time.time() - start_time
        timestamps.append(current_time)
        
        # Get GPU utilization
        gpu_util = get_gpu_utilization()
        gpu_utilization.append(gpu_util)
        
        # Get CPU utilization
        cpu_util = get_cpu_utilization()
        cpu_utilization.append(cpu_util)
        
        # Get memory usage
        mem_usage = get_memory_usage()
        memory_usage.append(mem_usage)
        
        # Log every 5 seconds
        if int(current_time) % 5 == 0 and current_time > 0:
            logger.info(f"GPU: {gpu_util:.1f}%, CPU: {cpu_util:.1f}%, Memory: {mem_usage:.1f}MB")
        
        time.sleep(0.5)  # Sample every 500ms

def plot_utilization():
    """Plot the utilization metrics"""
    plt.figure(figsize=(15, 10))
    
    # Plot GPU utilization
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, gpu_utilization, 'r-')
    plt.title('GPU Utilization')
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Plot CPU utilization
    plt.subplot(3, 1, 2)
    plt.plot(timestamps, cpu_utilization, 'b-')
    plt.title('CPU Utilization')
    plt.ylabel('Utilization (%)')
    plt.ylim(0, 100)
    plt.grid(True)
    
    # Plot memory usage
    plt.subplot(3, 1, 3)
    plt.plot(timestamps, memory_usage, 'g-')
    plt.title('Memory Usage')
    plt.ylabel('Usage (MB)')
    plt.xlabel('Time (s)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gpu_training_verification.png')
    logger.info("Utilization plot saved to 'gpu_training_verification.png'")
    plt.close()


def verify_cuda_capability():
    """Verify that PyTorch can access CUDA and report device information"""
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


def verify_pybind_extension():
    """Verify that the PyBind11 CUDA extension is properly loaded"""
    print("\n=== PyBind11 Extension Check ===")
    
    if not HAS_GPU_ENV:
        print("ERROR: GPU environment extension not found!")
        print("You need to build the extension first:")
        print("    cd ai_platform_trainer/cpp")
        print("    python setup.py build_ext --inplace")
        return False
    
    try:
        # Create an environment to test
        env = make_env()
        print("PyBind11 extension loaded successfully!")
        
        # Check observation and action spaces
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Action shape: {env.action_space.shape}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to create environment: {e}")
        return False


def benchmark_performance(
    num_envs=4,
    train_steps=2000,
    benchmark_cpu=True
):
    """
    Benchmark performance comparison between CPU and GPU.
    
    Args:
        num_envs: Number of environments to run in parallel
        train_steps: Number of training steps to perform
        benchmark_cpu: Whether to run CPU benchmark for comparison
    """
    print("\n=== Performance Benchmark ===")
    
    results = {}
    
    # Start with GPU benchmark
    if torch.cuda.is_available():
        print("\nRunning GPU benchmark...")
        # Create environment
        env = make_env()
        
        # Create PPO model on GPU
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, 
            device="cuda"
        )
        
        # Start monitoring thread
        global stop_monitoring
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Time the training
        start_time = time.time()
        
        # Train the model
        model.learn(total_timesteps=train_steps)
        
        # Calculate duration
        duration = time.time() - start_time
        steps_per_second = train_steps / duration
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        # Plot the utilization
        plot_utilization()
        
        # Log results
        print(f"GPU Training completed in {duration:.2f} seconds")
        print(f"Steps per second: {steps_per_second:.2f}")
        
        # Store results
        results['gpu'] = {
            'duration': duration,
            'steps_per_second': steps_per_second,
            'avg_gpu_utilization': np.mean(gpu_utilization) if gpu_utilization else 0,
            'max_gpu_utilization': np.max(gpu_utilization) if gpu_utilization else 0
        }
    
    # CPU benchmark (optional)
    if benchmark_cpu:
        print("\nRunning CPU benchmark...")
        # Reset monitoring variables
        global gpu_utilization, cpu_utilization, memory_usage, timestamps
        gpu_utilization = []
        cpu_utilization = []
        memory_usage = []
        timestamps = []
        
        # Create environment
        env = make_env()
        
        # Create PPO model on CPU
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=0, 
            device="cpu"
        )
        
        # Start monitoring thread
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        # Time the training
        start_time = time.time()
        
        # Train the model
        model.learn(total_timesteps=train_steps)
        
        # Calculate duration
        duration = time.time() - start_time
        steps_per_second = train_steps / duration
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        # Plot the utilization
        plot_utilization()
        
        # Log results
        print(f"CPU Training completed in {duration:.2f} seconds")
        print(f"Steps per second: {steps_per_second:.2f}")
        
        # Store results
        results['cpu'] = {
            'duration': duration,
            'steps_per_second': steps_per_second,
            'avg_cpu_utilization': np.mean(cpu_utilization) if cpu_utilization else 0,
            'max_cpu_utilization': np.max(cpu_utilization) if cpu_utilization else 0
        }
    
    # Print comparison if both benchmarks were run
    if benchmark_cpu and torch.cuda.is_available() and 'gpu' in results and 'cpu' in results:
        speedup = results['cpu']['duration'] / results['gpu']['duration']
        print("\n=== Benchmark Comparison ===")
        print(f"GPU training was {speedup:.2f}x faster than CPU training")
        print(f"GPU: {results['gpu']['steps_per_second']:.2f} steps/s,")
        print(f"     avg utilization: {results['gpu']['avg_gpu_utilization']:.1f}%")
        print(f"CPU: {results['cpu']['steps_per_second']:.2f} steps/s,")
        print(f"     avg utilization: {results['cpu']['avg_cpu_utilization']:.1f}%")
    
    return results


def run_cuda_kernels_test():
    """
    Run a test that specifically exercises the CUDA kernels to verify
    they're being executed on the GPU.
    """
    print("\n=== CUDA Kernels Test ===")
    
    if not HAS_GPU_ENV:
        print("ERROR: GPU environment extension not found!")
        return False
    
    try:
        # Create environment
        env = make_env()
        
        # Create a bunch of missiles to test physics calculations
        missile_count = 100
        print(f"Creating environment with {missile_count} missiles...")
        
        # Reset to get initial state
        _ = env.reset()  # Initial observation not used
        
        # Step a few times to generate missiles
        steps = 0
        start_time = time.time()
        
        # Global variables for monitoring
        global stop_monitoring, gpu_utilization, cpu_utilization, memory_usage, timestamps
        gpu_utilization = []
        cpu_utilization = []
        memory_usage = []
        timestamps = []
        
        # Start monitoring
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        print("Running CUDA kernel test...")
        while steps < 500:
            # Take random action
            action = env.action_space.sample()
            _, reward, done, _, info = env.step(action)  # Observation not used
            
            # Print info every 100 steps
            if steps % 100 == 0:
                print(f"Step {steps}, missiles: {info.get('missile_count', 0)}")
            
            steps += 1
            if done:
                _ = env.reset()  # Observation not used
        
        # Calculate test duration (not used but kept for debugging)
        _ = time.time() - start_time
        
        # Stop monitoring
        stop_monitoring = True
        monitor_thread.join()
        
        # Plot utilization
        plot_utilization()
        
        # Check if GPU was utilized
        if gpu_utilization:
            avg_gpu = np.mean(gpu_utilization)
            max_gpu = np.max(gpu_utilization)
            print(f"Average GPU utilization: {avg_gpu:.1f}%")
            print(f"Maximum GPU utilization: {max_gpu:.1f}%")
            
            if max_gpu > 10.0:  # Threshold to consider GPU being used
                print("SUCCESS: CUDA kernels are running on the GPU!")
            else:
                print("WARNING: Low GPU utilization. CUDA kernels might not be using the GPU effectively.")
        else:
            print("WARNING: No GPU utilization data collected.")
        
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to run CUDA kernel test: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify GPU utilization during training")
    parser.add_argument('--cuda-check', action='store_true', help='Check CUDA capabilities')
    parser.add_argument('--extension-check', action='store_true', help='Check PyBind11 extension')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--cuda-test', action='store_true', help='Test CUDA kernels')
    parser.add_argument('--all', action='store_true', help='Run all checks')
    parser.add_argument('--train-steps', type=int, default=2000, help='Number of training steps')
    parser.add_argument('--no-cpu', action='store_true', help='Skip CPU benchmark')
    
    args = parser.parse_args()
    
    # If no specific check is requested, run all checks
    if not (args.cuda_check or args.extension_check or args.benchmark or args.cuda_test):
        args.all = True
    
    if args.all or args.cuda_check:
        verify_cuda_capability()
    
    if args.all or args.extension_check:
        verify_pybind_extension()
    
    if args.all or args.benchmark:
        benchmark_performance(
            train_steps=args.train_steps,
            benchmark_cpu=not args.no_cpu
        )
    
    if args.all or args.cuda_test:
        run_cuda_kernels_test()
