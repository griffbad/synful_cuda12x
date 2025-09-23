"""
CUDA 12.x and NVIDIA RTX 5090 optimized configurations for Synful
"""
import os

def configure_cuda_for_rtx5090():
    """
    Configure CUDA settings optimized for NVIDIA RTX 5090 GPUs
    """
    # Enable memory growth to avoid allocating all GPU memory at once
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Set compute capability for RTX 5090 (Ada Lovelace architecture)
    # RTX 5090 uses compute capability 8.9
    os.environ['CUDA_COMPUTE_CAPABILITY'] = '8.9'
    
    # Enable TensorFloat-32 (TF32) for better performance on Ampere/Ada Lovelace
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    
    # Optimize memory allocation
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Enable XLA JIT compilation for better performance
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    # Set optimal number of threads for RTX 5090
    os.environ['TF_NUM_INTEROP_THREADS'] = str(os.cpu_count())
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(os.cpu_count())
    
    # Enable mixed precision training for better performance
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

def get_optimal_batch_size():
    """
    Get optimal batch size for RTX 5090 with 24GB VRAM
    """
    return {
        'train': 4,  # Conservative for 3D volumes
        'predict': 8,  # Can be larger for inference
        'large_volumes': 2  # For very large 3D microscopy volumes
    }

def get_memory_config():
    """
    Memory configuration optimized for RTX 5090 24GB VRAM
    """
    return {
        'gpu_memory_limit': 22 * 1024,  # Reserve 2GB for system
        'allow_growth': True,
        'cache_size_gb': 4  # For data caching
    }

# CUDA 12.x specific optimizations
def enable_cuda12_optimizations():
    """
    Enable CUDA 12.x specific optimizations
    """
    # Use CUDA 12.x optimized libraries
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    
    # Enable CUDA graphs for better performance
    os.environ['TF_USE_CUDA_GRAPHS'] = '1'
    
    # Enable unified memory for large datasets
    os.environ['CUDA_MANAGED_FORCE_DEVICE_ALLOC'] = '1'
    
    # Optimize for Ada Lovelace architecture
    os.environ['NVIDIA_TF32_OVERRIDE'] = '1'

# Call configuration on import
configure_cuda_for_rtx5090()
enable_cuda12_optimizations()