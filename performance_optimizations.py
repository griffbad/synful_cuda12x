"""
Performance optimizations for 3D microscopy volume processing on modern hardware
"""
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

def optimize_memory_layout(volume: np.ndarray) -> np.ndarray:
    """
    Optimize memory layout for better cache performance on modern CPUs/GPUs
    """
    # Ensure C-contiguous layout for better memory access patterns
    if not volume.flags.c_contiguous:
        volume = np.ascontiguousarray(volume)
    
    # Align memory to 64-byte boundaries for AVX-512 optimization
    if volume.dtype == np.float32:
        # Pad to ensure proper alignment
        pad_size = (64 - (volume.nbytes % 64)) % 64
        if pad_size > 0:
            flat = volume.flatten()
            padded = np.pad(flat, (0, pad_size // 4), mode='constant')
            volume = padded[:volume.size].reshape(volume.shape)
    
    return volume

def get_optimal_chunk_size(volume_shape: Tuple[int, ...], 
                          available_memory_gb: float = 22) -> Tuple[int, ...]:
    """
    Calculate optimal chunk size for processing large 3D volumes
    considering GPU memory constraints
    """
    # Convert GB to bytes
    available_bytes = available_memory_gb * 1024**3
    
    # Estimate memory usage per voxel (including intermediate computations)
    # Assume float32 (4 bytes) + overhead for gradients and intermediate values
    bytes_per_voxel = 4 * 8  # 8x overhead for safety
    
    # Calculate maximum voxels that fit in memory
    max_voxels = available_bytes // bytes_per_voxel
    
    # For 3D volumes, try to maintain aspect ratio while fitting in memory
    if len(volume_shape) == 3:
        z, y, x = volume_shape
        total_voxels = z * y * x
        
        if total_voxels <= max_voxels:
            return volume_shape
        
        # Scale down proportionally
        scale_factor = (max_voxels / total_voxels) ** (1/3)
        
        chunk_z = max(1, int(z * scale_factor))
        chunk_y = max(1, int(y * scale_factor))
        chunk_x = max(1, int(x * scale_factor))
        
        # Ensure chunks are divisible by 8 for optimal GPU processing
        chunk_z = (chunk_z // 8) * 8 or 8
        chunk_y = (chunk_y // 8) * 8 or 8
        chunk_x = (chunk_x // 8) * 8 or 8
        
        return (chunk_z, chunk_y, chunk_x)
    
    return volume_shape

def benchmark_operations(volume_shape: Tuple[int, ...], 
                        operations: Dict[str, Any],
                        num_iterations: int = 5) -> Dict[str, float]:
    """
    Benchmark different operations on synthetic data
    """
    # Create synthetic volume for benchmarking
    volume = np.random.random(volume_shape).astype(np.float32)
    volume = optimize_memory_layout(volume)
    
    results = {}
    
    for name, operation in operations.items():
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            try:
                result = operation(volume)
                # Ensure computation is completed (for GPU operations)
                if hasattr(result, 'numpy'):
                    _ = result.numpy()
                elif isinstance(result, np.ndarray):
                    _ = result.sum()  # Force computation
                
                end_time = time.time()
                times.append(end_time - start_time)
                
            except Exception as e:
                print(f"Operation {name} failed: {e}")
                times.append(float('inf'))
        
        # Use median time for more stable results
        results[name] = np.median(times)
    
    return results

def configure_tensorflow_for_performance():
    """
    Configure TensorFlow for optimal performance on modern hardware
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow not available, skipping GPU configuration")
        return None
    
    # Check for GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                # Enable memory growth
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit for RTX 5090 (22GB to leave room for system)
                tf.config.experimental.set_memory_limit(gpu, 22 * 1024)
                
            # Enable mixed precision for better performance
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            # Enable XLA JIT compilation
            tf.config.optimizer.set_jit(True)
            
            print(f"Configured {len(gpus)} GPU(s) for optimal performance")
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration failed: {e}")
            return False
    else:
        print("No GPUs found, using CPU")
        return False

class PerformanceProfiler:
    """
    Profile performance of synapse detection operations
    """
    
    def __init__(self):
        self.profile_data = {}
    
    def profile_operation(self, name: str, operation, *args, **kwargs):
        """
        Profile a single operation
        """
        start_time = time.time()
        result = operation(*args, **kwargs)
        end_time = time.time()
        
        self.profile_data[name] = {
            'duration': end_time - start_time,
            'timestamp': start_time
        }
        
        return result
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get performance summary
        """
        return {name: data['duration'] for name, data in self.profile_data.items()}
    
    def print_summary(self):
        """
        Print performance summary
        """
        print("\n=== Performance Summary ===")
        for name, duration in self.get_summary().items():
            print(f"{name}: {duration:.3f}s")
        
        total_time = sum(self.get_summary().values())
        print(f"Total time: {total_time:.3f}s")

# Initialize GPU configuration on import
if HAS_TENSORFLOW:
    configure_tensorflow_for_performance()