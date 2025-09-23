"""
GPU Architecture Detection and Configuration Utility for Synful CUDA 12.x
Optimized for NVIDIA 5090 and other modern GPUs
"""

import json
import os
import subprocess
import sys
from typing import Dict, Optional, Tuple


class GPUDetector:
    """Detect GPU architecture and recommend optimal parameters."""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu()
        
    def _detect_gpu(self) -> Dict:
        """Detect available GPU(s) and their specifications."""
        try:
            import tensorflow as tf
            
            # Get GPU information from TensorFlow
            gpus = tf.config.list_physical_devices('GPU')
            gpu_info = {
                'available': len(gpus) > 0,
                'count': len(gpus),
                'devices': []
            }
            
            if gpus:
                for i, gpu in enumerate(gpus):
                    try:
                        # Get GPU details
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        device_name = gpu_details.get('device_name', 'Unknown')
                        
                        # Estimate memory (this is a rough estimate)
                        memory_mb = self._estimate_gpu_memory(device_name)
                        
                        gpu_info['devices'].append({
                            'id': i,
                            'name': device_name,
                            'memory_mb': memory_mb,
                            'compute_capability': self._get_compute_capability(device_name)
                        })
                        
                    except Exception as e:
                        print(f"Warning: Could not get details for GPU {i}: {e}")
                        
            return gpu_info
            
        except ImportError:
            print("TensorFlow not available for GPU detection")
            return {'available': False, 'count': 0, 'devices': []}
            
    def _estimate_gpu_memory(self, device_name: str) -> int:
        """Estimate GPU memory based on device name."""
        device_name = device_name.lower()
        
        # NVIDIA RTX 5090 and similar high-end cards
        if '5090' in device_name:
            return 24000  # 24GB
        elif '4090' in device_name:
            return 24000  # 24GB
        elif '4080' in device_name:
            return 16000  # 16GB
        elif '3090' in device_name:
            return 24000  # 24GB
        elif '3080' in device_name:
            return 10000  # 10GB
        elif 'titan' in device_name:
            return 12000  # 12GB
        elif 'v100' in device_name:
            return 32000  # 32GB
        elif 'a100' in device_name:
            return 40000  # 40GB
        else:
            return 8000   # Conservative estimate
            
    def _get_compute_capability(self, device_name: str) -> str:
        """Get compute capability based on device name."""
        device_name = device_name.lower()
        
        if '5090' in device_name or '5080' in device_name:
            return "8.9"  # Ada Lovelace
        elif '4090' in device_name or '4080' in device_name:
            return "8.9"  # Ada Lovelace
        elif '3090' in device_name or '3080' in device_name:
            return "8.6"  # Ampere
        elif 'a100' in device_name:
            return "8.0"  # Ampere
        elif 'v100' in device_name:
            return "7.0"  # Volta
        else:
            return "7.5"  # Conservative estimate
            
    def get_optimal_parameters(self, base_parameters: Dict) -> Dict:
        """Get optimal parameters based on detected GPU."""
        if not self.gpu_info['available']:
            print("No GPU detected, using CPU-optimized parameters")
            return self._get_cpu_parameters(base_parameters)
            
        # Use the first GPU for parameter optimization
        primary_gpu = self.gpu_info['devices'][0]
        memory_mb = primary_gpu['memory_mb']
        device_name = primary_gpu['name']
        
        print(f"Optimizing for GPU: {device_name} ({memory_mb}MB)")
        
        # Create optimized parameters based on memory and architecture
        optimized = base_parameters.copy()
        
        if memory_mb >= 20000:  # High-end GPUs (5090, 4090, 3090, A100)
            optimized.update({
                "input_size": [64, 640, 640],
                "fmap_num": 12,
                "fmap_inc_factor": 6,
                "downsample_factors": [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
                "learning_rate": 1e-4,
                "_gpu_optimization": "high_memory_gpu"
            })
        elif memory_mb >= 15000:  # Mid-high GPUs (4080, 3080Ti)
            optimized.update({
                "input_size": [48, 512, 512],
                "fmap_num": 8,
                "fmap_inc_factor": 5,
                "downsample_factors": [[1, 2, 2], [1, 2, 2], [2, 2, 2]],
                "learning_rate": 8e-5,
                "_gpu_optimization": "mid_high_memory_gpu"
            })
        elif memory_mb >= 10000:  # Mid-range GPUs (3080, 3070Ti)
            optimized.update({
                "input_size": [42, 430, 430],
                "fmap_num": 6,
                "fmap_inc_factor": 4,
                "downsample_factors": [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                "learning_rate": 6e-5,
                "_gpu_optimization": "mid_memory_gpu"
            })
        else:  # Lower-end GPUs
            optimized.update({
                "input_size": [32, 320, 320],
                "fmap_num": 4,
                "fmap_inc_factor": 3,
                "downsample_factors": [[1, 3, 3], [1, 3, 3]],
                "learning_rate": 5e-5,
                "_gpu_optimization": "low_memory_gpu"
            })
            
        return optimized
        
    def _get_cpu_parameters(self, base_parameters: Dict) -> Dict:
        """Get CPU-optimized parameters."""
        cpu_params = base_parameters.copy()
        cpu_params.update({
            "input_size": [24, 240, 240],
            "fmap_num": 2,
            "fmap_inc_factor": 2,
            "downsample_factors": [[1, 3, 3], [1, 3, 3]],
            "learning_rate": 1e-5,
            "_gpu_optimization": "cpu_only"
        })
        return cpu_params
        
    def print_gpu_info(self):
        """Print detailed GPU information."""
        print("\n=== GPU Detection Results ===")
        if not self.gpu_info['available']:
            print("No GPUs detected. Running in CPU mode.")
            return
            
        print(f"Found {self.gpu_info['count']} GPU(s):")
        for gpu in self.gpu_info['devices']:
            print(f"  GPU {gpu['id']}: {gpu['name']}")
            print(f"    Memory: ~{gpu['memory_mb']}MB")
            print(f"    Compute Capability: {gpu['compute_capability']}")
            
    def save_optimized_config(self, output_path: str, base_config_path: str = None):
        """Save optimized configuration to file."""
        if base_config_path and os.path.exists(base_config_path):
            with open(base_config_path, 'r') as f:
                base_params = json.load(f)
        else:
            # Default parameters
            base_params = {
                "input_size": [42, 430, 430],
                "downsample_factors": [[1, 3, 3], [1, 3, 3], [3, 3, 3]],
                "fmap_num": 4,
                "fmap_inc_factor": 5,
                "unet_model": "vanilla",
                "learning_rate": 0.5e-4,
                "loss_comb_type": "sum",
                "m_loss_scale": 1.0,
                "d_loss_scale": 1.0,
                "reject_probability": 0.95,
                "blob_radius": 10,
                "max_iteration": 700000,
                "blob_mode": "ball",
                "d_scale": 1,
                "d_blob_radius": 100,
                "cliprange": [7e-4, 0.9993],
                "voxel_size": [40, 4, 4]
            }
            
        optimized_params = self.get_optimal_parameters(base_params)
        
        with open(output_path, 'w') as f:
            json.dump(optimized_params, f, indent=2)
            
        print(f"Optimized configuration saved to: {output_path}")


def main():
    """Main function for command-line usage."""
    detector = GPUDetector()
    detector.print_gpu_info()
    
    # Save optimized configuration
    base_config = "parameter.json" if os.path.exists("parameter.json") else None
    output_config = "parameter_optimized.json"
    
    detector.save_optimized_config(output_config, base_config)
    
    print(f"\nRecommended usage:")
    print(f"  python generate_network_tf2.py --config {output_config}")


if __name__ == "__main__":
    main()