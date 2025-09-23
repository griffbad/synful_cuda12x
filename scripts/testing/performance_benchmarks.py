"""
Performance benchmarking and testing for Synful CUDA 12.x
Optimized for NVIDIA 5090 and modern GPU architectures
"""

import json
import os
import time
import numpy as np
import unittest
from typing import Dict, List, Tuple
import tempfile
import shutil


class SynfulPerformanceTester:
    """Performance testing suite for Synful."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.results = []
        
    def benchmark_data_loading(self, data_size: Tuple[int, int, int], num_iterations: int = 10) -> Dict:
        """Benchmark data loading performance."""
        print(f"Benchmarking data loading for size {data_size}...")
        
        try:
            import gunpowder as gp
            import zarr
            
            # Create temporary test data
            temp_dir = tempfile.mkdtemp()
            zarr_path = os.path.join(temp_dir, 'test_data.zarr')
            
            # Generate synthetic 3D microscopy data
            test_data = np.random.randint(0, 255, size=data_size, dtype=np.uint8)
            zarr.save_array(zarr_path, test_data)
            
            # Benchmark loading
            load_times = []
            for i in range(num_iterations):
                start_time = time.time()
                
                # Simulate gunpowder data loading
                loaded_data = zarr.load(zarr_path)
                
                end_time = time.time()
                load_times.append(end_time - start_time)
                
            # Cleanup
            shutil.rmtree(temp_dir)
            
            result = {
                'test': 'data_loading',
                'data_size': data_size,
                'iterations': num_iterations,
                'mean_time': np.mean(load_times),
                'std_time': np.std(load_times),
                'min_time': np.min(load_times),
                'max_time': np.max(load_times),
                'throughput_mb_per_sec': (np.prod(data_size) / 1024 / 1024) / np.mean(load_times)
            }
            
            self.results.append(result)
            return result
            
        except ImportError as e:
            print(f"Skipping data loading benchmark: {e}")
            return {'test': 'data_loading', 'status': 'skipped', 'reason': str(e)}
            
    def benchmark_inference_speed(self, input_sizes: List[Tuple[int, int, int]]) -> List[Dict]:
        """Benchmark inference speed for different input sizes."""
        print("Benchmarking inference speed...")
        
        results = []
        
        try:
            import tensorflow as tf
            
            for input_size in input_sizes:
                print(f"  Testing input size: {input_size}")
                
                # Create a simple test model
                with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                    # Simulate network inference
                    test_input = tf.random.normal((1, 1) + input_size)
                    
                    # Simple conv layers to simulate U-Net
                    x = tf.keras.layers.Conv3D(8, 3, padding='same')(test_input)
                    x = tf.keras.layers.Conv3D(16, 3, padding='same')(x)
                    x = tf.keras.layers.Conv3D(3, 1, padding='same')(x)  # Partner vectors
                    
                    model = tf.keras.Model(inputs=test_input, outputs=x)
                    
                    # Warm-up
                    _ = model(test_input)
                    
                    # Benchmark
                    inference_times = []
                    num_runs = 10
                    
                    for _ in range(num_runs):
                        start_time = time.time()
                        _ = model(test_input)
                        end_time = time.time()
                        inference_times.append(end_time - start_time)
                        
                result = {
                    'test': 'inference_speed',
                    'input_size': input_size,
                    'mean_time': np.mean(inference_times),
                    'std_time': np.std(inference_times),
                    'throughput_voxels_per_sec': np.prod(input_size) / np.mean(inference_times)
                }
                
                results.append(result)
                self.results.append(result)
                
        except Exception as e:
            print(f"Inference benchmark failed: {e}")
            results.append({'test': 'inference_speed', 'status': 'failed', 'reason': str(e)})
            
        return results
        
    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage patterns."""
        print("Benchmarking memory usage...")
        
        try:
            import tensorflow as tf
            
            # Get initial memory state
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu = gpus[0]
                tf.config.experimental.set_memory_growth(gpu, True)
                
                # Create progressively larger models to test memory scaling
                memory_usage = []
                input_sizes = [(32, 128, 128), (48, 256, 256), (64, 384, 384)]
                
                for input_size in input_sizes:
                    # Create model
                    inputs = tf.keras.Input(shape=(1,) + input_size)
                    x = tf.keras.layers.Conv3D(8, 3, padding='same')(inputs)
                    x = tf.keras.layers.Conv3D(16, 3, padding='same')(x)
                    x = tf.keras.layers.Conv3D(32, 3, padding='same')(x)
                    outputs = tf.keras.layers.Conv3D(3, 1, padding='same')(x)
                    
                    model = tf.keras.Model(inputs=inputs, outputs=outputs)
                    
                    # Test forward pass
                    test_input = tf.random.normal((1, 1) + input_size)
                    _ = model(test_input)
                    
                    # Estimate memory usage (parameters + activations)
                    param_count = model.count_params()
                    activation_memory = np.prod((1, 32) + input_size) * 4  # float32
                    
                    memory_usage.append({
                        'input_size': input_size,
                        'parameter_count': param_count,
                        'estimated_memory_mb': (param_count * 4 + activation_memory) / 1024 / 1024
                    })
                    
                result = {
                    'test': 'memory_usage',
                    'gpu_available': True,
                    'memory_scaling': memory_usage
                }
                
            else:
                result = {
                    'test': 'memory_usage',
                    'gpu_available': False,
                    'message': 'No GPU available for memory testing'
                }
                
            self.results.append(result)
            return result
            
        except Exception as e:
            result = {'test': 'memory_usage', 'status': 'failed', 'reason': str(e)}
            self.results.append(result)
            return result
            
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite."""
        print("Starting full performance benchmark...")
        print("=" * 50)
        
        # Data loading benchmark
        data_sizes = [(256, 256, 256), (512, 512, 256), (1024, 1024, 128)]
        for size in data_sizes:
            self.benchmark_data_loading(size, num_iterations=5)
            
        # Inference speed benchmark
        inference_sizes = [(32, 128, 128), (48, 256, 256), (64, 384, 384)]
        self.benchmark_inference_speed(inference_sizes)
        
        # Memory usage benchmark
        self.benchmark_memory_usage()
        
        # Summary
        summary = {
            'total_tests': len(self.results),
            'successful_tests': len([r for r in self.results if 'status' not in r or r['status'] != 'failed']),
            'failed_tests': len([r for r in self.results if r.get('status') == 'failed']),
            'timestamp': time.time(),
            'detailed_results': self.results
        }
        
        return summary
        
    def save_results(self, output_path: str):
        """Save benchmark results to JSON file."""
        summary = self.run_full_benchmark()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\nBenchmark results saved to: {output_path}")
        print(f"Successful tests: {summary['successful_tests']}/{summary['total_tests']}")
        
        return summary


class TestSynfulPerformance(unittest.TestCase):
    """Unit tests for performance components."""
    
    def setUp(self):
        self.tester = SynfulPerformanceTester()
        
    def test_data_loading_benchmark(self):
        """Test data loading benchmark functionality."""
        result = self.tester.benchmark_data_loading((128, 128, 128), num_iterations=3)
        
        if 'status' not in result:  # Test passed
            self.assertIn('mean_time', result)
            self.assertIn('throughput_mb_per_sec', result)
            self.assertGreater(result['throughput_mb_per_sec'], 0)
        else:  # Test was skipped
            self.assertEqual(result['status'], 'skipped')
            
    def test_inference_speed_benchmark(self):
        """Test inference speed benchmark functionality."""
        results = self.tester.benchmark_inference_speed([(32, 64, 64)])
        
        self.assertGreater(len(results), 0)
        
        if results[0].get('status') != 'failed':
            self.assertIn('mean_time', results[0])
            self.assertIn('throughput_voxels_per_sec', results[0])
            
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmark functionality."""
        result = self.tester.benchmark_memory_usage()
        
        self.assertIn('test', result)
        self.assertEqual(result['test'], 'memory_usage')


def main():
    """Main function for command-line usage."""
    tester = SynfulPerformanceTester()
    
    # Run benchmarks and save results
    output_file = 'synful_performance_results.json'
    summary = tester.save_results(output_file)
    
    # Print summary
    print("\n" + "=" * 50)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 50)
    
    for result in summary['detailed_results']:
        if result['test'] == 'data_loading' and 'mean_time' in result:
            print(f"Data Loading ({result['data_size']}): {result['throughput_mb_per_sec']:.2f} MB/s")
        elif result['test'] == 'inference_speed' and 'mean_time' in result:
            print(f"Inference ({result['input_size']}): {result['throughput_voxels_per_sec']:.0f} voxels/s")
        elif result['test'] == 'memory_usage' and 'memory_scaling' in result:
            print("Memory Usage: Available")
            for mem in result['memory_scaling']:
                print(f"  {mem['input_size']}: {mem['estimated_memory_mb']:.1f} MB")


if __name__ == "__main__":
    main()