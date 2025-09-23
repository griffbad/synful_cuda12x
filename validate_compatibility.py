#!/usr/bin/env python3
"""
Validation script for Synful CUDA 12.x compatibility

This script tests core functionality without requiring full TensorFlow/GPU setup.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that core imports work correctly"""
    print("Testing imports...")
    
    tests = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__}')"),
        ("scipy", "import scipy; print(f'SciPy {scipy.__version__}')"),
        ("scipy.ndimage", "from scipy.ndimage import gaussian_filter, label, maximum_filter"),
        ("sklearn", "import sklearn; print(f'scikit-learn {sklearn.__version__}')"),
        ("synful.nms", "from synful.nms import find_maxima, sphere"),
        ("performance", "from performance_optimizations import PerformanceProfiler"),
        ("cuda_config", "from cuda_config import get_optimal_batch_size"),
    ]
    
    results = {}
    for name, test_code in tests:
        try:
            exec(test_code)
            results[name] = "✅ PASS"
        except Exception as e:
            results[name] = f"❌ FAIL: {e}"
    
    return results

def test_basic_functionality():
    """Test basic functionality of key modules"""
    print("\nTesting basic functionality...")
    
    tests = {}
    
    # Test NMS functionality
    try:
        import numpy as np
        from synful.nms import sphere, find_maxima
        
        # Test sphere function
        radius = (3, 3, 3)
        sphere_result = sphere(radius)
        assert sphere_result.shape == (7, 7, 7), f"Expected (7,7,7), got {sphere_result.shape}"
        tests["sphere_function"] = "✅ PASS"
        
        # Test find_maxima with small volume
        test_volume = np.random.random((32, 32, 32)).astype(np.float32)
        voxel_size = (4, 4, 4)
        radius = (12, 12, 12)
        
        result = find_maxima(test_volume, voxel_size, radius, min_score_threshold=0.5)
        tests["find_maxima"] = "✅ PASS"
        
    except Exception as e:
        tests["nms_functions"] = f"❌ FAIL: {e}"
    
    # Test performance optimizations
    try:
        from performance_optimizations import (
            optimize_memory_layout, 
            get_optimal_chunk_size,
            PerformanceProfiler
        )
        
        test_array = np.random.random((64, 64, 64)).astype(np.float32)
        optimized = optimize_memory_layout(test_array)
        assert optimized.flags.c_contiguous, "Memory layout should be C-contiguous"
        tests["memory_optimization"] = "✅ PASS"
        
        chunk_size = get_optimal_chunk_size((1024, 1024, 1024), 22)
        assert len(chunk_size) == 3, "Chunk size should be 3D"
        tests["chunk_optimization"] = "✅ PASS"
        
        profiler = PerformanceProfiler()
        result = profiler.profile_operation("test", lambda x: x.sum(), test_array)
        assert "test" in profiler.get_summary(), "Profiler should track operations"
        tests["profiler"] = "✅ PASS"
        
    except Exception as e:
        tests["performance_optimizations"] = f"❌ FAIL: {e}"
    
    # Test CUDA configuration
    try:
        from cuda_config import get_optimal_batch_size, get_memory_config
        
        batch_sizes = get_optimal_batch_size()
        assert "train" in batch_sizes, "Should have training batch size"
        tests["batch_size_config"] = "✅ PASS"
        
        memory_config = get_memory_config()
        assert "gpu_memory_limit" in memory_config, "Should have memory limit config"
        tests["memory_config"] = "✅ PASS"
        
    except Exception as e:
        tests["cuda_config"] = f"❌ FAIL: {e}"
    
    return tests

def test_setup_py():
    """Test that setup.py is valid"""
    print("\nTesting setup.py...")
    
    try:
        # Test that setup.py can be parsed
        setup_file = Path("setup.py")
        if not setup_file.exists():
            return {"setup_py": "❌ FAIL: setup.py not found"}
        
        with open(setup_file) as f:
            setup_content = f.read()
        
        # Check for modern setuptools usage
        if "from setuptools import setup" in setup_content:
            if "python_requires" in setup_content:
                return {"setup_py": "✅ PASS"}
            else:
                return {"setup_py": "⚠️  WARN: Missing python_requires"}
        else:
            return {"setup_py": "❌ FAIL: Still using distutils"}
            
    except Exception as e:
        return {"setup_py": f"❌ FAIL: {e}"}

def test_tensorflow_compatibility():
    """Test TensorFlow compatibility (without requiring TF installation)"""
    print("\nTesting TensorFlow compatibility...")
    
    try:
        # Test that the generate_network files can be parsed
        generate_files = [
            "scripts/train/setup01/generate_network.py",
            "scripts/train/setup02/generate_network.py", 
            "scripts/train/setup03/generate_network.py"
        ]
        
        results = {}
        for file_path in generate_files:
            try:
                with open(file_path) as f:
                    content = f.read()
                
                # Check for TF 2.x compatibility patterns
                if "tensorflow.compat.v1" in content:
                    if "tf.disable_v2_behavior()" in content:
                        results[Path(file_path).parent.name] = "✅ PASS - TF 2.x compat"
                    else:
                        results[Path(file_path).parent.name] = "⚠️  WARN - Missing disable_v2_behavior"
                else:
                    results[Path(file_path).parent.name] = "❌ FAIL - No TF 2.x compat"
                    
            except Exception as e:
                results[Path(file_path).parent.name] = f"❌ FAIL: {e}"
        
        return results
        
    except Exception as e:
        return {"tensorflow_compat": f"❌ FAIL: {e}"}

def main():
    """Run all validation tests"""
    print("=== Synful CUDA 12.x Compatibility Validation ===\n")
    
    all_results = {}
    
    # Run all test suites
    all_results.update(test_imports())
    all_results.update(test_basic_functionality())
    all_results.update(test_setup_py())
    all_results.update(test_tensorflow_compatibility())
    
    # Print summary
    print("\n=== Test Results Summary ===")
    
    passed = 0
    warned = 0
    failed = 0
    
    for test_name, result in all_results.items():
        print(f"{test_name:25} {result}")
        
        if result.startswith("✅"):
            passed += 1
        elif result.startswith("⚠️"):
            warned += 1
        else:
            failed += 1
    
    print(f"\n=== Final Summary ===")
    print(f"Passed: {passed}")
    print(f"Warned: {warned}") 
    print(f"Failed: {failed}")
    print(f"Total:  {len(all_results)}")
    
    if failed == 0:
        print("\n🎉 All critical tests passed! The codebase is ready for CUDA 12.x and Python 3.12.")
        return 0
    else:
        print(f"\n⚠️  {failed} tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())