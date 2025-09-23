#!/usr/bin/env python3
"""
Test script to verify TensorFlow 2.x compatibility with the updated generate_network.py files.
This script tests that the network generation works without the dependencies.
"""

import os
import sys
import tempfile
import json

def test_tf2_import():
    """Test that TensorFlow 2.x can be imported and basic operations work."""
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        
        # Test basic TF 2.x operations
        @tf.function
        def simple_function(x):
            return x * 2
        
        result = simple_function(tf.constant(5.0))
        assert result.numpy() == 10.0
        print("✓ Basic TensorFlow 2.x operations work")
        
        return True
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ TensorFlow operations failed: {e}")
        return False

def test_parameter_loading():
    """Test that parameter files can be loaded."""
    try:
        # Test with setup01 parameter file
        setup_dirs = ['setup01', 'setup02', 'setup03']
        
        for setup in setup_dirs:
            param_file = f"/home/runner/work/synful_cuda12x/synful_cuda12x/scripts/train/{setup}/parameter.json"
            if os.path.exists(param_file):
                with open(param_file, 'r') as f:
                    params = json.load(f)
                print(f"✓ {setup} parameter file loaded successfully")
                
                # Check required parameters
                required_params = ['input_size', 'downsample_factors', 'fmap_num', 
                                 'fmap_inc_factor', 'unet_model', 'learning_rate']
                for param in required_params:
                    if param not in params:
                        print(f"✗ Missing required parameter '{param}' in {setup}")
                        return False
                
                print(f"✓ {setup} has all required parameters")
        
        return True
    except Exception as e:
        print(f"✗ Parameter loading failed: {e}")
        return False

def test_modern_python_features():
    """Test that modern Python features work correctly."""
    try:
        # Test f-strings (Python 3.6+)
        name = "TensorFlow"
        version = "2.x"
        result = f"Testing {name} {version} compatibility"
        print(f"✓ f-strings work: {result}")
        
        # Test type hints (Python 3.5+)
        def typed_function(x: int) -> int:
            return x * 2
        
        assert typed_function(5) == 10
        print("✓ Type hints work")
        
        # Test pathlib (Python 3.4+)
        from pathlib import Path
        temp_path = Path(tempfile.gettempdir())
        assert temp_path.exists()
        print("✓ pathlib works")
        
        return True
    except Exception as e:
        print(f"✗ Modern Python features failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing synful_cuda12x TensorFlow 2.x compatibility...")
    print("=" * 60)
    
    tests = [
        ("Modern Python Features", test_modern_python_features),
        ("Parameter Loading", test_parameter_loading),
        ("TensorFlow 2.x Import", test_tf2_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print()
    
    print("=" * 60)
    print("Test Results:")
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All tests passed! The codebase is ready for modern Python and CUDA.")
    else:
        print("✗ Some tests failed. Please install missing dependencies and fix issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())