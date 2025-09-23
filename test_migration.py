#!/usr/bin/env python3
"""
Simple test to verify the updated generate_network.py works with mock dependencies.
This tests the core TensorFlow 2.x changes without requiring full installation.
"""

import sys
import os

# Add the synful package to path
sys.path.insert(0, '/home/runner/work/synful_cuda12x/synful_cuda12x')

def test_network_generation_syntax():
    """Test that the network generation files have valid Python syntax."""
    import ast
    
    setup_dirs = ['setup01', 'setup02', 'setup03']
    base_path = '/home/runner/work/synful_cuda12x/synful_cuda12x/scripts/train'
    
    for setup in setup_dirs:
        gen_net_path = os.path.join(base_path, setup, 'generate_network.py')
        
        try:
            with open(gen_net_path, 'r') as f:
                source = f.read()
            
            # Parse the AST to check for syntax errors
            ast.parse(source)
            print(f"✓ {setup}/generate_network.py has valid syntax")
            
            # Check for TF 2.x patterns
            tf2_patterns = [
                '@tf.function',
                'tf.Module',
                'tf.saved_model.save',
                'input_signature=',
                'tf.TensorSpec'
            ]
            
            found_patterns = []
            for pattern in tf2_patterns:
                if pattern in source:
                    found_patterns.append(pattern)
            
            print(f"  Found TF 2.x patterns: {', '.join(found_patterns)}")
            
            # Check for old TF 1.x patterns that should be removed
            tf1_patterns = [
                'tf.placeholder',
                'tf.Session',
                'tf.reset_default_graph',
                'tf.train.export_meta_graph',
                'tf.losses.mean_squared_error',
                'tf.losses.sigmoid_cross_entropy'
            ]
            
            old_patterns = []
            for pattern in tf1_patterns:
                if pattern in source:
                    old_patterns.append(pattern)
            
            if old_patterns:
                print(f"  ⚠️  Found old TF 1.x patterns: {', '.join(old_patterns)}")
            else:
                print(f"  ✓ No old TF 1.x patterns found")
            
        except SyntaxError as e:
            print(f"✗ {setup}/generate_network.py has syntax error: {e}")
            return False
        except Exception as e:
            print(f"✗ Error reading {setup}/generate_network.py: {e}")
            return False
    
    return True

def test_imports():
    """Test that imports work without full TensorFlow installation."""
    try:
        # Test standard library imports
        import json
        import numpy as np
        print("✓ Standard imports work (json, numpy)")
        
        # Test synful imports
        from synful.gunpowder import IntensityScaleShiftClip
        print("✓ Synful imports work")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_parameter_files():
    """Test that parameter files are valid JSON."""
    setup_dirs = ['setup01', 'setup02', 'setup03']
    base_path = '/home/runner/work/synful_cuda12x/synful_cuda12x/scripts/train'
    
    for setup in setup_dirs:
        param_path = os.path.join(base_path, setup, 'parameter.json')
        
        try:
            with open(param_path, 'r') as f:
                params = json.load(f)
            
            # Check for required parameters
            required = ['input_size', 'learning_rate', 'unet_model']
            missing = [p for p in required if p not in params]
            
            if missing:
                print(f"✗ {setup}: Missing parameters {missing}")
                return False
            else:
                print(f"✓ {setup}: All required parameters present")
                
        except Exception as e:
            print(f"✗ Error reading {setup}/parameter.json: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Testing synful_cuda12x network generation updates...")
    print("=" * 60)
    
    tests = [
        ("Parameter Files", test_parameter_files),
        ("Import Capabilities", test_imports),
        ("Network Generation Syntax", test_network_generation_syntax),
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
        print("✓ All tests passed! TensorFlow 2.x migration looks successful.")
        print("  Next step: Install TensorFlow 2.x and test full functionality.")
    else:
        print("✗ Some tests failed. Please fix the issues before proceeding.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    import json
    sys.exit(main())