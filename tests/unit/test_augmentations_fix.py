#!/usr/bin/env python3
"""
Quick test script to verify the augmentation fixes work correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from synful.data.augmentations import SynapseAugmentations

def test_augmentations():
    """Test that the augmentations work without errors."""
    print("🧪 Testing fixed augmentations...")
    
    # Create test data
    raw_tensor = torch.randn(1, 1, 32, 64, 64)
    mask_tensor = torch.randint(0, 2, (1, 1, 32, 64, 64)).float()
    direction_tensor = torch.randn(1, 3, 32, 64, 64)
    
    print(f"Input shapes: raw={raw_tensor.shape}, mask={mask_tensor.shape}, directions={direction_tensor.shape}")
    
    # Create augmentation with all transforms enabled
    augmentations = SynapseAugmentations(
        rotation_range=(-15, 15),
        flip_probability=0.5,
        elastic_deformation=True,
        elastic_strength=2.0,
        intensity_scale_range=(0.9, 1.1),
        intensity_shift_range=(-0.1, 0.1),
        noise_std=0.05,
        apply_probability=1.0  # Always apply for testing
    )
    
    try:
        # Test augmentation
        result = augmentations(raw_tensor, mask_tensor, direction_tensor)
        
        print("✅ Augmentation completed successfully!")
        print(f"Output shapes: raw={result['raw'].shape}, mask={result['mask'].shape}, directions={result['direction'].shape}")
        
        # Verify shapes are preserved
        assert result['raw'].shape == raw_tensor.shape, f"Raw shape mismatch: {result['raw'].shape} != {raw_tensor.shape}"
        assert result['mask'].shape == mask_tensor.shape, f"Mask shape mismatch: {result['mask'].shape} != {mask_tensor.shape}"
        assert result['direction'].shape == direction_tensor.shape, f"Direction shape mismatch: {result['direction'].shape} != {direction_tensor.shape}"
        
        print("✅ All shape checks passed!")
        
        # Test multiple runs to ensure consistency
        print("🔄 Testing multiple augmentation runs...")
        for i in range(3):
            result = augmentations(raw_tensor, mask_tensor, direction_tensor)
            print(f"   Run {i+1}: ✅")
        
        print("🎉 All augmentation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_augmentations()
    sys.exit(0 if success else 1)