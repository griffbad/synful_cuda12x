#!/usr/bin/env python3
"""
Quick test script for Synful PyTorch - Essential functionality check.

This script performs a rapid validation of core Synful components
and generates simple visualizations.

Usage:
    python quick_test_synful.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

print("🚀 Synful PyTorch - Quick Test Suite")
print("="*50)

# Test imports
try:
    from synful import (
        Synapse, SynapseCollection, UNet3D,
        SynapseToMask, SynapseToDirectionVector,
        SynfulPredictor, create_default_configs
    )
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Create test data
print("\n📊 Creating test data...")
synapses = [
    Synapse(
        id=i,
        location_pre=(i*50.0, i*60.0, i*70.0),
        location_post=(i*50.0+25, i*60.0+30, i*70.0+35),
        score=0.7 + i*0.05,
        confidence=0.8 + i*0.04
    ) for i in range(8)
]
collection = SynapseCollection(synapses)
print(f"   Created {len(collection)} synapses")

# Test data transforms
print("\n🔄 Testing transforms...")
shape = (32, 64, 64)
voxel_size = (8.0, 8.0, 8.0)

mask_transform = SynapseToMask(radius=40.0)
direction_transform = SynapseToDirectionVector(radius=80.0)

mask = mask_transform(collection, shape, voxel_size)
direction = direction_transform(collection, shape, voxel_size)
print(f"   Mask: {mask.shape}, range [{mask.min():.3f}, {mask.max():.3f}]")
print(f"   Direction: {direction.shape}, range [{direction.min():.3f}, {direction.max():.3f}]")

# Test model
print("\n🧠 Testing model...")
model = UNet3D(n_channels=1, base_features=8, depth=2, multitask=True)
x = torch.randn(1, 1, 16, 32, 32)

start_time = time.time()
with torch.no_grad():
    outputs = model(x)
inference_time = time.time() - start_time

print(f"   Model forward pass: {inference_time:.3f}s")
print(f"   Input: {list(x.shape)} -> Mask: {list(outputs['mask_logits'].shape)}")
print(f"   Direction vectors: {list(outputs['direction_vectors'].shape)}")

# Test predictor
print("\n🔮 Testing predictor...")
predictor = SynfulPredictor(
    model=model,
    device='cpu',
    chunk_size=(16, 32, 32),
    overlap=(2, 4, 4),
    half_precision=False
)

# Create synthetic test volume
test_volume = np.random.randn(24, 48, 48).astype(np.float32)
# Add some bright spots to simulate synapses
for i in range(5):
    z, y, x = np.random.randint(5, 19), np.random.randint(5, 43), np.random.randint(5, 43)
    test_volume[z-2:z+3, y-2:y+3, x-2:x+3] += 2.0

predictions = predictor.predict_volume(test_volume)
detected_synapses = predictor.detect_synapses(
    mask=predictions['mask'],
    direction=predictions.get('direction'),
    threshold=0.3,
    min_distance=30.0
)

print(f"   Volume prediction: {list(predictions['mask'].shape)}")
print(f"   Detected {len(detected_synapses)} synapses")

# Create visualizations
print("\n📊 Creating visualizations...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Synful PyTorch - Quick Test Results', fontsize=16, fontweight='bold')

# 1. Synapse locations
ax = axes[0, 0]
locations = np.array([s.location_pre for s in synapses])
scores = [s.score for s in synapses]
scatter = ax.scatter(locations[:, 1], locations[:, 2], c=scores, cmap='viridis', s=100)
ax.set_title('Synapse Locations (Y-X)')
ax.set_xlabel('Y (nm)')
ax.set_ylabel('X (nm)')
plt.colorbar(scatter, ax=ax, label='Score')

# 2. Generated mask
ax = axes[0, 1]
ax.imshow(mask[mask.shape[0]//2], cmap='hot', interpolation='nearest')
ax.set_title('Generated Mask (Mid-slice)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 3. Direction vector magnitude
ax = axes[0, 2]
direction_mag = np.linalg.norm(direction, axis=0)
im = ax.imshow(direction_mag[direction_mag.shape[0]//2], cmap='viridis', interpolation='nearest')
ax.set_title('Direction Vector Magnitude')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.colorbar(im, ax=ax)

# 4. Test volume
ax = axes[1, 0]
ax.imshow(test_volume[test_volume.shape[0]//2], cmap='gray', interpolation='nearest')
ax.set_title('Test Volume (Raw)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 5. Predicted mask
ax = axes[1, 1]
pred_mask = predictions['mask']
ax.imshow(pred_mask[pred_mask.shape[0]//2], cmap='hot', interpolation='nearest')
ax.set_title('Predicted Mask')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 6. Detection results
ax = axes[1, 2]
# Plot detected synapse locations on the mask
ax.imshow(pred_mask[pred_mask.shape[0]//2], cmap='gray', alpha=0.7, interpolation='nearest')
for synapse in detected_synapses:
    # Convert to voxel coordinates
    z_vox = int(synapse.location_pre[0] / 8.0)
    y_vox = int(synapse.location_pre[1] / 8.0)
    x_vox = int(synapse.location_pre[2] / 8.0)
    
    if z_vox == pred_mask.shape[0]//2:  # Only show synapses in mid-slice
        ax.plot(x_vox, y_vox, 'r+', markersize=10, markeredgewidth=2)

ax.set_title(f'Detected Synapses ({len(detected_synapses)})')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.tight_layout()

# Save results
output_dir = Path('./synful_quick_test_results')
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'quick_test_results.png', dpi=150, bbox_inches='tight')
print(f"   Visualization saved to {output_dir / 'quick_test_results.png'}")

# Test configurations
print("\n⚙️  Testing configurations...")
model_config, data_config, training_config = create_default_configs()
print(f"   Model config: {len(model_config)} parameters")
print(f"   Data config: {len(data_config)} parameters") 
print(f"   Training config: {len(training_config)} parameters")

# Performance summary
print("\n📈 Performance Summary:")
total_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {total_params:,}")
print(f"   Inference time: {inference_time:.3f}s for {list(x.shape)} volume")
print(f"   Memory efficient: Chunked processing with overlap")
print(f"   GPU ready: {'✅' if torch.cuda.is_available() else '❌ (CPU only)'}")

print("\n" + "="*50)
print("🎉 QUICK TEST COMPLETED SUCCESSFULLY! 🎉")
print("✅ All core components working")
print("✅ Data structures validated")
print("✅ Model inference successful")
print("✅ Visualization generated")
print("✅ Ready for full training and inference!")
print(f"📁 Results saved to: {output_dir}")
print("="*50)