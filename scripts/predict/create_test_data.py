#!/usr/bin/env python3
"""
Test script for PyTorch prediction pipeline.

Creates synthetic test data and runs prediction to verify the complete pipeline works.
"""

import logging
import sys
import zarr
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_volume(output_path: str, shape=(64, 256, 256)):
    """Create a synthetic test volume for prediction."""
    logger.info(f"Creating test volume: {output_path} with shape {shape}")
    
    # Create zarr group
    root = zarr.open(output_path, mode='w')
    
    # Create synthetic raw data - add some structure
    raw_data = np.random.randint(50, 200, shape, dtype=np.uint8)
    
    # Add some blob-like structures that might look like synapses
    for i in range(10):
        z = np.random.randint(5, shape[0]-5)
        y = np.random.randint(20, shape[1]-20)
        x = np.random.randint(20, shape[2]-20)
        
        # Create a small bright blob
        for dz in range(-2, 3):
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if 0 <= z+dz < shape[0] and 0 <= y+dy < shape[1] and 0 <= x+dx < shape[2]:
                        distance = np.sqrt(dz**2 + dy**2 + dx**2)
                        if distance <= 3:
                            raw_data[z+dz, y+dy, x+dx] = min(255, int(200 + 50 * np.exp(-distance)))
    
    # Store the raw data
    raw_dataset = root.create_dataset(
        'raw',
        data=raw_data,
        chunks=(32, 128, 128),
        compression='gzip'
    )
    
    # Add metadata
    raw_dataset.attrs['voxel_size'] = [40, 4, 4]  # nm per voxel
    raw_dataset.attrs['offset'] = [0, 0, 0]
    raw_dataset.attrs['description'] = 'Synthetic test volume for prediction'
    
    logger.info(f"✅ Test volume created: {output_path}")
    logger.info(f"   Shape: {shape}")
    logger.info(f"   Data range: {raw_data.min()} - {raw_data.max()}")
    
    return output_path

def main():
    """Run prediction test."""
    logger.info("🧪 Testing Synful PyTorch Prediction Pipeline")
    logger.info("=" * 50)
    
    # Create test data
    test_volume_path = "test_volume.zarr"
    create_test_volume(test_volume_path)
    
    # Create output directory
    Path("test_prediction_output").mkdir(exist_ok=True)
    
    logger.info("📊 Test volume created successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run prediction:")
    logger.info("   python predict_pytorch.py test_prediction_config.json")
    logger.info("")
    logger.info("2. Check outputs in: test_prediction_output/")
    logger.info("")
    logger.info("3. For debugging:")
    logger.info("   python predict_pytorch.py test_prediction_config.json --debug")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())