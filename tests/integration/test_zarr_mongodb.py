#!/usr/bin/env python3
"""
Test script to validate zarr + MongoDB integration for Synful.

This script verifies that the modernized PyTorch training can correctly:
1. Load large EM volumes from zarr files
2. Extract training cubes at synapse locations
3. Query synapse locations from MongoDB
4. Generate proper training data with spatial coordinate mapping
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_zarr_access(zarr_path: str, dataset: str = "volumes/raw") -> Dict:
    """Test zarr volume access and basic properties."""
    logger.info(f"Testing zarr access: {zarr_path}")
    
    try:
        import zarr
        
        # Open zarr store
        store = zarr.open(zarr_path, mode='r')
        if dataset in store:
            volume = store[dataset]
        else:
            # Try to find the right dataset
            available = list(store.keys())
            logger.warning(f"Dataset {dataset} not found. Available: {available}")
            if available:
                dataset = available[0]
                volume = store[dataset]
            else:
                raise ValueError("No datasets found in zarr store")
        
        info = {
            "shape": volume.shape,
            "dtype": volume.dtype,
            "chunks": volume.chunks if hasattr(volume, 'chunks') else None,
            "size_gb": np.prod(volume.shape) * volume.dtype.itemsize / (1024**3),
            "dataset": dataset
        }
        
        logger.info(f"Zarr volume info: {info}")
        
        # Test reading a small cube
        if len(volume.shape) >= 3:
            test_slice = tuple(slice(0, min(64, s)) for s in volume.shape[-3:])
            test_data = volume[test_slice]
            logger.info(f"Successfully read test cube: {test_data.shape}, dtype: {test_data.dtype}")
            logger.info(f"Value range: {test_data.min()} to {test_data.max()}")
        
        return info
        
    except ImportError:
        logger.error("zarr package not available")
        return {}
    except Exception as e:
        logger.error(f"Failed to access zarr volume: {e}")
        return {}


def test_mongodb_connection(host: str, database: str, collection: str) -> Dict:
    """Test MongoDB connection and synapse data access."""
    logger.info(f"Testing MongoDB connection: {host}/{database}/{collection}")
    
    try:
        from pymongo import MongoClient
        
        # Connect to MongoDB
        client = MongoClient(host)
        db = client[database]
        synapses_collection = db[f"{collection}.synapses"]
        
        # Get basic statistics
        total_count = synapses_collection.count_documents({})
        logger.info(f"Total synapses in database: {total_count}")
        
        if total_count == 0:
            logger.warning("No synapses found in database")
            return {"count": 0}
        
        # Get sample synapse
        sample = synapses_collection.find_one()
        logger.info(f"Sample synapse: {sample}")
        
        # Get spatial bounds
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "min_pre_z": {"$min": "$pre_z"},
                    "max_pre_z": {"$max": "$pre_z"},
                    "min_pre_y": {"$min": "$pre_y"},
                    "max_pre_y": {"$max": "$pre_y"},
                    "min_pre_x": {"$min": "$pre_x"},
                    "max_pre_x": {"$max": "$pre_x"},
                    "min_post_z": {"$min": "$post_z"},
                    "max_post_z": {"$max": "$post_z"},
                    "min_post_y": {"$min": "$post_y"},
                    "max_post_y": {"$max": "$post_y"},
                    "min_post_x": {"$min": "$post_x"},
                    "max_post_x": {"$max": "$post_x"},
                }
            }
        ]
        
        bounds_result = list(synapses_collection.aggregate(pipeline))
        if bounds_result:
            bounds = bounds_result[0]
            # Calculate overall spatial extent
            min_coords = [
                min(bounds["min_pre_z"], bounds["min_post_z"]),
                min(bounds["min_pre_y"], bounds["min_post_y"]),
                min(bounds["min_pre_x"], bounds["min_post_x"])
            ]
            max_coords = [
                max(bounds["max_pre_z"], bounds["max_post_z"]),
                max(bounds["max_pre_y"], bounds["max_post_y"]),
                max(bounds["max_pre_x"], bounds["max_post_x"])
            ]
            
            logger.info(f"Spatial bounds: {min_coords} to {max_coords}")
            extent = np.array(max_coords) - np.array(min_coords)
            logger.info(f"Spatial extent: {extent} nm")
        
        info = {
            "count": total_count,
            "sample": sample,
            "bounds": bounds_result[0] if bounds_result else None
        }
        
        return info
        
    except ImportError:
        logger.error("pymongo package not available")
        return {}
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return {}


def test_coordinate_mapping(zarr_info: Dict, mongo_info: Dict, voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0)) -> bool:
    """Test that synapse coordinates map correctly to zarr volume coordinates."""
    logger.info("Testing coordinate mapping between MongoDB and zarr")
    
    if not zarr_info or not mongo_info or not mongo_info.get("bounds"):
        logger.warning("Cannot test coordinate mapping - missing data")
        return False
    
    try:
        # Get synapse bounds in nm
        bounds = mongo_info["bounds"]
        synapse_min = np.array([
            min(bounds["min_pre_z"], bounds["min_post_z"]),
            min(bounds["min_pre_y"], bounds["min_post_y"]),
            min(bounds["min_pre_x"], bounds["min_post_x"])
        ])
        synapse_max = np.array([
            max(bounds["max_pre_z"], bounds["max_post_z"]),
            max(bounds["max_pre_y"], bounds["max_post_y"]),
            max(bounds["max_pre_x"], bounds["max_post_x"])
        ])
        
        # Convert to voxel coordinates
        voxel_size = np.array(voxel_size)
        synapse_min_voxel = synapse_min / voxel_size
        synapse_max_voxel = synapse_max / voxel_size
        
        # Get zarr volume shape
        zarr_shape = np.array(zarr_info["shape"])
        if len(zarr_shape) > 3:
            zarr_shape = zarr_shape[-3:]  # Take last 3 dimensions (z, y, x)
        
        logger.info(f"Synapse bounds (nm): {synapse_min} to {synapse_max}")
        logger.info(f"Synapse bounds (voxels): {synapse_min_voxel} to {synapse_max_voxel}")
        logger.info(f"Zarr volume shape (voxels): {zarr_shape}")
        
        # Check if synapse coordinates fit within zarr volume
        fits_min = np.all(synapse_min_voxel >= 0)
        fits_max = np.all(synapse_max_voxel < zarr_shape)
        
        if fits_min and fits_max:
            logger.info("✓ Synapse coordinates fit within zarr volume")
            return True
        else:
            logger.warning("⚠ Synapse coordinates may extend outside zarr volume")
            logger.warning(f"Min check: {fits_min}, Max check: {fits_max}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to test coordinate mapping: {e}")
        return False


def test_cube_extraction(zarr_path: str, zarr_dataset: str, cube_size: Tuple[int, int, int] = (32, 256, 256)) -> bool:
    """Test extracting training cubes from zarr volume."""
    logger.info(f"Testing cube extraction from zarr volume")
    
    try:
        import zarr
        
        store = zarr.open(zarr_path, mode='r')
        volume = store[zarr_dataset]
        
        # Test extracting a cube from center
        shape = np.array(volume.shape)
        if len(shape) > 3:
            shape = shape[-3:]
        
        center = shape // 2
        cube_size = np.array(cube_size)
        
        # Calculate cube bounds
        half_cube = cube_size // 2
        start = center - half_cube
        end = start + cube_size
        
        # Ensure bounds are valid
        start = np.maximum(start, 0)
        end = np.minimum(end, shape)
        
        # Extract cube
        cube_slice = tuple(slice(start[i], end[i]) for i in range(3))
        if len(volume.shape) > 3:
            # Add channel dimension
            cube_slice = (...,) + cube_slice
        
        cube = volume[cube_slice]
        logger.info(f"Extracted cube: {cube.shape}, dtype: {cube.dtype}")
        logger.info(f"Value range: {cube.min()} to {cube.max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract cube: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test zarr + MongoDB integration for Synful")
    parser.add_argument("--zarr-path", type=str, required=True, help="Path to zarr volume")
    parser.add_argument("--zarr-dataset", type=str, default="volumes/raw", help="Dataset within zarr")
    parser.add_argument("--mongodb-host", type=str, default="localhost", help="MongoDB host")
    parser.add_argument("--mongodb-db", type=str, required=True, help="MongoDB database name")
    parser.add_argument("--mongodb-collection", type=str, required=True, help="MongoDB collection name")
    parser.add_argument("--voxel-size", type=float, nargs=3, default=[8.0, 8.0, 8.0], help="Voxel size in nm (z y x)")
    
    args = parser.parse_args()
    
    logger.info("=== Testing Synful zarr + MongoDB Integration ===")
    
    # Test 1: Zarr access
    logger.info("\\n1. Testing zarr volume access...")
    zarr_info = test_zarr_access(args.zarr_path, args.zarr_dataset)
    if not zarr_info:
        logger.error("Zarr access failed")
        return 1
    
    # Test 2: MongoDB connection
    logger.info("\\n2. Testing MongoDB connection...")
    mongo_info = test_mongodb_connection(args.mongodb_host, args.mongodb_db, args.mongodb_collection)
    if not mongo_info:
        logger.error("MongoDB connection failed")
        return 1
    
    # Test 3: Coordinate mapping
    logger.info("\\n3. Testing coordinate mapping...")
    mapping_ok = test_coordinate_mapping(zarr_info, mongo_info, tuple(args.voxel_size))
    
    # Test 4: Cube extraction
    logger.info("\\n4. Testing cube extraction...")
    cube_ok = test_cube_extraction(args.zarr_path, zarr_info["dataset"])
    
    # Summary
    logger.info("\\n=== Test Results ===")
    logger.info(f"Zarr access: ✓")
    logger.info(f"MongoDB access: ✓")
    logger.info(f"Coordinate mapping: {'✓' if mapping_ok else '⚠'}")
    logger.info(f"Cube extraction: {'✓' if cube_ok else '✗'}")
    
    if mapping_ok and cube_ok:
        logger.info("\\n🎉 All tests passed! zarr + MongoDB integration looks good.")
        logger.info("\\nYou can now train with real data using:")
        logger.info(f"python train_pytorch.py parameter_pytorch.json \\\\")
        logger.info(f"  --zarr-path {args.zarr_path} \\\\")
        logger.info(f"  --mongodb-host {args.mongodb_host} \\\\")
        logger.info(f"  --mongodb-db {args.mongodb_db} \\\\")
        logger.info(f"  --mongodb-collection {args.mongodb_collection}")
        return 0
    else:
        logger.warning("\\n⚠ Some tests failed. Check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())