#!/usr/bin/env python3
"""
Modern PyTorch-based training script for synaptic partner detection.

This replaces the original TensorFlow 1.x training pipeline with a modern
PyTorch Lightning implementation while maintaining compatibility with the
original parameter.json configuration.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import lightning as L
import numpy as np
from torch.utils.data import DataLoader

# Add the src directory to the path so we can import synful
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synful import (
    UNet3D, SynfulTrainer, SynfulLightningModule,
    create_default_configs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_legacy_parameters(legacy_params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Convert legacy parameter.json to modern configuration dictionaries.
    
    Args:
        legacy_params: Original parameter.json configuration or modern nested format
        
    Returns:
        Tuple of (model_config, data_config, training_config)
    """
    
    # Check if this is already in the new format
    if "model" in legacy_params and "data" in legacy_params and "training" in legacy_params:
        # Already in modern format, just return the sections
        return legacy_params["model"], legacy_params["data"], legacy_params["training"]
    
    # Convert from old flat format
    # Convert input size from [D, H, W] to modern format
    input_size = legacy_params["input_size"]
    voxel_size = legacy_params["voxel_size"]
    
    # Calculate features based on legacy parameters
    base_features = legacy_params["fmap_num"]
    depth = len(legacy_params["downsample_factors"])
    
    # Determine if multitask based on unet_model
    multitask = legacy_params["unet_model"] == "dh_unet"
    
    # Create model configuration
    model_config = {
        "n_channels": 1,
        "base_features": base_features,
        "depth": depth,
        "multitask": multitask,
        "activation": "ReLU",
        "dropout": 0.1
    }
    
    # Create data configuration
    data_config = {
        "input_size": tuple(input_size),
        "voxel_size": tuple(voxel_size),
        "augmentation": {
            "elastic": True,
            "intensity": True,
            "spatial": True,
            "noise": False
        },
        "reject_probability": legacy_params["reject_probability"],
        "batch_size": 1,  # Original used batch size 1
        "num_workers": 4
    }
    
    # Create training configuration
    training_config = {
        "max_epochs": legacy_params["max_iteration"] // 1000,  # Convert iterations to epochs
        "learning_rate": legacy_params["learning_rate"],
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "loss_weights": {
            "mask": legacy_params["m_loss_scale"],
            "direction": legacy_params["d_loss_scale"]
        },
        "val_check_interval": 1000,  # Check validation every 1000 iterations
        "save_every": 30000,  # Save checkpoint every 30k iterations
        "gradient_clip_val": 1.0,
        "early_stopping_patience": 50,
        "precision": "16-mixed"  # Use mixed precision for efficiency
    }
    
    return model_config, data_config, training_config


def create_real_dataloader(data_config: Dict[str, Any], zarr_path: str, synapse_source: Dict[str, str]) -> DataLoader:
    """
    Create a real dataloader using zarr volumes and synapse data (MongoDB or TSV).
    
    This integrates with the original Synful data pipeline for production training.
    """
    try:
        from synful.data.mongodb_source import ZarrMongoDBDataSource
        from synful.data.tsv_source import TSVSynapseSource
        from gunpowder import ArrayKey, GraphKey
        logger.info(f"Creating real data loader with zarr: {zarr_path}, synapses: {synapse_source}")
        
        # This would integrate with gunpowder pipeline
        # For now, fall back to synthetic data until gunpowder integration is complete
        if synapse_source.get('type') == 'mongodb':
            logger.warning("Real zarr+MongoDB integration requires gunpowder - using synthetic data")
        elif synapse_source.get('type') == 'tsv':
            logger.warning("Real zarr+TSV integration requires gunpowder - using synthetic data")
        else:
            logger.warning("Unknown synapse source type - using synthetic data")
            
        return create_synthetic_dataloader(data_config, num_samples=1000)
        
    except ImportError as e:
        logger.warning(f"Real data loading dependencies not available: {e}")
        logger.warning("Using synthetic data instead")
        return create_synthetic_dataloader(data_config, num_samples=1000)


def create_synthetic_dataloader(data_config: Dict[str, Any], num_samples: int = 1000) -> DataLoader:
    """
    Create a synthetic dataloader for demonstration purposes.
    
    In a real scenario, you would replace this with your actual data loading logic
    that reads from HDF5 files, applies gunpowder augmentations, etc.
    """
    from torch.utils.data import Dataset
    
    class SyntheticSynapseDataset(Dataset):
        def __init__(self, num_samples: int, input_size: tuple):
            self.num_samples = num_samples
            self.input_size = input_size
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Create synthetic volume
            volume = torch.randn(1, *self.input_size)
            
            # Create synthetic mask (sparse synaptic locations)
            mask = torch.zeros(1, *self.input_size)
            # Add some random synaptic locations
            num_synapses = np.random.randint(5, 20)
            for _ in range(num_synapses):
                z = np.random.randint(5, self.input_size[0] - 5)
                y = np.random.randint(5, self.input_size[1] - 5)
                x = np.random.randint(5, self.input_size[2] - 5)
                mask[0, z-2:z+3, y-2:y+3, x-2:x+3] = 1.0
            
            # Create synthetic direction vectors
            directions = torch.randn(3, *self.input_size) * mask
            
            return {
                'raw': volume,
                'mask': mask,
                'direction': directions
            }
    
    dataset = SyntheticSynapseDataset(num_samples, data_config["input_size"])
    return DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True
    )


def setup_data_directories():
    """Create necessary directories for training outputs."""
    directories = [
        "checkpoints",
        "logs", 
        "tensorboard",
        "snapshots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info(f"Created training directories: {', '.join(directories)}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train synaptic partner detection model")
    parser.add_argument(
        "parameter_file",
        default="parameter.json",
        nargs='?',
        help="Path to parameter.json file (default: parameter.json)"
    )
    parser.add_argument(
        "--data-dir",
        default="../../../../../data/cremi/",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing (no real data required)"
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        help="Path to zarr volume for real data training"
    )
    parser.add_argument(
        "--mongodb-host",
        type=str,
        default="localhost",
        help="MongoDB host for synapse data"
    )
    parser.add_argument(
        "--mongodb-db",
        type=str,
        help="MongoDB database name"
    )
    parser.add_argument(
        "--mongodb-collection",
        type=str,
        help="MongoDB collection name"
    )
    parser.add_argument(
        "--tsv-synapses",
        type=str,
        help="Path to TSV file with synapse locations (pre_x,pre_y,pre_z,post_x,post_y,post_z)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override max epochs from parameter file"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=1000,
        help="Save snapshots every N training steps (default: 1000)"
    )
    parser.add_argument(
        "--keep-snapshots",
        type=int,
        default=5,
        help="Number of snapshots to keep (default: 5)"
    )
    parser.add_argument(
        "--resume-snapshot",
        type=str,
        help="Resume training from snapshot file (use 'latest' for most recent)"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup directories
    setup_data_directories()
    
    # Load and convert parameters
    logger.info(f"Loading parameters from {args.parameter_file}")
    try:
        with open(args.parameter_file, 'r') as f:
            legacy_params = json.load(f)
    except FileNotFoundError:
        logger.error(f"Parameter file {args.parameter_file} not found")
        return 1
    
    logger.info("Converting legacy parameters to modern configuration")
    model_config, data_config, training_config = convert_legacy_parameters(legacy_params)
    
    # Override epochs if specified
    if args.epochs:
        training_config["max_epochs"] = args.epochs
        logger.info(f"Overriding max epochs to {args.epochs}")
        
    # Add snapshot configuration
    training_config["snapshot_every"] = args.snapshot_every
    training_config["keep_snapshots"] = args.keep_snapshots
    
    # Log configurations
    logger.info(f"Model config: {model_config}")
    logger.info(f"Data config: {data_config}")
    logger.info(f"Training config: {training_config}")
    
    # Setup device
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
        logger.info(f"Using GPU: {device}")
    else:
        device = "cpu"
        logger.warning("CUDA not available, using CPU")
    
    # Create data loaders
    if args.synthetic:
        logger.info("Creating synthetic data loaders")
        train_loader = create_synthetic_dataloader(data_config, num_samples=1000)
        val_loader = create_synthetic_dataloader(data_config, num_samples=200)
    elif args.zarr_path and args.mongodb_db and args.mongodb_collection:
        logger.info("Creating real data loaders with zarr + MongoDB")
        synapse_source = {
            "type": "mongodb",
            "host": args.mongodb_host,
            "database": args.mongodb_db,
            "collection": args.mongodb_collection
        }
        train_loader = create_real_dataloader(data_config, args.zarr_path, synapse_source)
        val_loader = create_real_dataloader(data_config, args.zarr_path, synapse_source)
    elif args.zarr_path and args.tsv_synapses:
        logger.info("Creating real data loaders with zarr + TSV synapses")
        synapse_source = {
            "type": "tsv",
            "path": args.tsv_synapses
        }
        train_loader = create_real_dataloader(data_config, args.zarr_path, synapse_source)
        val_loader = create_real_dataloader(data_config, args.zarr_path, synapse_source)
    else:
        logger.warning("No data source specified")
        logger.warning("Use --synthetic for testing, --zarr-path + --mongodb-* for MongoDB, or --zarr-path + --tsv-synapses for TSV")
        logger.warning("Falling back to synthetic data")
        train_loader = create_synthetic_dataloader(data_config, num_samples=1000)
        val_loader = create_synthetic_dataloader(data_config, num_samples=200)
    
    # Initialize trainer
    logger.info("Initializing PyTorch Lightning trainer")
    trainer = SynfulTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        output_dir="./",
        experiment_name="setup03_modernized"
    )
    
    # Handle snapshot resuming
    resume_from_checkpoint = None
    if args.resume_snapshot:
        try:
            resume_from_checkpoint = trainer.resume_from_snapshot(args.resume_snapshot)
            logger.info(f"🔄 Will resume training from snapshot: {resume_from_checkpoint}")
        except Exception as e:
            logger.error(f"Failed to resume from snapshot: {e}")
            return 1
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Training for {training_config['max_epochs']} epochs")
    
    try:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            max_epochs=training_config["max_epochs"],
            resume_from_checkpoint=resume_from_checkpoint
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {trainer.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())