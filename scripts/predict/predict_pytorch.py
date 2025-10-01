#!/usr/bin/env python3
"""
Modern PyTorch-based prediction script for synaptic partner detection.

This replaces the original TensorFlow 1.x prediction pipeline with a modern
PyTorch implementation while maintaining compatibility with the original
configuration format and blockwise processing approach.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import zarr

# Add the src directory to the path so we can import synful
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from synful import (
    UNet3D, SynfulPredictor,
    Synapse, SynapseCollection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: str, model_config: Dict[str, Any]) -> torch.nn.Module:
    """
    Load a trained PyTorch model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pth or .ckpt file)
        model_config: Model configuration for architecture
        
    Returns:
        Loaded PyTorch model in evaluation mode
    """
    # Create model with the specified configuration  
    # Filter out any unsupported parameters
    supported_params = {
        'n_channels', 'n_classes_mask', 'n_classes_vector', 'base_features',
        'depth', 'downsample_factors', 'multitask', 'bilinear', 'dropout', 'activation'
    }
    model_kwargs = {k: v for k, v in model_config.items() if k in supported_params}
    # Map dropout_rate to dropout if present
    if 'dropout_rate' in model_config:
        model_kwargs['dropout'] = model_config['dropout_rate']
    
    model = UNet3D(**model_kwargs)
    
    # Load checkpoint
    if checkpoint_path.endswith('.ckpt'):
        # Lightning checkpoint
        from synful.training import SynfulLightningModule
        lightning_model = SynfulLightningModule.load_from_checkpoint(
            checkpoint_path,
            model=model,  # Provide the model instance
            model_config=model_config,
            training_config=None  # Not needed for inference
        )
        model = lightning_model.model
    else:
        # Regular PyTorch checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_blockwise_pytorch(
    setup: str,
    iteration: int,
    checkpoint_path: str,
    raw_file: str,
    raw_dataset: str,
    out_directory: str,
    out_filename: str,
    chunk_size: tuple = (64, 512, 512),
    overlap: tuple = (8, 64, 64),
    device: str = "auto",
    output_datasets: Optional[Dict[str, Dict]] = None,
    threshold: float = 0.5,
    detect_synapses: bool = True
):
    """
    Run PyTorch-based prediction on a volume using blockwise processing.
    
    Args:
        setup: Name of the training setup
        iteration: Training iteration/epoch number
        checkpoint_path: Path to the trained model checkpoint
        raw_file: Input raw data file path
        raw_dataset: Dataset name within the file
        out_directory: Output directory
        out_filename: Output filename
        chunk_size: Size of processing chunks (D, H, W)
        overlap: Overlap between chunks for blending (D, H, W)
        device: Device to use ('cuda', 'cpu', or 'auto')
        output_datasets: Configuration for output datasets
        threshold: Threshold for synapse detection
        detect_synapses: Whether to detect individual synapses
    """
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Using device: {device}")
    
    # Load input data
    logger.info(f"Loading input data from {raw_file}:{raw_dataset}")
    
    if raw_file.endswith('.zarr'):
        input_array = zarr.open(raw_file, mode='r')[raw_dataset]
    elif raw_file.endswith('.hdf') or raw_file.endswith('.h5'):
        import h5py
        with h5py.File(raw_file, 'r') as f:
            input_array = f[raw_dataset][:]
    else:
        raise ValueError(f"Unsupported file format: {raw_file}")
    
    input_volume = np.array(input_array)
    logger.info(f"Input volume shape: {input_volume.shape}")
    
    # Create model configuration (you might want to load this from a config file)
    model_config = {
        "n_channels": 1,
        "base_features": 4,  # This should match your training setup
        "depth": 4,
        "multitask": True,
        "activation": "ReLU",
        "dropout": 0.1
    }
    
    # Load trained model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(checkpoint_path, model_config)
    
    # Create predictor
    predictor = SynfulPredictor(
        model=model,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device
    )
    
    # Run prediction
    logger.info("Running prediction...")
    start_time = time.time()
    
    predictions = predictor.predict_volume(input_volume)
    
    prediction_time = time.time() - start_time
    logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
    
    # Setup output file
    output_path = Path(out_directory) / setup / str(iteration) / out_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to {output_path}")
    
    # Save predictions
    if output_path.suffix == '.zarr':
        # Save as Zarr
        output_zarr = zarr.open(str(output_path), mode='w')
        
        # Save mask predictions
        mask_data = predictions['mask'].squeeze().astype(np.float32)
        output_zarr.create_dataset(
            'volumes/pred_syn_indicator',
            data=mask_data,
            chunks=chunk_size,
            compression='gzip',
            compression_opts=5
        )
        
        # Save direction vector predictions
        if 'directions' in predictions:
            direction_data = predictions['directions'].astype(np.float32)
            output_zarr.create_dataset(
                'volumes/pred_partner_vectors',
                data=direction_data,
                chunks=(3,) + chunk_size,
                compression='gzip',
                compression_opts=5
            )
        
    elif output_path.suffix in ['.hdf', '.h5']:
        # Save as HDF5
        import h5py
        with h5py.File(str(output_path), 'w') as f:
            # Save mask predictions
            f.create_dataset(
                'volumes/pred_syn_indicator',
                data=predictions['mask'].squeeze().astype(np.float32),
                compression='gzip'
            )
            
            # Save direction vector predictions
            if 'directions' in predictions:
                f.create_dataset(
                    'volumes/pred_partner_vectors',
                    data=predictions['directions'].astype(np.float32),
                    compression='gzip'
                )
    
    # Detect synapses if requested
    if detect_synapses:
        logger.info("Detecting individual synapses...")
        synapses = predictor.detect_synapses(
            predictions['mask'].squeeze(),
            threshold=threshold
        )
        
        logger.info(f"Detected {len(synapses)} synapses")
        
        # Save synapse locations
        synapse_file = output_path.with_name(output_path.stem + '_synapses.json')
        synapse_collection = SynapseCollection(synapses=synapses)
        
        with open(synapse_file, 'w') as f:
            json.dump(synapse_collection.to_dict(), f, indent=2)
        
        logger.info(f"Saved synapse locations to {synapse_file}")
    
    logger.info("Prediction pipeline completed successfully!")
    
    return str(output_path)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Run PyTorch-based synaptic partner prediction")
    parser.add_argument(
        "config_file",
        help="Path to prediction configuration JSON file"
    )
    parser.add_argument(
        "--checkpoint",
        help="Override checkpoint path from config"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for prediction"
    )
    parser.add_argument(
        "--chunk-size",
        nargs=3,
        type=int,
        default=[64, 512, 512],
        help="Chunk size for processing (D H W)"
    )
    parser.add_argument(
        "--overlap",
        nargs=3,
        type=int,
        default=[8, 64, 64],
        help="Overlap between chunks (D H W)"
    )
    parser.add_argument(
        "--no-synapses",
        action="store_true",
        help="Skip synapse detection step"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for synapse detection"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config_file}")
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {args.config_file} not found")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        return 1
    
    # Extract configuration parameters
    setup = config.get('setup', 'setup03')
    iteration = config.get('iteration', 90000)
    raw_file = config.get('raw_file')
    raw_dataset = config.get('raw_dataset', 'volumes/raw')
    out_directory = config.get('out_directory', 'output/')
    out_filename = config.get('out_filename', 'predictions.zarr')
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Default checkpoint path based on setup and iteration
        checkpoint_path = f"../train/{setup}/checkpoints/epoch_{iteration}.ckpt"
        if not os.path.exists(checkpoint_path):
            # Try alternative path
            checkpoint_path = f"../train/{setup}/model_{iteration}.pth"
            if not os.path.exists(checkpoint_path):
                logger.error(f"No checkpoint found at {checkpoint_path}")
                logger.info("Please specify checkpoint path with --checkpoint")
                return 1
    
    logger.info(f"Configuration loaded: setup={setup}, iteration={iteration}")
    logger.info(f"Input: {raw_file}:{raw_dataset}")
    logger.info(f"Output: {out_directory}/{setup}/{iteration}/{out_filename}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    # Run prediction
    try:
        output_path = predict_blockwise_pytorch(
            setup=setup,
            iteration=iteration,
            checkpoint_path=checkpoint_path,
            raw_file=raw_file,
            raw_dataset=raw_dataset,
            out_directory=out_directory,
            out_filename=out_filename,
            chunk_size=tuple(args.chunk_size),
            overlap=tuple(args.overlap),
            device=args.device,
            threshold=args.threshold,
            detect_synapses=not args.no_synapses
        )
        
        logger.info(f"Prediction completed successfully! Output saved to: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())