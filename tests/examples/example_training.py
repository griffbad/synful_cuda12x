#!/usr/bin/env python3
"""
Example training script for Synful PyTorch.

This script demonstrates how to set up and run training with the modernized
Synful package using synthetic data.

Usage:
    python example_training.py [--epochs 10] [--batch-size 2] [--output-dir ./training_results]
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

# Synful imports
from synful import (
    UNet3D, SynfulTrainer, SynfulLightningModule,
    Synapse, SynapseCollection,
    SynapseToMask, SynapseToDirectionVector,
    ToTensor, Normalize, SynfulAugmentations,
    create_default_configs
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticSynapseDataset(Dataset):
    """
    Synthetic dataset for demonstrating training.
    
    Generates random 3D volumes with simulated synaptic structures
    and corresponding ground truth masks and direction vectors.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        volume_shape: Tuple[int, int, int] = (32, 128, 128),
        voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        num_synapses_range: Tuple[int, int] = (5, 25),
        multitask: bool = True,
        augment: bool = True,
    ):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples in dataset
            volume_shape: Shape of each volume (z, y, x)
            voxel_size: Physical voxel size (z, y, x) in nm
            num_synapses_range: Range of number of synapses per volume
            multitask: Whether to generate direction vectors
            augment: Whether to apply data augmentation
        """
        self.num_samples = num_samples
        self.volume_shape = volume_shape
        self.voxel_size = voxel_size
        self.num_synapses_range = num_synapses_range
        self.multitask = multitask
        
        # Setup transforms
        self.mask_transform = SynapseToMask(radius=40.0, soft_boundary=True)
        if multitask:
            self.direction_transform = SynapseToDirectionVector(
                radius=80.0, normalize=True, falloff='linear'
            )
        
        self.to_tensor = ToTensor()
        self.normalize = Normalize(target_range=(-1, 1))
        
        # Setup augmentations
        if augment:
            self.augmentations = SynfulAugmentations(
                prob_geometric=0.8,
                prob_intensity=0.7,
                prob_noise=0.5
            )
        else:
            self.augmentations = None
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a single training sample."""
        # Generate synthetic volume with synapses
        volume, synapses = self._generate_volume_with_synapses()
        
        # Create synapse collection
        synapse_collection = SynapseCollection(synapses)
        
        # Generate ground truth mask
        mask = self.mask_transform(
            synapse_collection, 
            self.volume_shape, 
            self.voxel_size
        )
        
        # Generate direction vectors if multitask
        if self.multitask:
            direction = self.direction_transform(
                synapse_collection,
                self.volume_shape,
                self.voxel_size
            )
        
        # Convert to tensors
        data_dict = {'raw': volume, 'mask': mask}
        if self.multitask:
            data_dict['direction'] = direction
        
        tensor_data = self.to_tensor(data_dict)
        normalized_data = self.normalize(tensor_data)
        
        # Add batch and channel dimensions
        raw = normalized_data['raw'].unsqueeze(0)  # Add channel dim
        mask = normalized_data['mask'].unsqueeze(0)  # Add channel dim
        
        result = {'raw': raw, 'mask': mask}
        
        if self.multitask:
            # Direction already has 3 channels
            result['direction'] = normalized_data['direction']
        
        # Apply augmentations if enabled
        if self.augmentations is not None:
            result = self.augmentations(
                result['raw'].unsqueeze(0),  # Add batch dim for augmentation
                result['mask'].unsqueeze(0),
                result.get('direction', torch.zeros(1, 3, *self.volume_shape)).unsqueeze(0)
            )
            
            # Remove batch dimension after augmentation
            result = {k: v.squeeze(0) for k, v in result.items()}
        
        return result
    
    def _generate_volume_with_synapses(self) -> Tuple[np.ndarray, list]:
        """Generate a synthetic volume with synaptic structures."""
        # Create base volume with noise
        volume = np.random.randn(*self.volume_shape).astype(np.float32) * 0.1
        
        # Determine number of synapses
        num_synapses = np.random.randint(*self.num_synapses_range)
        synapses = []
        
        # Add synaptic structures
        for i in range(num_synapses):
            # Random presynaptic location (avoid edges)
            margin = 15
            pre_z = np.random.randint(margin, self.volume_shape[0] - margin)
            pre_y = np.random.randint(margin, self.volume_shape[1] - margin)
            pre_x = np.random.randint(margin, self.volume_shape[2] - margin)
            
            # Random postsynaptic location (nearby)
            offset = np.random.uniform(-10, 10, 3)
            post_z = int(np.clip(pre_z + offset[0], margin, self.volume_shape[0] - margin))
            post_y = int(np.clip(pre_y + offset[1], margin, self.volume_shape[1] - margin))
            post_x = int(np.clip(pre_x + offset[2], margin, self.volume_shape[2] - margin))
            
            # Add bright structures at synapse locations
            for (z, y, x) in [(pre_z, pre_y, pre_x), (post_z, post_y, post_x)]:
                # Create spherical structure
                radius = np.random.uniform(2, 5)
                intensity = np.random.uniform(0.5, 1.5)
                
                # Generate coordinate grid
                zz, yy, xx = np.ogrid[:self.volume_shape[0], :self.volume_shape[1], :self.volume_shape[2]]
                distance = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
                
                # Add Gaussian structure
                gaussian = intensity * np.exp(-0.5 * (distance / radius)**2)
                volume += gaussian
            
            # Create synapse object
            synapse = Synapse(
                id=i,
                location_pre=(
                    pre_z * self.voxel_size[0],
                    pre_y * self.voxel_size[1],
                    pre_x * self.voxel_size[2]
                ),
                location_post=(
                    post_z * self.voxel_size[0],
                    post_y * self.voxel_size[1],
                    post_x * self.voxel_size[2]
                ),
                score=np.random.uniform(0.7, 1.0),
                confidence=np.random.uniform(0.8, 1.0)
            )
            synapses.append(synapse)
        
        return volume, synapses


def create_dataloaders(
    train_samples: int = 80,
    val_samples: int = 20,
    batch_size: int = 2,
    num_workers: int = 2,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    
    # Training dataset with augmentation
    train_dataset = SyntheticSynapseDataset(
        num_samples=train_samples,
        augment=True,
        **dataset_kwargs
    )
    
    # Validation dataset without augmentation
    val_dataset = SyntheticSynapseDataset(
        num_samples=val_samples,
        augment=False,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Synful PyTorch Training Example")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--output-dir", default="./training_results", help="Output directory")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--no-multitask", action="store_true", help="Disable multitask learning")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting Synful training example")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    
    # Create configurations
    model_config, data_config, training_config = create_default_configs()
    
    # Update configs based on arguments
    model_config['multitask'] = not args.no_multitask
    model_config['base_features'] = 8  # Smaller for example
    model_config['depth'] = 3  # Smaller for example
    
    training_config['learning_rate'] = args.learning_rate
    
    # Create dataloaders
    logger.info("Creating synthetic datasets...")
    train_loader, val_loader = create_dataloaders(
        train_samples=40,  # Small for example
        val_samples=10,
        batch_size=args.batch_size,
        volume_shape=(32, 64, 64),  # Smaller for example
        multitask=model_config['multitask'],
        num_workers=0 if args.cpu else 2  # Use 0 workers if CPU only
    )
    
    logger.info(f"Created datasets: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create trainer
    trainer = SynfulTrainer(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        output_dir=output_dir,
        experiment_name="synful_example",
        use_wandb=False  # Disable wandb for example
    )
    
    # Setup trainer arguments
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'accelerator': 'cpu' if args.cpu else 'auto',
        'devices': 1,
        'precision': '32',  # Use 32-bit for stability in example
    }
    
    # Test one batch first
    logger.info("Testing data loading...")
    train_batch = next(iter(train_loader))
    logger.info(f"Batch shapes: raw {train_batch['raw'].shape}, mask {train_batch['mask'].shape}")
    if 'direction' in train_batch:
        logger.info(f"Direction shape: {train_batch['direction'].shape}")
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            **trainer_kwargs
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test the trained model
    logger.info("Testing trained model...")
    model = trainer.model.eval()
    
    with torch.no_grad():
        test_batch = next(iter(val_loader))
        outputs = model(test_batch['raw'])
        
        logger.info("Test inference successful!")
        logger.info(f"Output shapes: mask {outputs['mask_logits'].shape}")
        if 'direction_vectors' in outputs:
            logger.info(f"Direction vectors: {outputs['direction_vectors'].shape}")
    
    logger.info(f"Training example completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()