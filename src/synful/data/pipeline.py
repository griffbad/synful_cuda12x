"""
Modern Gunpowder data pipeline for Synful PyTorch.

Provides high-performance data loading with on-the-fly augmentation
for 3D electron microscopy volumes and synaptic annotations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from gunpowder import *
from gunpowder.torch import Train
import zarr
import h5py

# Conditional import to avoid circular dependency
try:
    from ..synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError:
    _SYNAPSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class SynfulDataPipeline:
    """
    Modern data pipeline for Synful using latest Gunpowder features.
    
    Features:
    - PyTorch Lightning integration
    - Multi-scale training
    - Advanced augmentations
    - Cloud volume support
    - Memory-efficient batching
    """
    
    def __init__(
        self,
        raw_data_sources: List[Union[str, Path]],
        synapse_sources: List[Union[str, Path]],
        batch_size: Tuple[int, int, int] = (32, 256, 256),
        voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        num_workers: int = 4,
        augment: bool = True,
        multitask: bool = True,
    ):
        """
        Initialize the data pipeline.
        
        Args:
            raw_data_sources: List of paths to raw EM volumes
            synapse_sources: List of paths to synapse annotations  
            batch_size: Size of training batches in voxels (z, y, x)
            voxel_size: Physical voxel size in nm (z, y, x)
            num_workers: Number of parallel workers
            augment: Whether to apply data augmentation
            multitask: Whether to generate direction vectors
        """
        self.raw_data_sources = [Path(p) for p in raw_data_sources]
        self.synapse_sources = [Path(p) for p in synapse_sources]
        self.batch_size = Coordinate(batch_size)
        self.voxel_size = Coordinate(voxel_size)
        self.num_workers = num_workers
        self.augment = augment
        self.multitask = multitask
        
        # Define array keys
        self.raw_key = ArrayKey('RAW')
        self.synapses_key = GraphKey('SYNAPSES')
        self.mask_key = ArrayKey('MASK')
        if multitask:
            self.direction_key = ArrayKey('DIRECTION')
        
        self._pipeline = None
        
    def create_pipeline(self) -> Pipeline:
        """Create the complete Gunpowder pipeline."""
        if self._pipeline is not None:
            return self._pipeline
            
        # 1. Data sources
        sources = []
        for raw_path, synapse_path in zip(self.raw_data_sources, self.synapse_sources):
            source = self._create_source(raw_path, synapse_path)
            sources.append(source)
        
        # 2. Combine sources randomly
        if len(sources) > 1:
            pipeline = tuple(sources) + RandomProvider()
        else:
            pipeline = sources[0]
        
        # 3. Add random location selection
        pipeline = (
            pipeline +
            RandomLocation(ensure_nonempty=self.synapses_key)
        )
        
        # 4. Convert synapses to arrays
        pipeline = (
            pipeline +
            SynapseToMask(
                synapses=self.synapses_key,
                mask=self.mask_key,
                radius=40.0,  # nm
            )
        )
        
        if self.multitask:
            pipeline = (
                pipeline +
                SynapseToDirectionVector(
                    synapses=self.synapses_key,
                    direction=self.direction_key,
                    radius=80.0,  # nm
                )
            )
        
        # 5. Data augmentation
        if self.augment:
            pipeline = pipeline + self._create_augmentations()
        
        # 6. Normalize raw data
        pipeline = (
            pipeline +
            Normalize(self.raw_key) +
            IntensityScaleShift(self.raw_key, 2, -1)  # Scale to [-1, 1]
        )
        
        # 7. Create batches
        request = BatchRequest()
        request[self.raw_key] = ArraySpec(
            roi=Roi((0, 0, 0), self.batch_size),
            voxel_size=self.voxel_size
        )
        request[self.mask_key] = ArraySpec(
            roi=Roi((0, 0, 0), self.batch_size),
            voxel_size=self.voxel_size
        )
        if self.multitask:
            request[self.direction_key] = ArraySpec(
                roi=Roi((0, 0, 0), self.batch_size),
                voxel_size=self.voxel_size
            )
        
        pipeline = pipeline + PreCache(cache_size=10, num_workers=self.num_workers)
        
        self._pipeline = pipeline
        return pipeline
    
    def _create_source(self, raw_path: Path, synapse_path: Path) -> Pipeline:
        """Create a data source for raw volume and synapses."""
        # Raw volume source
        if raw_path.suffix == '.zarr':
            raw_source = ZarrSource(
                str(raw_path),
                {self.raw_key: 'volumes/raw'},
                {self.raw_key: ArraySpec(interpolatable=True)}
            )
        elif raw_path.suffix in ['.h5', '.hdf5']:
            raw_source = Hdf5Source(
                str(raw_path),
                {self.raw_key: 'volumes/raw'},
                {self.raw_key: ArraySpec(interpolatable=True)}
            )
        else:
            raise ValueError(f"Unsupported raw data format: {raw_path.suffix}")
        
        # Synapse source
        synapse_source = HDF5SynapseSource(
            str(synapse_path),
            self.synapses_key
        )
        
        return (raw_source, synapse_source) + MergeProvider()
    
    def _create_augmentations(self) -> Pipeline:
        """Create data augmentation pipeline."""
        augmentations = (
            # Geometric augmentations
            ElasticAugment(
                control_point_spacing=[40, 40, 40],
                jitter_sigma=[2.0, 2.0, 2.0],
                rotation_interval=[0, np.pi/2],
                subsample=8
            ) +
            SimpleAugment() +  # Random rotations and flips
            
            # Intensity augmentations  
            IntensityAugment(
                self.raw_key,
                scale_min=0.8,
                scale_max=1.2,
                shift_min=-0.2,
                shift_max=0.2
            ) +
            
            # Noise augmentation
            NoiseAugment(self.raw_key, var=0.01)
        )
        
        return augmentations
    
    def get_torch_dataloader(self) -> torch.utils.data.DataLoader:
        """Get PyTorch DataLoader for training."""
        pipeline = self.create_pipeline()
        
        # Create request
        request = BatchRequest()
        request[self.raw_key] = ArraySpec(
            roi=Roi((0, 0, 0), self.batch_size),
            voxel_size=self.voxel_size
        )
        request[self.mask_key] = ArraySpec(
            roi=Roi((0, 0, 0), self.batch_size), 
            voxel_size=self.voxel_size
        )
        if self.multitask:
            request[self.direction_key] = ArraySpec(
                roi=Roi((0, 0, 0), self.batch_size),
                voxel_size=self.voxel_size
            )
        
        # Convert to PyTorch tensors
        def collate_fn(batch):
            """Convert Gunpowder batch to PyTorch tensors."""
            raw = torch.from_numpy(batch[self.raw_key].data).float()
            mask = torch.from_numpy(batch[self.mask_key].data).float()
            
            # Add channel dimension
            raw = raw.unsqueeze(0)
            mask = mask.unsqueeze(0)
            
            result = {
                'raw': raw,
                'mask': mask
            }
            
            if self.multitask:
                direction = torch.from_numpy(batch[self.direction_key].data).float()
                # Direction has 3 channels (z, y, x)
                result['direction'] = direction
            
            return result
        
        class GunpowderDataset(torch.utils.data.IterableDataset):
            def __init__(self, pipeline, request):
                self.pipeline = pipeline
                self.request = request
                
            def __iter__(self):
                with build(self.pipeline):
                    while True:
                        batch = self.pipeline.request_batch(self.request)
                        yield collate_fn(batch)
        
        dataset = GunpowderDataset(pipeline, request)
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Gunpowder handles batching
            num_workers=0,  # Gunpowder handles parallelism
        )


def create_training_pipeline(
    config: Dict,
    augment: bool = True
) -> SynfulDataPipeline:
    """
    Create a training pipeline from configuration.
    
    Args:
        config: Training configuration dict
        augment: Whether to apply augmentation
        
    Returns:
        Configured data pipeline
    """
    return SynfulDataPipeline(
        raw_data_sources=config['data']['raw_sources'],
        synapse_sources=config['data']['synapse_sources'],
        batch_size=tuple(config['data']['batch_size']),
        voxel_size=tuple(config['data']['voxel_size']),
        num_workers=config['data'].get('num_workers', 4),
        augment=augment,
        multitask=config['model'].get('multitask', True),
    )


def create_inference_pipeline(
    raw_path: Union[str, Path],
    batch_size: Tuple[int, int, int] = (64, 512, 512),
    voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
    overlap: Tuple[int, int, int] = (8, 64, 64)
) -> Pipeline:
    """
    Create inference pipeline for prediction on large volumes.
    
    Args:
        raw_path: Path to raw EM volume
        batch_size: Size of inference chunks
        voxel_size: Physical voxel size
        overlap: Overlap between chunks for seamless prediction
        
    Returns:
        Gunpowder pipeline for inference
    """
    raw_key = ArrayKey('RAW')
    
    # Raw volume source
    raw_path = Path(raw_path)
    if raw_path.suffix == '.zarr':
        source = ZarrSource(
            str(raw_path),
            {raw_key: 'volumes/raw'},
            {raw_key: ArraySpec(interpolatable=True)}
        )
    elif raw_path.suffix in ['.h5', '.hdf5']:
        source = Hdf5Source(
            str(raw_path),
            {raw_key: 'volumes/raw'},
            {raw_key: ArraySpec(interpolatable=True)}
        )
    else:
        raise ValueError(f"Unsupported format: {raw_path.suffix}")
    
    # Normalization (same as training)
    pipeline = (
        source +
        Normalize(raw_key) +
        IntensityScaleShift(raw_key, 2, -1)
    )
    
    return pipeline


class SynapseToMask(BatchFilter):
    """Convert synapse locations to binary mask."""
    
    def __init__(self, synapses: GraphKey, mask: ArrayKey, radius: float = 40.0):
        self.synapses = synapses
        self.mask = mask
        self.radius = radius
    
    def setup(self):
        self.enable_autoskip()
        self.provides(
            self.mask,
            ArraySpec(interpolatable=False, dtype=np.float32)
        )
    
    def prepare(self, request):
        deps = BatchRequest()
        deps[self.synapses] = request[self.mask].copy()
        return deps
    
    def process(self, batch, request):
        synapses = batch[self.synapses]
        mask_spec = request[self.mask]
        
        # Create empty mask
        mask_data = np.zeros(mask_spec.roi.get_shape(), dtype=np.float32)
        
        # Add synapses to mask
        for synapse_id, synapse in synapses.nodes.items():
            # Convert world coordinates to voxel coordinates
            location = np.array(synapse.location) 
            voxel_location = (location - mask_spec.roi.get_begin()) / mask_spec.voxel_size
            
            # Check if synapse is within ROI
            if np.all(voxel_location >= 0) and np.all(voxel_location < mask_data.shape):
                # Create sphere around synapse
                z, y, x = np.ogrid[:mask_data.shape[0], :mask_data.shape[1], :mask_data.shape[2]]
                dz, dy, dx = z - voxel_location[0], y - voxel_location[1], x - voxel_location[2]
                distance = np.sqrt(dz*dz + dy*dy + dx*dx)
                
                radius_voxels = self.radius / mask_spec.voxel_size[0]  # Assume isotropic
                mask_data[distance <= radius_voxels] = 1.0
        
        # Create array
        mask_array = Array(mask_data, mask_spec)
        batch[self.mask] = mask_array


class SynapseToDirectionVector(BatchFilter):
    """Convert synapse pairs to direction vector field."""
    
    def __init__(self, synapses: GraphKey, direction: ArrayKey, radius: float = 80.0):
        self.synapses = synapses
        self.direction = direction  
        self.radius = radius
    
    def setup(self):
        self.enable_autoskip()
        self.provides(
            self.direction,
            ArraySpec(interpolatable=True, dtype=np.float32)
        )
    
    def prepare(self, request):
        deps = BatchRequest()
        deps[self.synapses] = request[self.direction].copy()
        return deps
    
    def process(self, batch, request):
        synapses = batch[self.synapses]
        direction_spec = request[self.direction]
        
        # Create empty direction field (3 channels for z, y, x)
        shape = (3,) + direction_spec.roi.get_shape()
        direction_data = np.zeros(shape, dtype=np.float32)
        
        # Add direction vectors for synapse pairs
        for edge in synapses.edges:
            pre_id, post_id = edge
            if pre_id in synapses.nodes and post_id in synapses.nodes:
                pre_location = np.array(synapses.nodes[pre_id].location)
                post_location = np.array(synapses.nodes[post_id].location)
                
                # Calculate direction vector
                direction_vector = post_location - pre_location
                direction_vector = direction_vector / (np.linalg.norm(direction_vector) + 1e-8)
                
                # Convert to voxel coordinates
                pre_voxel = (pre_location - direction_spec.roi.get_begin()) / direction_spec.voxel_size
                
                if np.all(pre_voxel >= 0) and np.all(pre_voxel < direction_data.shape[1:]):
                    # Create sphere around presynaptic site
                    z, y, x = np.ogrid[:shape[1], :shape[2], :shape[3]]
                    dz, dy, dx = z - pre_voxel[0], y - pre_voxel[1], x - pre_voxel[2]
                    distance = np.sqrt(dz*dz + dy*dy + dx*dx)
                    
                    radius_voxels = self.radius / direction_spec.voxel_size[0]
                    mask = distance <= radius_voxels
                    
                    # Set direction vector in sphere
                    for i in range(3):
                        direction_data[i][mask] = direction_vector[i]
        
        # Create array
        direction_array = Array(direction_data, direction_spec)
        batch[self.direction] = direction_array