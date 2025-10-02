"""
Data transforms for Synful PyTorch.

Provides transforms for converting between different data representations
used in synaptic detection tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Handle different scipy versions
try:
    from scipy.ndimage import distance_transform_edt, gaussian_filter
except ImportError:
    # Fallback for older scipy versions
    from scipy.ndimage.morphology import distance_transform_edt
    from scipy.ndimage.filters import gaussian_filter

# Conditional import to avoid circular dependency
try:
    from ..synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError:
    _SYNAPSE_AVAILABLE = False
    # Create placeholder classes if synapse module not available
    class Synapse:
        pass
    class SynapseCollection:
        pass

logger = logging.getLogger(__name__)


class SynapseToMask:
    """
    Convert synapse locations to binary mask.
    
    Creates spherical masks around synapse locations for training
    semantic segmentation models.
    """
    
    def __init__(
        self,
        radius: float = 40.0,
        soft_boundary: bool = True,
        boundary_width: float = 10.0,
    ):
        """
        Initialize synapse to mask transform.
        
        Args:
            radius: Radius of synapse mask in nm
            soft_boundary: Whether to create soft boundaries with Gaussian falloff
            boundary_width: Width of soft boundary in nm
        """
        self.radius = radius
        self.soft_boundary = soft_boundary
        self.boundary_width = boundary_width
    
    def __call__(
        self,
        synapses: SynapseCollection,
        shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float],
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Convert synapses to binary mask.
        
        Args:
            synapses: Collection of synapses
            shape: Output shape (z, y, x)
            voxel_size: Voxel size in nm (z, y, x)
            offset: Spatial offset in nm (z, y, x)
            
        Returns:
            Binary mask array
        """
        mask = np.zeros(shape, dtype=np.float32)
        
        radius_voxels = np.array([
            self.radius / voxel_size[0],
            self.radius / voxel_size[1], 
            self.radius / voxel_size[2]
        ])
        
        for synapse in synapses:
            # Convert world coordinates to voxel coordinates
            for location in [synapse.location_pre, synapse.location_post]:
                voxel_location = np.array([
                    (location[0] - offset[0]) / voxel_size[0],
                    (location[1] - offset[1]) / voxel_size[1],
                    (location[2] - offset[2]) / voxel_size[2]
                ])
                
                # Check if location is within volume
                if (np.all(voxel_location >= 0) and 
                    np.all(voxel_location < np.array(shape))):
                    
                    # Create coordinate grids
                    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
                    
                    # Calculate distance from synapse location
                    dz = (z - voxel_location[0]) * voxel_size[0] / self.radius
                    dy = (y - voxel_location[1]) * voxel_size[1] / self.radius  
                    dx = (x - voxel_location[2]) * voxel_size[2] / self.radius
                    
                    distance = np.sqrt(dz*dz + dy*dy + dx*dx)
                    
                    if self.soft_boundary:
                        # Create soft boundary with Gaussian falloff
                        boundary_sigma = self.boundary_width / self.radius
                        synapse_mask = np.exp(-(distance - 1.0)**2 / (2 * boundary_sigma**2))
                        synapse_mask[distance > 1.0 + 2*boundary_sigma] = 0.0
                        synapse_mask[distance <= 1.0] = 1.0
                    else:
                        # Hard boundary
                        synapse_mask = (distance <= 1.0).astype(np.float32)
                    
                    # Add to mask (take maximum for overlapping synapses)
                    mask = np.maximum(mask, synapse_mask)
        
        return mask


class SynapseToDirectionVector:
    """
    Convert synapse pairs to direction vector field.
    
    Creates vector fields pointing from presynaptic to postsynaptic locations
    for training direction prediction models.
    """
    
    def __init__(
        self,
        radius: float = 80.0,
        normalize: bool = True,
        falloff: str = 'linear',  # 'linear', 'gaussian', or 'constant'
    ):
        """
        Initialize synapse to direction vector transform.
        
        Args:
            radius: Radius of influence in nm
            normalize: Whether to normalize direction vectors to unit length
            falloff: Type of distance-based falloff ('linear', 'gaussian', 'constant')
        """
        self.radius = radius
        self.normalize = normalize
        self.falloff = falloff
    
    def __call__(
        self,
        synapses: SynapseCollection,
        shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float],
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Convert synapses to direction vector field.
        
        Args:
            synapses: Collection of synapses
            shape: Output shape (z, y, x)
            voxel_size: Voxel size in nm (z, y, x)
            offset: Spatial offset in nm (z, y, x)
            
        Returns:
            Direction vector field [3, z, y, x] where first dim is (z, y, x) components
        """
        direction_field = np.zeros((3,) + shape, dtype=np.float32)
        weight_field = np.zeros(shape, dtype=np.float32)
        
        for synapse in synapses:
            # Calculate direction vector
            pre_loc = np.array(synapse.location_pre)
            post_loc = np.array(synapse.location_post)
            direction_world = post_loc - pre_loc
            
            if self.normalize:
                direction_norm = np.linalg.norm(direction_world)
                if direction_norm > 1e-8:
                    direction_world = direction_world / direction_norm
            
            # Convert presynaptic location to voxel coordinates
            pre_voxel = np.array([
                (pre_loc[0] - offset[0]) / voxel_size[0],
                (pre_loc[1] - offset[1]) / voxel_size[1], 
                (pre_loc[2] - offset[2]) / voxel_size[2]
            ])
            
            # Check if presynaptic location is within volume
            if (np.all(pre_voxel >= 0) and 
                np.all(pre_voxel < np.array(shape))):
                
                # Create coordinate grids
                z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
                
                # Calculate distance from presynaptic location
                dz = (z - pre_voxel[0]) * voxel_size[0]
                dy = (y - pre_voxel[1]) * voxel_size[1]
                dx = (x - pre_voxel[2]) * voxel_size[2]
                
                distance = np.sqrt(dz*dz + dy*dy + dx*dx)
                
                # Create influence mask
                influence_mask = distance <= self.radius
                
                if np.any(influence_mask):
                    # Calculate weights based on falloff type
                    if self.falloff == 'linear':
                        weights = np.maximum(0, 1.0 - distance / self.radius)
                    elif self.falloff == 'gaussian':
                        sigma = self.radius / 3.0  # 3-sigma falloff
                        weights = np.exp(-0.5 * (distance / sigma)**2)
                    else:  # constant
                        weights = np.ones_like(distance)
                    
                    weights[~influence_mask] = 0.0
                    
                    # Apply synapse confidence as additional weight
                    weights = weights * synapse.confidence
                    
                    # Update direction field (weighted average)
                    for i in range(3):
                        direction_field[i] += weights * direction_world[i]
                    
                    weight_field += weights
        
        # Normalize by total weights
        nonzero_weights = weight_field > 1e-8
        for i in range(3):
            direction_field[i][nonzero_weights] /= weight_field[nonzero_weights]
        
        return direction_field


class SynapseToDistanceTransform:
    """
    Convert synapse locations to distance transform.
    
    Creates distance transform maps that encode the distance to the
    nearest synapse location, useful for certain loss functions.
    """
    
    def __init__(
        self,
        max_distance: float = 200.0,
        normalize: bool = True,
    ):
        """
        Initialize distance transform.
        
        Args:
            max_distance: Maximum distance to compute in nm
            normalize: Whether to normalize distances to [0, 1]
        """
        self.max_distance = max_distance
        self.normalize = normalize
    
    def __call__(
        self,
        synapses: SynapseCollection,
        shape: Tuple[int, int, int],
        voxel_size: Tuple[float, float, float],
        offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        """
        Convert synapses to distance transform.
        
        Args:
            synapses: Collection of synapses
            shape: Output shape (z, y, x)
            voxel_size: Voxel size in nm (z, y, x)
            offset: Spatial offset in nm (z, y, x)
            
        Returns:
            Distance transform array
        """
        # Create binary mask of synapse locations
        mask = np.zeros(shape, dtype=bool)
        
        for synapse in synapses:
            for location in [synapse.location_pre, synapse.location_post]:
                voxel_location = np.array([
                    int((location[0] - offset[0]) / voxel_size[0]),
                    int((location[1] - offset[1]) / voxel_size[1]),
                    int((location[2] - offset[2]) / voxel_size[2])
                ])
                
                # Check bounds
                if (np.all(voxel_location >= 0) and 
                    np.all(voxel_location < np.array(shape))):
                    mask[tuple(voxel_location)] = True
        
        # Compute distance transform
        distance_transform = distance_transform_edt(
            ~mask,
            sampling=voxel_size
        )
        
        # Clip to maximum distance
        distance_transform = np.minimum(distance_transform, self.max_distance)
        
        # Normalize if requested
        if self.normalize:
            distance_transform = distance_transform / self.max_distance
        
        return distance_transform.astype(np.float32)


class ComposeTransforms:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List):
        """
        Initialize composed transforms.
        
        Args:
            transforms: List of transform objects
        """
        self.transforms = transforms
    
    def __call__(self, *args, **kwargs):
        """Apply all transforms in sequence."""
        result = args[0] if args else kwargs
        
        for transform in self.transforms:
            if isinstance(result, dict):
                result = transform(**result)
            else:
                result = transform(result, *args[1:], **kwargs)
        
        return result


def create_training_transforms(
    mask_radius: float = 40.0,
    direction_radius: float = 80.0,
    soft_boundary: bool = True,
    multitask: bool = True,
) -> Dict[str, callable]:
    """
    Create standard training transforms.
    
    Args:
        mask_radius: Radius for mask generation in nm
        direction_radius: Radius for direction field generation in nm
        soft_boundary: Whether to use soft boundaries for masks
        multitask: Whether to include direction vector generation
        
    Returns:
        Dictionary of transform functions
    """
    transforms = {
        'mask': SynapseToMask(
            radius=mask_radius,
            soft_boundary=soft_boundary
        )
    }
    
    if multitask:
        transforms['direction'] = SynapseToDirectionVector(
            radius=direction_radius,
            normalize=True,
            falloff='linear'
        )
    
    return transforms