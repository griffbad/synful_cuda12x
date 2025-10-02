"""
Data augmentations for Synful PyTorch.

Provides modern data augmentation techniques optimized for
3D electron microscopy and synaptic detection tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class SynfulAugmentations:
    """
    Comprehensive augmentation suite for 3D EM data.
    
    Includes geometric, intensity, and noise augmentations
    specifically designed for synaptic detection.
    """
    
    def __init__(
        self,
        prob_geometric: float = 0.8,
        prob_intensity: float = 0.7,
        prob_noise: float = 0.5,
        elastic_strength: float = 2.0,
        rotation_range: float = 180.0,
        intensity_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.05,
    ):
        """
        Initialize augmentation suite.
        
        Args:
            prob_geometric: Probability of applying geometric augmentations
            prob_intensity: Probability of applying intensity augmentations  
            prob_noise: Probability of applying noise augmentations
            elastic_strength: Strength of elastic deformation
            rotation_range: Range of random rotations in degrees
            intensity_range: Range for intensity scaling (min, max)
            noise_std: Standard deviation of Gaussian noise
        """
        self.prob_geometric = prob_geometric
        self.prob_intensity = prob_intensity
        self.prob_noise = prob_noise
        self.elastic_strength = elastic_strength
        self.rotation_range = rotation_range
        self.intensity_range = intensity_range
        self.noise_std = noise_std
    
    def __call__(
        self,
        raw: torch.Tensor,
        mask: torch.Tensor,
        direction: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply augmentations to a batch.
        
        Args:
            raw: Raw EM data [B, C, D, H, W]
            mask: Binary mask [B, C, D, H, W]
            direction: Direction vectors [B, 3, D, H, W] (optional)
            
        Returns:
            Dictionary with augmented tensors
        """
        result = {'raw': raw, 'mask': mask}
        if direction is not None:
            result['direction'] = direction
        
        device = raw.device
        
        # Apply geometric augmentations
        if torch.rand(1, device=device) < self.prob_geometric:
            result = self._apply_geometric(result)
        
        # Apply intensity augmentations (only to raw)
        if torch.rand(1, device=device) < self.prob_intensity:
            result['raw'] = self._apply_intensity(result['raw'])
        
        # Apply noise augmentations (only to raw)
        if torch.rand(1, device=device) < self.prob_noise:
            result['raw'] = self._apply_noise(result['raw'])
        
        return result
    
    def _apply_geometric(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply geometric augmentations."""
        device = data['raw'].device
        
        # Random rotation around z-axis
        if torch.rand(1, device=device) < 0.5:
            angle = torch.rand(1, device=device) * self.rotation_range - self.rotation_range / 2
            data = self._rotate_z(data, angle.item())
        
        # Random flips
        if torch.rand(1, device=device) < 0.5:
            data = self._flip_x(data)
        
        if torch.rand(1, device=device) < 0.5:
            data = self._flip_y(data)
        
        # Elastic deformation (simplified)
        if torch.rand(1, device=device) < 0.3:
            data = self._elastic_deform(data)
        
        return data
    
    def _apply_intensity(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply intensity augmentations."""
        device = raw.device
        
        # Random intensity scaling
        scale = torch.rand(1, device=device) * (self.intensity_range[1] - self.intensity_range[0]) + self.intensity_range[0]
        raw = raw * scale
        
        # Random intensity shift
        shift = torch.rand(1, device=device) * 0.2 - 0.1
        raw = raw + shift
        
        # Random gamma correction
        if torch.rand(1, device=device) < 0.3:
            gamma = torch.rand(1, device=device) * 0.4 + 0.8  # 0.8 to 1.2
            raw = torch.sign(raw) * torch.pow(torch.abs(raw), gamma)
        
        return raw
    
    def _apply_noise(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentations."""
        device = raw.device
        
        # Gaussian noise
        noise = torch.randn_like(raw) * self.noise_std
        raw = raw + noise
        
        # Salt and pepper noise (rare)
        if torch.rand(1, device=device) < 0.1:
            mask = torch.rand_like(raw) < 0.01
            raw[mask] = torch.rand_like(raw[mask]) * 2 - 1  # Random values in [-1, 1]
        
        return raw
    
    def _rotate_z(self, data: Dict[str, torch.Tensor], angle_deg: float) -> Dict[str, torch.Tensor]:
        """Rotate around z-axis."""
        angle_rad = np.radians(angle_deg)
        
        # Create 3D rotation matrix for z-axis rotation
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0, 0],
            [sin_a, cos_a, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32, device=data['raw'].device)
        
        # Create affine grid for 3D data
        N, C, D, H, W = data['raw'].shape
        grid = F.affine_grid(
            rotation_matrix.unsqueeze(0).repeat(N, 1, 1),
            (N, C, D, H, W),
            align_corners=False
        )
        
        # Apply rotation to volumes
        result = {}
        for key, tensor in data.items():
            if key == 'direction':
                # For direction vectors, we need to apply the rotation transformation
                # to both the spatial coordinates and the vector values themselves
                rotated_tensor = F.grid_sample(
                    tensor, grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False
                )
                
                # Also rotate the direction vectors themselves (the vector values)
                # Apply rotation matrix to each voxel's direction vector
                N, C, D, H, W = rotated_tensor.shape
                rotated_directions = torch.zeros_like(rotated_tensor)
                for i in range(3):  # x, y, z components
                    for j in range(3):
                        if j < C:  # Make sure we don't go out of bounds
                            rotated_directions[:, i] += rotation_matrix[i, j] * rotated_tensor[:, j]
                
                result[key] = rotated_directions
            else:
                # Standard rotation for raw and mask
                result[key] = F.grid_sample(
                    tensor, grid,
                    mode='bilinear' if key == 'raw' else 'nearest',
                    padding_mode='zeros',
                    align_corners=False
                )
        
        return result
    
    def _flip_x(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flip along x-axis (width)."""
        result = {}
        for key, tensor in data.items():
            result[key] = torch.flip(tensor, dims=[-1])  # Flip width dimension
            
            # Flip x-component of direction vectors
            if key == 'direction':
                result[key][:, 2] = -result[key][:, 2]  # Flip x-component
        
        return result
    
    def _flip_y(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flip along y-axis (height)."""
        result = {}
        for key, tensor in data.items():
            result[key] = torch.flip(tensor, dims=[-2])  # Flip height dimension
            
            # Flip y-component of direction vectors
            if key == 'direction':
                result[key][:, 1] = -result[key][:, 1]  # Flip y-component
        
        return result
    
    def _elastic_deform(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply simplified elastic deformation."""
        # This is a simplified version - in practice you'd want more sophisticated
        # elastic deformation using proper displacement fields
        
        N, C, D, H, W = data['raw'].shape
        
        # Create random displacement field
        displacement_std = self.elastic_strength
        
        # Generate smooth displacement field for all 3 dimensions
        dz = torch.randn(N, 1, D//4, H//4, W//4, device=data['raw'].device) * displacement_std * 0.5  # Smaller for z
        dy = torch.randn(N, 1, D//4, H//4, W//4, device=data['raw'].device) * displacement_std
        dx = torch.randn(N, 1, D//4, H//4, W//4, device=data['raw'].device) * displacement_std
        
        # Upsample displacement fields
        dz = F.interpolate(dz, size=(D, H, W), mode='trilinear', align_corners=False)
        dy = F.interpolate(dy, size=(D, H, W), mode='trilinear', align_corners=False)
        dx = F.interpolate(dx, size=(D, H, W), mode='trilinear', align_corners=False)
        
        # Create 3D grid
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, D, device=data['raw'].device),
            torch.linspace(-1, 1, H, device=data['raw'].device),
            torch.linspace(-1, 1, W, device=data['raw'].device),
            indexing='ij'
        )
        
        # Expand grids to batch size
        grid_z = grid_z.unsqueeze(0).repeat(N, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(N, 1, 1, 1)
        grid_x = grid_x.unsqueeze(0).repeat(N, 1, 1, 1)
        
        # Apply displacement
        grid_z = grid_z + dz.squeeze(1) * 0.05  # Scale down displacement
        grid_y = grid_y + dy.squeeze(1) * 0.1
        grid_x = grid_x + dx.squeeze(1) * 0.1
        
        # Stack grid for 3D sampling [N, D, H, W, 3] where last dim is [x, y, z]
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        # Apply deformation
        result = {}
        for key, tensor in data.items():
            if tensor.dim() == 5:  # [N, C, D, H, W]
                result[key] = F.grid_sample(
                    tensor, grid,
                    mode='bilinear' if key == 'raw' else 'nearest',
                    padding_mode='zeros',
                    align_corners=False
                )
            else:
                result[key] = tensor
        
        return result


class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    
    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert all arrays in data dict to tensors."""
        result = {}
        for key, array in data.items():
            if isinstance(array, np.ndarray):
                result[key] = torch.from_numpy(array.copy()).float()
            else:
                result[key] = array
        return result


class Normalize:
    """Normalize raw data to specified range."""
    
    def __init__(
        self,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        target_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Initialize normalization.
        
        Args:
            mean: Mean for z-score normalization (computed if None)
            std: Std for z-score normalization (computed if None)
            target_range: Target range for min-max normalization
        """
        self.mean = mean
        self.std = std
        self.target_range = target_range
    
    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize raw data."""
        result = data.copy()
        
        if 'raw' in data:
            raw = data['raw']
            
            if self.mean is not None and self.std is not None:
                # Z-score normalization
                raw = (raw - self.mean) / (self.std + 1e-8)
            else:
                # Min-max normalization to target range
                raw_min = raw.min()
                raw_max = raw.max()
                raw_range = raw_max - raw_min + 1e-8
                
                # Scale to [0, 1]
                raw = (raw - raw_min) / raw_range
                
                # Scale to target range
                target_min, target_max = self.target_range
                raw = raw * (target_max - target_min) + target_min
            
            result['raw'] = raw
        
        return result