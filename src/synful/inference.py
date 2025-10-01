"""
Modern inference module for Synful PyTorch.

Provides efficient inference on large 3D volumes with:
- Chunked processing for memory efficiency
- Overlap-and-blend for seamless predictions
- Multi-scale inference support
- GPU acceleration and batching
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

# Handle different scikit-image versions
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    # In newer scikit-image versions, it's in a different location
    try:
        from skimage.segmentation import peak_local_maxima
    except ImportError:
        # If still not found, use scipy alternative
        from scipy.ndimage import maximum_filter
        def peak_local_maxima(image, min_distance=1, threshold_abs=None, exclude_border=True):
            """Fallback implementation using scipy."""
            from scipy.ndimage import label as scipy_label
            
            # Apply threshold
            if threshold_abs is not None:
                binary = image >= threshold_abs
            else:
                binary = image > 0
            
            # Find local maxima using maximum filter
            local_maxima = maximum_filter(image, size=min_distance*2+1) == image
            local_maxima = local_maxima & binary
            
            # Exclude border if requested
            if exclude_border:
                local_maxima[:min_distance] = False
                local_maxima[-min_distance:] = False
                local_maxima[:, :min_distance] = False
                local_maxima[:, -min_distance:] = False
                if local_maxima.ndim > 2:
                    local_maxima[:, :, :min_distance] = False
                    local_maxima[:, :, -min_distance:] = False
            
            return np.where(local_maxima)

from skimage.measure import label, regionprops

from .models import UNet3D
from .synapse import Synapse, SynapseCollection
from .data import Normalize, ToTensor

logger = logging.getLogger(__name__)


class SynfulPredictor:
    """
    High-performance inference engine for Synful models.
    
    Features:
    - Memory-efficient chunked processing
    - Overlap-and-blend for seamless predictions
    - Multi-scale inference
    - Post-processing and synapse detection
    - Batch processing support
    """
    
    def __init__(
        self,
        model: UNet3D,
        device: Optional[Union[str, torch.device]] = None,
        chunk_size: Tuple[int, int, int] = (64, 512, 512),
        overlap: Tuple[int, int, int] = (8, 64, 64),
        blend_sigma: float = 8.0,
        normalize_input: bool = True,
        half_precision: bool = True,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Trained UNet3D model
            device: Device for inference ("cuda", "cpu", or torch.device)
            chunk_size: Size of processing chunks (z, y, x)
            overlap: Overlap between chunks for blending (z, y, x)
            blend_sigma: Sigma for Gaussian blending weights
            normalize_input: Whether to normalize input data
            half_precision: Whether to use half precision for inference
        """
        self.model = model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = np.array(chunk_size)
        self.overlap = np.array(overlap)
        self.blend_sigma = blend_sigma
        self.normalize_input = normalize_input
        self.half_precision = half_precision
        
        # Move model to device
        self.model = self.model.to(self.device)
        if half_precision and self.device != "cpu":
            self.model = self.model.half()
        
        # Setup normalization
        if normalize_input:
            self.normalizer = Normalize(target_range=(-1, 1))
            self.to_tensor = ToTensor()
        
        # Precompute blending weights
        self._compute_blend_weights()
        
        logger.info(f"Initialized predictor on {self.device}")
        logger.info(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    
    def _compute_blend_weights(self):
        """Precompute Gaussian blending weights for overlapping regions."""
        # Create coordinate grids
        z, y, x = np.ogrid[:self.chunk_size[0], :self.chunk_size[1], :self.chunk_size[2]]
        
        # Distance from edges
        dist_z = np.minimum(z, self.chunk_size[0] - 1 - z)
        dist_y = np.minimum(y, self.chunk_size[1] - 1 - y)
        dist_x = np.minimum(x, self.chunk_size[2] - 1 - x)
        
        # Minimum distance to any edge
        dist_to_edge = np.minimum(np.minimum(dist_z, dist_y), dist_x)
        
        # Gaussian weights based on distance to edge
        weights = np.exp(-0.5 * (dist_to_edge / self.blend_sigma) ** 2)
        weights = weights / weights.max()  # Normalize to [0, 1]
        
        self.blend_weights = torch.from_numpy(weights).float()
        if self.device != "cpu":
            self.blend_weights = self.blend_weights.to(self.device)
        if self.half_precision and self.device != "cpu":
            self.blend_weights = self.blend_weights.half()
    
    @torch.no_grad()
    def predict_volume(
        self,
        volume: np.ndarray,
        voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        return_direction: bool = None,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Predict on a 3D volume using chunked processing.
        
        Args:
            volume: Input volume array [Z, Y, X]
            voxel_size: Physical voxel size (z, y, x) in nm
            return_direction: Whether to return direction vectors (auto-detect if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with predictions:
            - 'mask': Binary mask predictions [Z, Y, X]
            - 'direction': Direction vectors [3, Z, Y, X] (if multitask)
        """
        volume_shape = np.array(volume.shape)
        return_direction = return_direction or self.model.multitask
        
        # Initialize output volumes
        mask_pred = np.zeros(volume_shape, dtype=np.float32)
        weight_sum = np.zeros(volume_shape, dtype=np.float32)
        
        if return_direction:
            direction_pred = np.zeros((3,) + tuple(volume_shape), dtype=np.float32)
            direction_weight_sum = np.zeros((3,) + tuple(volume_shape), dtype=np.float32)
        
        # Calculate chunk positions
        chunk_positions = self._calculate_chunk_positions(volume_shape)
        total_chunks = len(chunk_positions)
        
        logger.info(f"Processing {total_chunks} chunks for volume {volume_shape}")
        
        # Process each chunk
        for i, (start, end) in enumerate(chunk_positions):
            # Extract chunk
            chunk = volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            
            # Pad if necessary
            padded_chunk, pad_info = self._pad_chunk(chunk)
            
            # Normalize and convert to tensor
            if self.normalize_input:
                tensor_data = self.to_tensor({'raw': padded_chunk})
                tensor_data = self.normalizer(tensor_data)
                chunk_tensor = tensor_data['raw']
            else:
                chunk_tensor = torch.from_numpy(padded_chunk).float()
            
            # Add batch and channel dimensions
            chunk_tensor = chunk_tensor.unsqueeze(0).unsqueeze(0)
            chunk_tensor = chunk_tensor.to(self.device)
            
            if self.half_precision and self.device != "cpu":
                chunk_tensor = chunk_tensor.half()
            
            # Forward pass
            outputs = self.model(chunk_tensor)
            
            # Extract predictions
            mask_chunk = torch.sigmoid(outputs['mask_logits']).squeeze().cpu()
            if return_direction:
                direction_chunk = outputs['direction_vectors'].squeeze().cpu()
            
            # Remove padding
            if any(pad_info):
                mask_chunk = self._unpad_chunk(mask_chunk, pad_info)
                if return_direction:
                    direction_chunk = self._unpad_chunk(direction_chunk, pad_info)
            
            # Convert to numpy
            mask_chunk = mask_chunk.float().numpy()
            if return_direction:
                direction_chunk = direction_chunk.float().numpy()
            
            # Get blend weights for this chunk
            chunk_shape = end - start
            weights = self._get_chunk_weights(chunk_shape)
            
            # Accumulate predictions with blending
            mask_pred[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += mask_chunk * weights
            weight_sum[start[0]:end[0], start[1]:end[1], start[2]:end[2]] += weights
            
            if return_direction:
                for c in range(3):
                    direction_pred[c, start[0]:end[0], start[1]:end[1], start[2]:end[2]] += direction_chunk[c] * weights
                    direction_weight_sum[c, start[0]:end[0], start[1]:end[1], start[2]:end[2]] += weights
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_chunks)
        
        # Normalize by weights
        mask_pred /= (weight_sum + 1e-8)
        
        result = {'mask': mask_pred}
        
        if return_direction:
            for c in range(3):
                direction_pred[c] /= (direction_weight_sum[c] + 1e-8)
            result['direction'] = direction_pred
        
        logger.info("Volume prediction completed")
        return result
    
    def _calculate_chunk_positions(self, volume_shape: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Calculate start and end positions for all chunks."""
        positions = []
        
        # Calculate number of chunks in each dimension
        step = self.chunk_size - self.overlap
        
        for z_start in range(0, volume_shape[0], step[0]):
            for y_start in range(0, volume_shape[1], step[1]):
                for x_start in range(0, volume_shape[2], step[2]):
                    start = np.array([z_start, y_start, x_start])
                    end = np.minimum(start + self.chunk_size, volume_shape)
                    
                    # Adjust start if chunk would be too small
                    actual_size = end - start
                    if np.any(actual_size < self.chunk_size // 2):
                        start = np.maximum(0, end - self.chunk_size)
                    
                    positions.append((start, end))
        
        return positions
    
    def _pad_chunk(self, chunk: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Pad chunk to required size if necessary."""
        current_shape = np.array(chunk.shape)
        required_shape = self.chunk_size
        
        if np.all(current_shape >= required_shape):
            return chunk, (0, 0, 0, 0, 0, 0)  # No padding needed
        
        # Calculate padding
        pad_before = (required_shape - current_shape) // 2
        pad_after = required_shape - current_shape - pad_before
        pad_before = np.maximum(0, pad_before)
        pad_after = np.maximum(0, pad_after)
        
        # Pad chunk
        pad_width = list(zip(pad_before, pad_after))
        padded_chunk = np.pad(chunk, pad_width, mode='reflect')
        
        pad_info = tuple(pad_before) + tuple(pad_after)
        return padded_chunk, pad_info
    
    def _unpad_chunk(self, chunk: torch.Tensor, pad_info: Tuple[int, ...]) -> torch.Tensor:
        """Remove padding from chunk."""
        if len(pad_info) == 6:  # 3D case
            pad_z_before, pad_y_before, pad_x_before = pad_info[:3]
            pad_z_after, pad_y_after, pad_x_after = pad_info[3:]
            
            if chunk.dim() == 3:  # [Z, Y, X]
                z_end = chunk.shape[0] - pad_z_after if pad_z_after > 0 else chunk.shape[0]
                y_end = chunk.shape[1] - pad_y_after if pad_y_after > 0 else chunk.shape[1]
                x_end = chunk.shape[2] - pad_x_after if pad_x_after > 0 else chunk.shape[2]
                return chunk[pad_z_before:z_end, pad_y_before:y_end, pad_x_before:x_end]
            
            elif chunk.dim() == 4:  # [C, Z, Y, X]
                z_end = chunk.shape[1] - pad_z_after if pad_z_after > 0 else chunk.shape[1]
                y_end = chunk.shape[2] - pad_y_after if pad_y_after > 0 else chunk.shape[2]
                x_end = chunk.shape[3] - pad_x_after if pad_x_after > 0 else chunk.shape[3]
                return chunk[:, pad_z_before:z_end, pad_y_before:y_end, pad_x_before:x_end]
        
        return chunk
    
    def _get_chunk_weights(self, chunk_shape: np.ndarray) -> np.ndarray:
        """Get blending weights for a chunk of given shape."""
        if np.array_equal(chunk_shape, self.chunk_size):
            return self.blend_weights.cpu().numpy()
        
        # Create weights for smaller chunk
        z, y, x = np.ogrid[:chunk_shape[0], :chunk_shape[1], :chunk_shape[2]]
        
        dist_z = np.minimum(z, chunk_shape[0] - 1 - z)
        dist_y = np.minimum(y, chunk_shape[1] - 1 - y)
        dist_x = np.minimum(x, chunk_shape[2] - 1 - x)
        
        dist_to_edge = np.minimum(np.minimum(dist_z, dist_y), dist_x)
        weights = np.exp(-0.5 * (dist_to_edge / self.blend_sigma) ** 2)
        weights = weights / weights.max()
        
        return weights
    
    def detect_synapses(
        self,
        mask: np.ndarray,
        direction: Optional[np.ndarray] = None,
        voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        threshold: float = 0.5,
        min_distance: float = 80.0,
        max_synapses: Optional[int] = None,
    ) -> SynapseCollection:
        """
        Detect synapses from mask and direction predictions.
        
        Args:
            mask: Binary mask predictions [Z, Y, X]
            direction: Direction vector predictions [3, Z, Y, X] (optional)
            voxel_size: Physical voxel size (z, y, x) in nm
            threshold: Detection threshold for mask
            min_distance: Minimum distance between synapses in nm
            max_synapses: Maximum number of synapses to return
            
        Returns:
            Collection of detected synapses
        """
        # Threshold mask
        binary_mask = mask > threshold
        
        # Find local maxima
        min_distance_voxels = min_distance / np.mean(voxel_size)
        peaks = peak_local_maxima(
            mask,
            min_distance=int(min_distance_voxels),
            threshold_abs=threshold,
            exclude_border=True
        )
        
        synapses = []
        
        for i, (z, y, x) in enumerate(zip(*peaks)):
            # Get detection score
            score = float(mask[z, y, x])
            
            # Convert to world coordinates
            pre_location = (
                z * voxel_size[0],
                y * voxel_size[1], 
                x * voxel_size[2]
            )
            
            # Estimate post-synaptic location using direction vector
            if direction is not None:
                direction_vec = direction[:, z, y, x]
                direction_norm = np.linalg.norm(direction_vec)
                
                if direction_norm > 0.1:  # Valid direction vector
                    # Scale direction vector by typical synapse distance
                    synapse_distance = 100.0  # nm
                    scaled_direction = direction_vec / direction_norm * synapse_distance
                    
                    post_location = (
                        pre_location[0] + scaled_direction[0],
                        pre_location[1] + scaled_direction[1],
                        pre_location[2] + scaled_direction[2]
                    )
                    confidence = min(score + direction_norm * 0.5, 1.0)
                else:
                    # No valid direction, use same location
                    post_location = pre_location
                    confidence = score * 0.8  # Lower confidence
            else:
                post_location = pre_location
                confidence = score
            
            synapse = Synapse(
                id=i,
                location_pre=pre_location,
                location_post=post_location,
                score=score,
                confidence=confidence
            )
            synapses.append(synapse)
        
        # Sort by score and limit number
        synapses.sort(key=lambda s: s.score, reverse=True)
        if max_synapses:
            synapses = synapses[:max_synapses]
        
        logger.info(f"Detected {len(synapses)} synapses")
        return SynapseCollection(synapses)
    
    def predict_and_detect(
        self,
        volume: np.ndarray,
        voxel_size: Tuple[float, float, float] = (8.0, 8.0, 8.0),
        detection_threshold: float = 0.5,
        min_distance: float = 80.0,
        max_synapses: Optional[int] = None,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[Dict[str, np.ndarray], SynapseCollection]:
        """
        Complete inference pipeline: prediction + detection.
        
        Args:
            volume: Input volume array [Z, Y, X]
            voxel_size: Physical voxel size (z, y, x) in nm
            detection_threshold: Detection threshold for synapses
            min_distance: Minimum distance between synapses in nm
            max_synapses: Maximum number of synapses to return
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (predictions_dict, synapse_collection)
        """
        # Predict
        predictions = self.predict_volume(
            volume, 
            voxel_size=voxel_size,
            progress_callback=progress_callback
        )
        
        # Detect synapses
        synapses = self.detect_synapses(
            mask=predictions['mask'],
            direction=predictions.get('direction'),
            voxel_size=voxel_size,
            threshold=detection_threshold,
            min_distance=min_distance,
            max_synapses=max_synapses,
        )
        
        return predictions, synapses


def load_model_for_inference(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    map_location: Optional[str] = None,
) -> UNet3D:
    """
    Load a trained model from checkpoint for inference.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        map_location: Map location for loading checkpoint
        
    Returns:
        Loaded UNet3D model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    map_location = map_location or device
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Extract model state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    # Extract model config from checkpoint if available
    if 'hyper_parameters' in checkpoint:
        model_config = checkpoint['hyper_parameters']
        model = UNet3D(**model_config)
    else:
        # Try to infer config from state dict
        logger.warning("No hyperparameters found in checkpoint. Using default config.")
        model = UNet3D()
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model