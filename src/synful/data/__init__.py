"""
Modern data pipeline for Synful using PyTorch.

This module provides PyTorch-compatible data loading and augmentation
for 3D electron microscopy data with synaptic annotations.
"""

# Import only the components that work without circular dependencies
try:
    from .augmentations import SynfulAugmentations, ToTensor, Normalize
    _AUGMENTATIONS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Augmentations unavailable: {e}")
    _AUGMENTATIONS_AVAILABLE = False

try:
    from .transforms import (
        SynapseToMask, 
        SynapseToDirectionVector, 
        SynapseToDistanceTransform,
        ComposeTransforms,
        create_training_transforms
    )
    _TRANSFORMS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Transforms unavailable: {e}")
    _TRANSFORMS_AVAILABLE = False

# Note: Pipeline and sources require gunpowder and may have circular imports
# from .pipeline import SynfulDataPipeline, create_training_pipeline, create_inference_pipeline  
# from .sources import HDF5SynapseSource, ZarrVolumeSource, CloudVolumeSource

# Build __all__ based on what's available
__all__ = []

if _AUGMENTATIONS_AVAILABLE:
    __all__.extend(["SynfulAugmentations", "ToTensor", "Normalize"])

if _TRANSFORMS_AVAILABLE:
    __all__.extend([
        "SynapseToMask",
        "SynapseToDirectionVector",
        "SynapseToDistanceTransform",
        "ComposeTransforms",
        "create_training_transforms",
    ])