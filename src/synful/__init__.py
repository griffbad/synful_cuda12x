"""
Synful PyTorch: Modern synaptic partner detection in 3D electron microscopy.

This is a complete modernization of the original Synful project, ported to PyTorch
with modern deep learning practices, Python 3.10+ support, and enhanced visualization.
"""

__version__ = "2.0.0"
__author__ = "Julia Buhmann, Jan Funke, and Modernization Team"
__email__ = "buhmannj@janelia.hhmi.org"

# Import core components
from .synapse import Synapse, SynapseCollection
from .models import UNet3D, ConvBlock3D, DoubleConv3D

# Import data processing components
from .data import (
    SynfulAugmentations, ToTensor, Normalize,
    SynapseToMask, SynapseToDirectionVector, SynapseToDistanceTransform,
    ComposeTransforms, create_training_transforms
)

# Import training and inference
from .training import SynfulTrainer, SynfulLightningModule, FocalLoss, create_default_configs
from .inference import SynfulPredictor, load_model_for_inference

# Import Gunpowder nodes for pipeline compatibility
from . import gunpowder

__all__ = [
    # Core data structures
    "Synapse", 
    "SynapseCollection",
    
    # PyTorch models
    "UNet3D", 
    "ConvBlock3D", 
    "DoubleConv3D",
    
    # Data processing
    "SynfulAugmentations",
    "ToTensor", 
    "Normalize",
    "SynapseToMask",
    "SynapseToDirectionVector", 
    "SynapseToDistanceTransform",
    "ComposeTransforms",
    "create_training_transforms",
    
    # Training
    "SynfulTrainer",
    "SynfulLightningModule", 
    "FocalLoss",
    "create_default_configs",
    
    # Inference
    "SynfulPredictor",
    "load_model_for_inference",
]

__all__ = [
    "Synapse",
    "SynapseCollection", 
    "UNet3D",
    "ConvBlock3D", 
    "DoubleConv3D",
    # Will add these once implemented:
    # "SynfulModel",
    # "SynfulTrainer", 
    # "SynfulPredictor",
    # "SynapticPartnerEvaluator",
]