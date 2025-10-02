"""
Synful PyTorch: Modern synaptic partner detection in 3D electron microscopy.

This is a complete modernization of the original Synful project, ported to PyTorch
with modern deep learning practices, Python 3.10+ support, and enhanced visualization.
"""

__version__ = "2.0.0"
__author__ = "Julia Buhmann, Jan Funke, and Modernization Team"
__email__ = "buhmannj@janelia.hhmi.org"

# Import core components with optional dependencies
try:
    from .synapse import Synapse, SynapseCollection
    _SYNAPSE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Synapse classes unavailable: {e}")
    _SYNAPSE_AVAILABLE = False

try:
    from .models import UNet3D, ConvBlock3D, DoubleConv3D
    _MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Models unavailable: {e}")
    _MODELS_AVAILABLE = False

# Import data processing components
try:
    from .data import (
        SynfulAugmentations, ToTensor, Normalize,
        SynapseToMask, SynapseToDirectionVector, SynapseToDistanceTransform,
        ComposeTransforms, create_training_transforms
    )
    _DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Data processing unavailable: {e}")
    _DATA_AVAILABLE = False

# Import training and inference
try:
    from .training import SynfulTrainer, SynfulLightningModule, FocalLoss, create_default_configs
    _TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Training components unavailable: {e}")
    _TRAINING_AVAILABLE = False

try:
    from .inference import SynfulPredictor, load_model_for_inference
    _INFERENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Inference components unavailable: {e}")
    _INFERENCE_AVAILABLE = False

# Import Gunpowder nodes for pipeline compatibility
try:
    from . import gunpowder
    _GUNPOWDER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Gunpowder nodes unavailable: {e}")
    _GUNPOWDER_AVAILABLE = False

# Build __all__ dynamically based on what's available
__all__ = []

if _SYNAPSE_AVAILABLE:
    __all__.extend(["Synapse", "SynapseCollection"])

if _MODELS_AVAILABLE:
    __all__.extend(["UNet3D", "ConvBlock3D", "DoubleConv3D"])

if _DATA_AVAILABLE:
    __all__.extend([
        "SynfulAugmentations", "ToTensor", "Normalize",
        "SynapseToMask", "SynapseToDirectionVector", "SynapseToDistanceTransform",
        "ComposeTransforms", "create_training_transforms"
    ])

if _TRAINING_AVAILABLE:
    __all__.extend([
        "SynfulTrainer", "SynfulLightningModule", "FocalLoss", "create_default_configs"
    ])

if _INFERENCE_AVAILABLE:
    __all__.extend(["SynfulPredictor", "load_model_for_inference"])