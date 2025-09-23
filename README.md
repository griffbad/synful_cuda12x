[![DOI](https://zenodo.org/badge/166422086.svg)](https://zenodo.org/badge/latestdoi/166422086)

Synful - CUDA 12.x / RTX 5090 Compatible
======

## 🚀 Upgraded for Latest Hardware
This repository has been **upgraded for compatibility with**:
- **NVIDIA RTX 5090** and newer GPUs
- **CUDA 12.x** toolkit  
- **Python 3.8+** (up to Python 3.12+)
- **TensorFlow 2.12+** (latest stable versions)

## Overview
--------
Synful: A project for the automated detection of synaptic partners in Electron Microscopy brain data using U-Nets (type of Convolutional Neural Network).

This repository provides train and predict scripts for synaptic partner detection. For more details, see our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v1).

We used the method to predict 244 Million synaptic partners in the full adult fly brain (FAFB) dataset.
Please see https://github.com/funkelab/synful_fafb for data dissemination and benchmark datasets.

Please don't hesitate to open
an issue or write us an email ([Julia
Buhmann](mailto:buhmannj@janelia.hhmi.org) or [Jan
Funke](mailto:funkej@janelia.hhmi.org)) if you have any questions!

- [x] Add train scripts
- [x] Add inference scripts  
- [x] Add download links for pretrained models
- [x] **NEW**: Upgrade to TensorFlow 2.x and CUDA 12.x compatibility
- [x] **NEW**: Modern Python 3.8+ support

## 🔄 Major Updates (v2.0)
--------

### TensorFlow 2.x Migration
- **Replaced TensorFlow 1.x** → **TensorFlow 2.x** with eager execution
- **Removed deprecated APIs**: `tf.placeholder`, `tf.Session`, `tf.losses`  
- **New SavedModel format** instead of meta graphs
- **@tf.function decorators** for performance
- **Modern loss calculations** using native TF 2.x APIs

### CUDA 12.x Support  
- **Updated dependencies** for CUDA 12.x compatibility
- **Modern GPU memory management**
- **RTX 5090 optimizations**

### Python Modernization
- **Removed Python 2 compatibility** code
- **Modern setuptools** instead of distutils
- **Updated dependencies** to latest stable versions
- **Type hints ready** (Python 3.8+ features)

Method
------
The pipeline processes 3D raw data in two steps into synaptic partners:
  1) inference of a) `syn_indicator_mask` (postsynaptic locations) and b) `direction_vector` (vector pointing from postsynaptic location to its presynaptic partner)
  2) synapse extraction: a) locations extractions based on `syn_indicator_mask` and b) finding presynaptic partner based on `direction_vector`


![method_figure](docs/_static/method_overview.png)


System Requirements
-------------------

### Hardware Requirements
- **GPU**: NVIDIA RTX 5090, RTX 4090, or newer (24GB+ VRAM recommended)
- **RAM**: 32GB+ system memory recommended  
- **Storage**: Fast SSD recommended for data loading

### Software Requirements  
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11, or macOS
- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **CUDA**: 12.0+ (for GPU acceleration)
- **TensorFlow**: 2.12+ (installed automatically)

Installation Guide
------------------
from source (creating a conda env is recommended).

### Quick Start
```bash
# Create conda environment with Python 3.11
conda create -n synful_cuda12x python=3.11
conda activate synful_cuda12x

# Clone this repository
git clone https://github.com/griffbad/synful_cuda12x.git
cd synful_cuda12x

# Install dependencies
pip install -r requirements.txt
python setup.py install
```

### GPU Setup (CUDA 12.x)
```bash
# Install CUDA-compatible TensorFlow (automatically included in requirements.txt)
# For manual installation:
pip install tensorflow[and-cuda]>=2.12.0

# Verify GPU detection
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Verify Installation
```bash
# Run compatibility test
python test_tf2_compatibility.py
```

#### Install time
Installation should take around 5-10 mins (including dependencies).

## 🔧 Migration Notes
--------

### For Existing Users
If you have existing checkpoints from the original TensorFlow 1.x version:

1. **Checkpoint Migration**: Old `.meta` files need to be converted to SavedModel format
2. **Config Updates**: Update your config files to point to new model paths  
3. **API Changes**: Training and prediction scripts use new TensorFlow 2.x APIs

### Breaking Changes
- **Model Format**: `.meta` files → `SavedModel` directories
- **Training**: `gp.tensorflow.Train` requires gunpowder updates for TF 2.x
- **Prediction**: Updated `gp.tensorflow.Predict` usage

Training
--------

Training scripts are found in

```
scripts/train/<setup>
```

where `<setup>` is the name of a particular network configuration.
In such a <setup> directory, you will find two files:
- `generate_network.py` (generates a TensorFlow SavedModel based on the parameter.json file)
- `train.py` (starts training)


To get started, have a look at the train script in [scripts/train/setup01](scripts/train/setup01).

To start training:
```bash
cd scripts/train/setup01
python generate_network.py parameter.json
python train.py parameter.json
```

### Training Setups
- **setup01**: parameter.json is set to train a network on post-synaptic sites (single-task network)
- **setup02**: parameter.json is set to train on direction vectors (single-task network)  
- **setup03**: parameter.json is set to train on both post-synaptic sites and direction vectors (multi-task network)

### New Model Output
The updated `generate_network.py` creates:
- `<name>_model/` directory (SavedModel format)
- `<name>_config.json` (configuration file)

#### Training runtime
Training takes between 1-5 days on modern GPUs (depending on the size of the network), but you should see reasonable results within a day (after 90k iterations).

Inference
--------

Once you trained a network, you can use this script to run inference:

```bash
cd scripts/predict/
python predict_blockwise.py predict_template.json
```

Adapt following parameters in the configfile `scripts/predict/predict_template.json`:
- `db_host` → Put here the name of your running mongodb server
- `raw_file` → Put here the filepath of your raw data

#### Inference runtime
Processing a CREMI cube (5 microns X 5 microns x 5 microns) takes ~2-3 minutes on RTX 5090.

## 🐛 Troubleshooting
--------

### Common Issues

#### GPU Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size in parameter.json
# Set smaller input_size values
```

#### TensorFlow Import Errors  
```bash
# Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow[and-cuda]>=2.12.0
```

#### CUDA Version Conflicts
```bash
# Check CUDA version
nvcc --version

# Ensure TensorFlow CUDA compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

Pretrained Models / Original Setup
-----------------
We provide pretrained models, that we discuss in detail in our [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v2).

**Note**: Original pretrained models were trained with TensorFlow 1.x and will need conversion to work with this updated version.

For the latest compatible models and conversion tools, please check the releases section.

## 📚 Additional Resources
--------

- [Original Paper](https://www.biorxiv.org/content/10.1101/2019.12.12.874172v2)
- [FAFB Dataset](https://github.com/funkelab/synful_fafb)  
- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [CUDA 12.x Documentation](https://docs.nvidia.com/cuda/)

## 🤝 Contributing
--------

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch  
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License
--------
MIT License - see LICENSE file for details.
