# TensorFlow 2.x Migration Guide for Synful CUDA 12.x

This guide explains how to migrate from the original TensorFlow 1.x version to the new TensorFlow 2.x compatible version.

## 🚀 What's Changed

### Major Updates
1. **TensorFlow 1.x → 2.x**: Complete migration to modern TensorFlow
2. **Python 2 → 3**: Removed Python 2 compatibility code
3. **CUDA 12.x Support**: Updated for latest GPU hardware
4. **Modern Dependencies**: All packages updated to latest stable versions

### Breaking Changes

#### Model Format
- **Old**: `.meta` graph files + checkpoint files
- **New**: SavedModel format in directories

#### Network Generation
- **Old**: `tf.placeholder`, `tf.Session`, graph-based
- **New**: `@tf.function`, eager execution, SavedModel

#### Training/Prediction APIs
- **Old**: `gp.tensorflow.Train/Predict` with meta files
- **New**: Updated gunpowder nodes (requires gunpowder update)

## 🔧 Migration Steps

### 1. Environment Setup
```bash
# Create new environment
conda create -n synful_cuda12x python=3.11
conda activate synful_cuda12x

# Install updated dependencies
git clone https://github.com/griffbad/synful_cuda12x.git
cd synful_cuda12x
pip install -r requirements.txt
python setup.py install
```

### 2. Convert Existing Models
If you have trained models from the original version:

```python
# This is a conceptual example - actual conversion depends on your specific models
import tensorflow as tf

# Load old checkpoint (if possible)
# Convert to new SavedModel format
# This requires careful handling of variable names and structures
```

### 3. Update Training Scripts
The new training scripts use TensorFlow 2.x patterns:

```python
# Old TF 1.x style
tf.reset_default_graph()
raw = tf.placeholder(tf.float32, shape=input_shape)
# ... build graph
tf.train.export_meta_graph()

# New TF 2.x style  
@tf.function
def model(raw_input):
    # ... model definition
    return outputs

class SynfulModel(tf.Module):
    # ... model class
    
tf.saved_model.save(model, model_path)
```

### 4. Update Configuration Files
Your config files may need updates:

```json
{
  "model_path": "train_net_model",  // Changed from "train_net.meta"
  // ... other configs remain similar
}
```

## 🐛 Troubleshooting

### Common Issues

#### "No module named tensorflow"
```bash
pip install tensorflow[and-cuda]>=2.12.0
```

#### GPU not detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Test TensorFlow GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

#### Memory issues on RTX 5090
```bash
# Add to your training script:
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

#### Gunpowder compatibility issues
The training pipeline uses gunpowder, which may need updates for TF 2.x:

```bash
# Install latest gunpowder
pip install git+https://github.com/funkey/gunpowder.git
```

### Model Conversion
For existing trained models, you'll need to:

1. **Extract weights** from old checkpoints
2. **Rebuild the network** using new TF 2.x generate_network.py
3. **Load weights** into new model structure
4. **Save as SavedModel** format

Example conversion script structure:
```python
# This is a template - adapt to your specific models
import tensorflow as tf

def convert_checkpoint_to_savedmodel(old_checkpoint_path, new_model_path):
    # 1. Load old checkpoint variables
    # 2. Create new model using generate_network.py approach
    # 3. Map old variables to new model
    # 4. Save as SavedModel
    pass
```

## 📊 Performance Notes

### RTX 5090 Optimizations
- **Memory**: Use `tf.config.experimental.set_memory_growth()`
- **Precision**: Consider mixed precision training with `tf.keras.mixed_precision`
- **Batch Size**: Can likely increase batch sizes significantly with 24GB VRAM

### Training Speed Improvements
- **Eager Execution**: Generally faster development but may be slower than graph mode
- **@tf.function**: Use for production training loops
- **XLA**: Enable with `tf.config.optimizer.set_jit(True)` for potential speedups

## 🆘 Getting Help

1. **Check the test script**: `python test_tf2_compatibility.py`
2. **Review logs**: Enable TensorFlow logging for debugging
3. **Open an issue**: Report problems on GitHub
4. **Community**: Ask questions in the discussions section

## 📚 Additional Resources

- [TensorFlow 2.x Migration Guide](https://www.tensorflow.org/guide/migrate)
- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)
- [CUDA 12.x Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [Gunpowder Documentation](https://funkey.science/gunpowder/)