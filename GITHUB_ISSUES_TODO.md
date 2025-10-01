# Synful Modernization TODO - GitHub Issues

Copy each section below as a separate GitHub issue in your repository.

---

## Issue #1: Fix Gunpowder Import Compatibility

**Labels**: `bug`, `high-priority`, `gunpowder`

### 🎯 Task Description
Fix import compatibility issues in modernized Gunpowder nodes for current Gunpowder API

### 📋 Acceptance Criteria
- [ ] Fix `from gunpowder.contrib.points import PreSynPoint, PostSynPoint` 
- [ ] Fix `from gunpowder.profiling import Timing`
- [ ] All Gunpowder nodes import without errors
- [ ] Nodes are functional with modern Gunpowder API

### 📝 Implementation Notes
- Check current Gunpowder API documentation
- Update import paths in `src/synful/gunpowder/` files
- May need to implement alternative point types if contrib.points is deprecated

### 🔗 Files to Update
- `src/synful/gunpowder/hdf5_points_source.py`
- `src/synful/gunpowder/add_partner_vector_map.py`

---

## Issue #2: Port Core Detection Algorithms

**Labels**: `enhancement`, `high-priority`, `core-functionality`

### 🎯 Task Description
Port critical detection algorithms from `synful_original/` to modernized implementation

### 📋 Acceptance Criteria
- [ ] Port `detection.py` with synapse detection algorithms
- [ ] Port `database.py` for MongoDB data management
- [ ] Port `evaluation.py` for model evaluation metrics
- [ ] Port `synapse_mapping.py` for coordinate utilities
- [ ] All ported modules work with Python 3.12+

### 📝 Implementation Notes
- Source files in `synful_original/` directory
- Update for modern Python and dependencies
- Maintain API compatibility where possible

### 🔗 Files to Create
- `src/synful/detection.py`
- `src/synful/database.py` 
- `src/synful/evaluation.py`
- `src/synful/synapse_mapping.py`

---

## Issue #3: Implement Multi-Zarr Data Loading

**Labels**: `enhancement`, `critical`, `data-loading`

### 🎯 Task Description
Implement support for multiple zarr files and multiple cubes per zarr volume

### 📋 Acceptance Criteria
- [ ] Support loading from multiple zarr files simultaneously
- [ ] Handle multiple cubes within each zarr volume
- [ ] Proper coordinate mapping between volumes
- [ ] TSV synapse coordinate transformation for each volume
- [ ] ROI management across multiple volumes

### 📝 Implementation Notes
- Extend `src/synful/data/` module
- Create multi-volume coordinate mapping system
- Update training script to accept multiple data sources

### 🔗 Target API
```python
trainer.train(
    zarr_volumes=["vol1.zarr", "vol2.zarr"],
    cubes_per_volume=[[(0,0,0), (1000,1000,1000)], [(0,0,0), (2000,2000,2000)]],
    synapse_tsvs=["synapses1.tsv", "synapses2.tsv"]
)
```

---

## Issue #4: Integrate Gunpowder Pipeline with PyTorch Lightning

**Labels**: `enhancement`, `critical`, `training`

### 🎯 Task Description
Create bridge between original Gunpowder training pipeline and PyTorch Lightning

### 📋 Acceptance Criteria
- [ ] Implement Gunpowder data pipeline within PyTorch Lightning training
- [ ] Support original augmentation chain (Elastic, Simple, Intensity)
- [ ] Integrate AddPartnerVectorMap for vector field generation
- [ ] Add RasterizePoints and BalanceLabels functionality
- [ ] Maintain PreCache and multi-worker capabilities

### 📝 Implementation Notes
- Create `GunpowderDataModule` for PyTorch Lightning
- Bridge Gunpowder batch system to PyTorch tensors
- Preserve exact augmentation behavior from original

### 🔗 Files to Update
- `src/synful/training.py`
- `src/synful/data/`
- `scripts/train/setup03/train_pytorch.py`

---

## Issue #5: Implement Daisy Blockwise Prediction

**Labels**: `enhancement`, `critical`, `prediction`

### 🎯 Task Description
Implement distributed blockwise prediction using Daisy framework

### 📋 Acceptance Criteria
- [ ] Daisy blockwise processing for large volumes
- [ ] MongoDB coordination for distributed workers
- [ ] Job queue management and worker coordination
- [ ] ROI-based chunking with proper context handling
- [ ] Multiple output dataset management

### 📝 Implementation Notes
- Study original `scripts/predict/predict_blockwise.py`
- Integrate with current `scripts/predict/predict_pytorch.py`
- Maintain compatibility with original Daisy workflow

### 🔗 Files to Update
- `scripts/predict/predict_pytorch.py`
- Create new `src/synful/prediction_blockwise.py`

---

## Issue #6: Comprehensive Test Suite

**Labels**: `testing`, `high-priority`, `quality-assurance`

### 🎯 Task Description
Create comprehensive test suite comparing original vs modernized implementation

### 📋 Acceptance Criteria
- [ ] Numerical accuracy validation tests
- [ ] Augmentation output comparison tests
- [ ] Training convergence comparison tests
- [ ] Prediction accuracy validation tests
- [ ] Performance benchmarking tests
- [ ] Coordinate system compatibility tests

### 📝 Implementation Notes
- Create test data sets for validation
- Compare outputs at each pipeline stage
- Document any acceptable differences

### 🔗 Files to Create
- `tests/test_gunpowder_compatibility.py`
- `tests/test_training_equivalence.py`
- `tests/test_prediction_equivalence.py`
- `tests/test_data_loading.py`

---

## Issue #7: Update Training Pipeline Visualization

**Labels**: `enhancement`, `documentation`, `visualization`

### 🎯 Task Description
Update Jupyter notebook to reflect complete training pipeline including new features

### 📋 Acceptance Criteria
- [ ] Show multi-zarr data loading process
- [ ] Visualize Gunpowder augmentation pipeline
- [ ] Display vector field generation
- [ ] Show training metrics and convergence
- [ ] Include snapshot management features

### 📝 Implementation Notes
- Update existing `synful_training_pipeline_visualization.ipynb`
- Add cells for new multi-zarr functionality
- Include Gunpowder pipeline visualization

### 🔗 Files to Update
- `synful_training_pipeline_visualization.ipynb`

---

## Issue #8: Dependency Modernization

**Labels**: `maintenance`, `dependencies`, `compatibility`

### 🎯 Task Description
Ensure all dependencies work correctly with Python 3.12+ and modern packages

### 📋 Acceptance Criteria
- [ ] Update MongoDB client integration for modern PyMongo
- [ ] CloudVolume optional dependency handling
- [ ] TensorFlow 1.x dependency removal
- [ ] All packages compatible with Python 3.12+
- [ ] Clean dependency specification in requirements.txt

### 📝 Implementation Notes
- Review and update all import statements
- Add proper optional dependency handling
- Test with fresh virtual environment

### 🔗 Files to Update
- `requirements.txt`
- `requirements_modern.txt`
- All import statements throughout codebase

---

## Priority Order for Implementation

1. **Issue #1** (Fix Gunpowder Imports) - Unblocks everything else
2. **Issue #2** (Port Core Modules) - Essential functionality
3. **Issue #3** (Multi-Zarr Loading) - Your specific requirement
4. **Issue #4** (Gunpowder-PyTorch Bridge) - Core training functionality
5. **Issue #5** (Daisy Blockwise Prediction) - Scalable prediction
6. **Issue #6** (Comprehensive Testing) - Quality assurance
7. **Issue #7** (Update Visualization) - Documentation
8. **Issue #8** (Dependency Modernization) - Polish and maintenance