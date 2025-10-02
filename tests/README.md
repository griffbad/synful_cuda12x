# Synful Test Suite

This directory contains comprehensive tests for the modernized Synful package.

## Test Structure

### Unit Tests (`tests/unit/`)
- `test_import_fix.py` - Tests for import compatibility and graceful degradation
- `test_augmentations_fix.py` - Tests for data augmentation functionality
- `test_training_fix.py` - Tests for training pipeline components

### Integration Tests (`tests/integration/`)
- `test_modernized_scripts.py` - Tests for modernized script functionality
- `test_synful_complete.py` - End-to-end testing of complete pipelines
- `test_zarr_mongodb.py` - Tests for Zarr and MongoDB integration

### Examples (`tests/examples/`)
- `example_snapshot_usage.py` - Example usage of snapshot functionality
- `example_training.py` - Example training pipeline setup

### Performance Tests (`tests/performance/`)
- `quick_test_synful.py` - Quick performance benchmarks
- `simple_test_synful.py` - Simple functionality validation

## Running Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-cov

# For full functionality
pip install torch torchvision pytorch-lightning gunpowder zarr h5py pydantic
```

### Running All Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/synful --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/examples/      # Examples only
pytest tests/performance/   # Performance tests only
```

### Running Individual Tests
```bash
# Run specific test files
python tests/unit/test_import_fix.py
python tests/examples/example_training.py
python tests/performance/quick_test_synful.py
```

## Test Categories

### Import Tests
Verify that the package imports correctly with graceful degradation when dependencies are missing.

### Unit Tests
Test individual components in isolation:
- Data augmentations
- Model architectures  
- Training utilities
- Inference utilities

### Integration Tests
Test complete workflows:
- Training pipeline end-to-end
- Prediction pipeline end-to-end
- Data loading from multiple sources
- Zarr and MongoDB integration

### Performance Tests
Benchmark critical operations:
- Data loading speed
- Model inference speed
- Memory usage
- GPU utilization

## Expected Test Environment

The tests are designed to work in multiple environments:
- ✅ **Minimal**: Basic imports with warnings for missing dependencies
- ✅ **Standard**: Full PyTorch functionality
- ✅ **Complete**: All dependencies including Gunpowder and Daisy

## Contributing Tests

When adding new functionality, please add corresponding tests:
1. **Unit tests** for isolated components
2. **Integration tests** for workflows
3. **Examples** for user documentation
4. **Performance tests** for critical paths

See `IMPLEMENTATION_STATUS.md` for current test coverage and priorities.