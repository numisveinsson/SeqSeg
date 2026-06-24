# SeqSeg Tests

This directory contains the test suite for SeqSeg.

## Test Files

### test_utils.py
**NEW** - Comprehensive unit tests for core utility modules.

This test suite validates:
- **Configuration loading** (`seqseg.modules.params`)
  - Loading valid YAML files
  - Handling missing files
  - Handling empty files
  
- **Dataset utilities** (`seqseg.modules.datasets`)
  - Directory path generation
  - JSON seed file parsing
  - Test sample loading
  
- **Configuration validation**
  - All config files are valid YAML
  - Required keys are present in configs
  
- **Test data structure**
  - seeds.json format validation
  - Structural integrity checks

**Dependencies:** pytest, PyYAML (no heavy ML dependencies required)

**Run tests:**
```bash
pytest seqseg/tests/test_utils.py -v
```

### test_segmentation.py
Integration test stub for segmentation functionality (work in progress).

### test_vtk.py
VTK functionality tests (work in progress).

### test_nnunet2.py
nnUNet integration tests (work in progress).

## Test Data

The `test_data/` directory contains:
- **seeds.json**: Sample seed points for test cases (PA000005, PA000016)
- **truths/**: Ground truth segmentation data

## Running Tests

### Run all unit tests:
```bash
pytest seqseg/tests/test_utils.py -v
```

### Run specific test class:
```bash
pytest seqseg/tests/test_utils.py::TestParamsModule -v
```

### Run specific test:
```bash
pytest seqseg/tests/test_utils.py::TestParamsModule::test_load_yaml_valid_file -v
```

### Run integration tests (requires full installation):
```bash
# Requires nnUNet, PyTorch, VTK, SimpleITK
bash seqseg/tests/test.sh
```

## CI/CD

Tests are automatically run via GitHub Actions:
- **pytest.yml**: Runs unit tests on push/PR (Ubuntu, Windows, macOS; Python 3.9, 3.11, 3.12)
- **test.yml**: Runs integration tests with the SeqSeg CLI
- **python-app.yml**: Tests installation across platforms

## Adding New Tests

When adding new tests:
1. Follow pytest conventions
2. Use descriptive test names
3. Add docstrings explaining what is being tested
4. Group related tests in classes
5. Mock external dependencies when possible
6. Keep tests independent and isolated
7. Update this README

## Test Coverage

Current test coverage:
- ✅ Configuration loading and validation
- ✅ Dataset utilities and JSON parsing
- ✅ Path generation
- ⏳ VTK functionality (in progress)
- ⏳ nnUNet integration (in progress)
- ⏳ Tracing algorithms (in progress)
- ⏳ Image processing (in progress)
