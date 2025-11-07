"""
Unit tests for SeqSeg utility modules.

This test suite validates core utility functions including:
- Configuration file loading (params.py)
- Dataset helper functions (datasets.py)
- JSON seed file parsing

These tests can run without heavy dependencies like PyTorch or nnUNet.
"""

import os
import json
import tempfile
import pytest
import yaml


class TestParamsModule:
    """Test suite for seqseg.modules.params"""
    
    def test_load_yaml_valid_file(self):
        """Test loading a valid YAML file"""
        from seqseg.modules.params import load_yaml
        
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_content = {
                'SEGMENTATION': True,
                'DEBUG': False,
                'VOLUME_SIZE_RATIO': 4,
                'STEP_SIZE': 2.0
            }
            yaml.dump(yaml_content, f)
            temp_file = f.name
        
        try:
            # Load the YAML file
            config = load_yaml(temp_file)
            
            # Verify the content
            assert config is not None
            assert config['SEGMENTATION'] is True
            assert config['DEBUG'] is False
            assert config['VOLUME_SIZE_RATIO'] == 4
            assert config['STEP_SIZE'] == 2.0
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_load_yaml_missing_file(self):
        """Test loading a non-existent YAML file raises FileNotFoundError"""
        from seqseg.modules.params import load_yaml
        
        # Try to load a non-existent file - should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_yaml('/nonexistent/path/to/file.yaml')
    
    def test_load_yaml_empty_file(self):
        """Test loading an empty YAML file"""
        from seqseg.modules.params import load_yaml
        
        # Create an empty YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            config = load_yaml(temp_file)
            assert config is None or config == {}
        finally:
            os.unlink(temp_file)


class TestDatasetsModule:
    """Test suite for seqseg.modules.datasets"""
    
    def test_get_directories(self):
        """Test get_directories function returns correct paths"""
        from seqseg.modules.datasets import get_directories
        
        directory_data = '/test/data/'
        case = 'test_case'
        img_ext = '.mha'
        
        dir_image, dir_seg, dir_cent, dir_surf = get_directories(
            directory_data, case, img_ext, dir_seg=True
        )
        
        assert dir_image == '/test/data/images/test_case.mha'
        assert dir_seg == '/test/data/truths/test_case.mha'
        assert dir_cent == '/test/data/centerlines/test_case.vtp'
        assert dir_surf == '/test/data/surfaces/test_case.vtp'
    
    def test_get_directories_no_seg(self):
        """Test get_directories when segmentation is disabled"""
        from seqseg.modules.datasets import get_directories
        
        directory_data = '/test/data/'
        case = 'test_case'
        img_ext = '.nii.gz'
        
        dir_image, dir_seg, dir_cent, dir_surf = get_directories(
            directory_data, case, img_ext, dir_seg=False
        )
        
        assert dir_image == '/test/data/images/test_case.nii.gz'
        assert dir_seg is None
        assert dir_cent == '/test/data/centerlines/test_case.vtp'
        assert dir_surf == '/test/data/surfaces/test_case.vtp'
    
    def test_get_testing_samples_json_valid(self):
        """Test parsing a valid seeds JSON file"""
        from seqseg.modules.datasets import get_testing_samples_json
        
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_content = [
                {
                    "name": "case1",
                    "seeds": [
                        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 2.5]
                    ]
                },
                {
                    "name": "case2",
                    "seeds": [
                        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], 3.0],
                        [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], 3.5]
                    ]
                }
            ]
            json.dump(json_content, f)
            temp_file = f.name
        
        try:
            samples = get_testing_samples_json(temp_file)
            
            assert samples is not None
            assert len(samples) == 2
            assert samples[0]['name'] == 'case1'
            assert samples[1]['name'] == 'case2'
            assert len(samples[0]['seeds']) == 1
            assert len(samples[1]['seeds']) == 2
            assert samples[0]['seeds'][0][0] == [1.0, 2.0, 3.0]
            assert samples[0]['seeds'][0][1] == [4.0, 5.0, 6.0]
            assert samples[0]['seeds'][0][2] == 2.5
        finally:
            os.unlink(temp_file)
    
    def test_get_testing_samples_json_empty(self):
        """Test parsing an empty JSON file"""
        from seqseg.modules.datasets import get_testing_samples_json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_file = f.name
        
        try:
            samples = get_testing_samples_json(temp_file)
            assert samples == []
        finally:
            os.unlink(temp_file)
    
    def test_get_testing_samples_with_data_dir(self, tmp_path):
        """Test get_testing_samples with a data directory containing seeds.json"""
        from seqseg.modules.datasets import get_testing_samples
        
        # Create a temporary data directory structure
        data_dir = tmp_path / "test_data"
        data_dir.mkdir()
        
        # Create seeds.json
        seeds_content = [
            {
                "name": "PA000005",
                "seeds": [
                    [[29.7686, -171.3896, -528.9000], [27.8291, -161.0459, -524.9000], 7]
                ]
            }
        ]
        
        seeds_file = data_dir / "seeds.json"
        with open(seeds_file, 'w') as f:
            json.dump(seeds_content, f)
        
        # Test the function
        testing_samples, directory = get_testing_samples('Dataset999_TEST', data_dir=str(data_dir) + '/')
        
        assert testing_samples is not None
        assert len(testing_samples) == 1
        assert testing_samples[0]['name'] == 'PA000005'
        assert directory == str(data_dir) + '/'


class TestConfigFiles:
    """Test suite for verifying configuration files are valid YAML"""
    
    def test_all_config_files_are_valid_yaml(self):
        """Test that all config files in seqseg/config/ are valid YAML"""
        from importlib import resources
        import yaml
        
        # Get all yaml files in config directory
        config_dir = resources.files('seqseg.config')
        
        yaml_files = []
        for item in config_dir.iterdir():
            if item.name.endswith('.yaml'):
                yaml_files.append(item.name)
        
        assert len(yaml_files) > 0, "No YAML config files found"
        
        # Try to load each config file
        for yaml_file in yaml_files:
            with resources.files('seqseg.config').joinpath(yaml_file).open('r') as f:
                try:
                    config = yaml.safe_load(f)
                    assert config is not None or config == {}, f"Config file {yaml_file} loaded as None"
                except yaml.YAMLError as e:
                    pytest.fail(f"Failed to parse {yaml_file}: {e}")
    
    def test_global_test_config_has_required_keys(self):
        """Test that global_test.yaml has expected configuration keys"""
        from importlib import resources
        import yaml
        
        with resources.files('seqseg.config').joinpath('global_test.yaml').open('r') as f:
            config = yaml.safe_load(f)
        
        # Check for some expected keys
        expected_keys = [
            'SEGMENTATION',
            'DEBUG',
            'VOLUME_SIZE_RATIO',
            'MIN_RADIUS',
            'STOP_PRE'
        ]
        
        for key in expected_keys:
            assert key in config, f"Expected key '{key}' not found in global_test.yaml"


class TestSeedsJSON:
    """Test suite for validating test data seeds.json structure"""
    
    def test_test_data_seeds_json_exists(self):
        """Test that test data seeds.json file exists"""
        test_data_dir = os.path.join(
            os.path.dirname(__file__),
            'test_data'
        )
        seeds_file = os.path.join(test_data_dir, 'seeds.json')
        
        assert os.path.exists(seeds_file), "seeds.json not found in test_data directory"
    
    def test_test_data_seeds_json_valid_structure(self):
        """Test that test data seeds.json has valid structure"""
        test_data_dir = os.path.join(
            os.path.dirname(__file__),
            'test_data'
        )
        seeds_file = os.path.join(test_data_dir, 'seeds.json')
        
        with open(seeds_file, 'r') as f:
            seeds = json.load(f)
        
        assert isinstance(seeds, list), "seeds.json should contain a list"
        assert len(seeds) > 0, "seeds.json should not be empty"
        
        # Check structure of first seed entry
        first_entry = seeds[0]
        assert 'name' in first_entry, "Each seed entry should have 'name'"
        assert 'seeds' in first_entry, "Each seed entry should have 'seeds'"
        assert isinstance(first_entry['seeds'], list), "'seeds' should be a list"
        
        if len(first_entry['seeds']) > 0:
            first_seed = first_entry['seeds'][0]
            assert isinstance(first_seed, list), "Each seed should be a list"
            assert len(first_seed) == 3, "Each seed should have [start_point, direction_point, radius]"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
