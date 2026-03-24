"""Unit tests for ConfigurationManager"""
import json
import yaml
import pytest
from pathlib import Path
from src.config.manager import ConfigurationManager
from src.config.models import (
    PreprocessorConfig,
    DetectorConfig,
    TrackerConfig,
    CalibratorConfig,
    SpeedEstimatorConfig,
    LoggingConfig,
    PipelineConfig
)


def test_config_manager_with_missing_file(tmp_path):
    """Test ConfigurationManager uses defaults when file is missing"""
    config_path = tmp_path / "nonexistent.yaml"
    manager = ConfigurationManager(str(config_path))
    
    # Should return default configurations
    preprocessor = manager.get_preprocessor_config()
    assert preprocessor.target_resolution == (1280, 720)
    assert preprocessor.noise_reduction_method == "gaussian"
    
    detector = manager.get_detector_config()
    assert detector.background_subtraction_method == "MOG2"
    
    tracker = manager.get_tracker_config()
    assert tracker.max_tracking_distance == 50.0


def test_config_manager_with_yaml_file(tmp_path):
    """Test ConfigurationManager loads YAML configuration"""
    config_path = tmp_path / "config.yaml"
    config_data = {
        'pipeline': {
            'enabled_components': ['preprocessor', 'detector']
        },
        'preprocessor': {
            'target_resolution': [640, 480],
            'noise_reduction_method': 'bilateral',
            'noise_reduction_kernel_size': 7,
            'normalize_intensity': False
        },
        'detector': {
            'background_subtraction_method': 'KNN',
            'background_learning_rate': 0.05,
            'min_contour_area': 1000.0
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    
    # Verify loaded configuration
    preprocessor = manager.get_preprocessor_config()
    assert preprocessor.target_resolution == (640, 480)
    assert preprocessor.noise_reduction_method == 'bilateral'
    assert preprocessor.noise_reduction_kernel_size == 7
    assert preprocessor.normalize_intensity == False
    
    detector = manager.get_detector_config()
    assert detector.background_subtraction_method == 'KNN'
    assert detector.background_learning_rate == 0.05
    assert detector.min_contour_area == 1000.0


def test_config_manager_with_json_file(tmp_path):
    """Test ConfigurationManager loads JSON configuration"""
    config_path = tmp_path / "config.json"
    config_data = {
        'tracker': {
            'max_tracking_distance': 75.0,
            'max_disappeared_frames': 50,
            'trajectory_history_length': 200
        },
        'speed_estimator': {
            'averaging_window_frames': 15,
            'output_unit': 'km/h'
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    
    # Verify loaded configuration
    tracker = manager.get_tracker_config()
    assert tracker.max_tracking_distance == 75.0
    assert tracker.max_disappeared_frames == 50
    assert tracker.trajectory_history_length == 200
    
    speed_estimator = manager.get_speed_estimator_config()
    assert speed_estimator.averaging_window_frames == 15
    assert speed_estimator.output_unit == 'km/h'


def test_config_manager_get_pipeline_config(tmp_path):
    """Test getting complete pipeline configuration"""
    config_path = tmp_path / "config.yaml"
    config_data = {
        'pipeline': {
            'enabled_components': ['preprocessor', 'tracker']
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    pipeline_config = manager.get_pipeline_config()
    
    assert isinstance(pipeline_config, PipelineConfig)
    assert pipeline_config.enabled_components == ['preprocessor', 'tracker']
    assert isinstance(pipeline_config.preprocessor, PreprocessorConfig)
    assert isinstance(pipeline_config.detector, DetectorConfig)
    assert isinstance(pipeline_config.tracker, TrackerConfig)


def test_config_manager_validate_valid_config(tmp_path):
    """Test validation with valid configuration"""
    config_path = tmp_path / "config.yaml"
    config_data = {
        'preprocessor': {
            'target_resolution': [1280, 720],
            'noise_reduction_method': 'gaussian',
            'noise_reduction_kernel_size': 5
        },
        'detector': {
            'background_subtraction_method': 'MOG2',
            'background_learning_rate': 0.01,
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'min_contour_area': 500.0,
            'max_contour_area': 50000.0
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    warnings = manager.validate()
    
    # Should have no warnings for valid config
    assert len(warnings) == 0


def test_config_manager_validate_invalid_config(tmp_path):
    """Test validation with invalid configuration"""
    config_path = tmp_path / "config.yaml"
    config_data = {
        'preprocessor': {
            'noise_reduction_method': 'invalid_method',
            'noise_reduction_kernel_size': 4  # Should be odd
        },
        'detector': {
            'background_learning_rate': 1.5,  # Should be <= 1.0
            'canny_threshold1': 150,
            'canny_threshold2': 50,  # Should be > threshold1
            'min_contour_area': 1000.0,
            'max_contour_area': 500.0  # Should be > min_contour_area
        },
        'tracker': {
            'max_tracking_distance': -10.0  # Should be positive
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    warnings = manager.validate()
    
    # Should have multiple warnings
    assert len(warnings) > 0
    assert any('noise_reduction_method' in w for w in warnings)
    assert any('noise_reduction_kernel_size should be odd' in w for w in warnings)
    assert any('background_learning_rate' in w for w in warnings)
    assert any('canny_threshold1 should be less than canny_threshold2' in w for w in warnings)
    assert any('min_contour_area should be less than max_contour_area' in w for w in warnings)
    assert any('max_tracking_distance should be positive' in w for w in warnings)


def test_config_manager_partial_config(tmp_path):
    """Test that partial configuration uses defaults for missing values"""
    config_path = tmp_path / "config.yaml"
    config_data = {
        'preprocessor': {
            'target_resolution': [800, 600]
            # Other fields should use defaults
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    manager = ConfigurationManager(str(config_path))
    preprocessor = manager.get_preprocessor_config()
    
    # Custom value
    assert preprocessor.target_resolution == (800, 600)
    # Default values
    assert preprocessor.noise_reduction_method == 'gaussian'
    assert preprocessor.noise_reduction_kernel_size == 5
    assert preprocessor.normalize_intensity == True


def test_config_manager_unsupported_format(tmp_path):
    """Test that unsupported file format raises error"""
    config_path = tmp_path / "config.txt"
    config_path.write_text("some text")
    
    with pytest.raises(RuntimeError, match="Unsupported configuration file format"):
        ConfigurationManager(str(config_path))


def test_config_manager_invalid_yaml(tmp_path):
    """Test that invalid YAML raises error"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("invalid: yaml: content: [")
    
    with pytest.raises(RuntimeError, match="Failed to load configuration"):
        ConfigurationManager(str(config_path))
