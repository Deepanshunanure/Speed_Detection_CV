"""Unit tests for Preprocessor module"""
import pytest
import numpy as np
import cv2
from src.components.preprocessor import Preprocessor
from src.config.models import PreprocessorConfig


class TestPreprocessor:
    """Test suite for Preprocessor class"""
    
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration"""
        config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(config)
        assert preprocessor.config == config
    
    def test_init_with_invalid_noise_method(self):
        """Test initialization fails with invalid noise reduction method"""
        config = PreprocessorConfig(noise_reduction_method="invalid")
        with pytest.raises(ValueError, match="Invalid noise_reduction_method"):
            Preprocessor(config)
    
    def test_init_with_even_kernel_size(self):
        """Test initialization fails with even kernel size"""
        config = PreprocessorConfig(noise_reduction_kernel_size=4)
        with pytest.raises(ValueError, match="Must be odd"):
            Preprocessor(config)
    
    def test_init_with_negative_kernel_size(self):
        """Test initialization fails with negative kernel size"""
        config = PreprocessorConfig(noise_reduction_kernel_size=-1)
        with pytest.raises(ValueError, match="Must be positive"):
            Preprocessor(config)
    
    def test_init_with_invalid_resolution(self):
        """Test initialization fails with invalid resolution"""
        config = PreprocessorConfig(target_resolution=(0, 480))
        with pytest.raises(ValueError, match="Invalid target_resolution"):
            Preprocessor(config)
    
    def test_process_converts_to_grayscale(self):
        """Test that process converts BGR frame to grayscale"""
        config = PreprocessorConfig(normalize_intensity=False)
        preprocessor = Preprocessor(config)
        
        # Create a BGR frame
        bgr_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(bgr_frame)
        
        # Result should be 2D (grayscale)
        assert result.ndim == 2
    
    def test_process_already_grayscale(self):
        """Test that process handles already grayscale frames"""
        config = PreprocessorConfig(normalize_intensity=False)
        preprocessor = Preprocessor(config)
        
        # Create a grayscale frame
        gray_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        result = preprocessor.process(gray_frame)
        
        # Should still work
        assert result.ndim == 2
    
    def test_process_resizes_to_target_resolution(self):
        """Test that process resizes frame to target resolution"""
        target_res = (320, 240)
        config = PreprocessorConfig(
            target_resolution=target_res,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        # Create a frame with different resolution
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Result should have target resolution (height, width)
        assert result.shape == (target_res[1], target_res[0])
    
    def test_process_normalizes_intensity_when_enabled(self):
        """Test that process normalizes pixel values to [0, 1] when enabled"""
        config = PreprocessorConfig(normalize_intensity=True)
        preprocessor = Preprocessor(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # All values should be in [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        # Should be float type
        assert result.dtype == np.float32
    
    def test_process_no_normalization_when_disabled(self):
        """Test that process keeps uint8 values when normalization disabled"""
        config = PreprocessorConfig(normalize_intensity=False)
        preprocessor = Preprocessor(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Should be uint8 type
        assert result.dtype == np.uint8
        # Values should be in [0, 255]
        assert np.all(result >= 0)
        assert np.all(result <= 255)
    
    def test_process_with_gaussian_filter(self):
        """Test that Gaussian filter is applied"""
        config = PreprocessorConfig(
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        # Create a noisy frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Should produce valid output
        assert result is not None
        assert result.shape == (720, 1280)  # Default target resolution
    
    def test_process_with_bilateral_filter(self):
        """Test that bilateral filter is applied"""
        config = PreprocessorConfig(
            noise_reduction_method="bilateral",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Should produce valid output
        assert result is not None
        assert result.shape == (720, 1280)
    
    def test_process_with_median_filter(self):
        """Test that median filter is applied"""
        config = PreprocessorConfig(
            noise_reduction_method="median",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Should produce valid output
        assert result is not None
        assert result.shape == (720, 1280)
    
    def test_process_with_none_frame(self):
        """Test that process raises error for None frame"""
        config = PreprocessorConfig()
        preprocessor = Preprocessor(config)
        
        with pytest.raises(ValueError, match="Invalid frame"):
            preprocessor.process(None)
    
    def test_process_with_empty_frame(self):
        """Test that process raises error for empty frame"""
        config = PreprocessorConfig()
        preprocessor = Preprocessor(config)
        
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Invalid frame"):
            preprocessor.process(empty_frame)
    
    def test_process_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(config)
        
        # Create a realistic BGR frame
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        # Verify all transformations
        assert result.ndim == 2  # Grayscale
        assert result.shape == (480, 640)  # Resized
        assert result.dtype == np.float32  # Normalized
        assert np.all(result >= 0.0) and np.all(result <= 1.0)  # In [0, 1]
