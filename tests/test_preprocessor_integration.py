"""Integration tests for Preprocessor module"""
import pytest
import numpy as np
import cv2
from src.components.preprocessor import Preprocessor
from src.config.models import PreprocessorConfig


class TestPreprocessorIntegration:
    """Integration tests for Preprocessor with realistic scenarios"""
    
    def test_process_realistic_video_frame(self):
        """Test preprocessing a realistic video frame"""
        config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(config)
        
        # Create a realistic frame with some structure
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Add some colored rectangles
        cv2.rectangle(frame, (100, 100), (500, 500), (255, 0, 0), -1)  # Blue
        cv2.rectangle(frame, (600, 200), (1000, 600), (0, 255, 0), -1)  # Green
        cv2.rectangle(frame, (1100, 300), (1500, 700), (0, 0, 255), -1)  # Red
        
        result = preprocessor.process(frame)
        
        # Verify output properties
        assert result.shape == (480, 640)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        # Should have some variation (not all zeros or ones)
        assert np.std(result) > 0.01
    
    def test_process_high_resolution_frame(self):
        """Test preprocessing a high-resolution frame"""
        config = PreprocessorConfig(
            target_resolution=(1280, 720),
            noise_reduction_method="bilateral",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        # Create a 4K frame
        frame = np.random.randint(0, 256, (2160, 3840, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        assert result.shape == (720, 1280)
        assert result.dtype == np.uint8
    
    def test_process_low_resolution_frame(self):
        """Test preprocessing a low-resolution frame (upscaling)"""
        config = PreprocessorConfig(
            target_resolution=(1280, 720),
            noise_reduction_method="median",
            noise_reduction_kernel_size=3,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        # Create a small frame
        frame = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        result = preprocessor.process(frame)
        
        assert result.shape == (720, 1280)
    
    def test_process_noisy_frame(self):
        """Test that noise reduction actually reduces noise"""
        config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=7,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(config)
        
        # Create a frame with salt-and-pepper noise
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        noise_mask = np.random.random((480, 640, 3)) > 0.95
        frame[noise_mask] = 255
        noise_mask = np.random.random((480, 640, 3)) < 0.05
        frame[noise_mask] = 0
        
        result = preprocessor.process(frame)
        
        # Result should be smoother (lower standard deviation)
        assert result is not None
        assert result.shape == (480, 640)
    
    def test_process_multiple_frames_consistency(self):
        """Test that processing multiple frames is consistent"""
        config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(config)
        
        # Process the same frame twice
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        result1 = preprocessor.process(frame)
        result2 = preprocessor.process(frame)
        
        # Results should be identical
        assert np.allclose(result1, result2)
    
    def test_different_noise_methods_produce_different_results(self):
        """Test that different noise reduction methods produce different outputs"""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        config_gaussian = PreprocessorConfig(
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        config_median = PreprocessorConfig(
            noise_reduction_method="median",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        
        preprocessor_gaussian = Preprocessor(config_gaussian)
        preprocessor_median = Preprocessor(config_median)
        
        result_gaussian = preprocessor_gaussian.process(frame)
        result_median = preprocessor_median.process(frame)
        
        # Results should be different
        assert not np.array_equal(result_gaussian, result_median)
    
    def test_performance_target(self):
        """Test that preprocessing meets performance target (<50ms for 1920x1080)"""
        import time
        
        config = PreprocessorConfig(
            target_resolution=(1280, 720),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(config)
        
        # Create a 1920x1080 frame
        frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        # Measure processing time
        start = time.time()
        result = preprocessor.process(frame)
        elapsed_ms = (time.time() - start) * 1000
        
        # Should be under 50ms (requirement 1.5)
        assert elapsed_ms < 50, f"Processing took {elapsed_ms:.2f}ms, expected <50ms"
        assert result is not None
