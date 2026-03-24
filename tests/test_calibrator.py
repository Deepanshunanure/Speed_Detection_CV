"""Unit tests for Calibrator module"""
import pytest
import numpy as np
import cv2
import json
import tempfile
from pathlib import Path
from src.components.calibrator import Calibrator
from src.config.models import CalibratorConfig
from src.models import CalibrationParameters


def create_chessboard_image(chessboard_size=(9, 6), image_size=(640, 480)):
    """Helper function to create synthetic chessboard calibration image"""
    square_size = 50
    # Add border around the chessboard
    border = square_size
    board_width = (chessboard_size[0] + 1) * square_size
    board_height = (chessboard_size[1] + 1) * square_size
    
    # Create white background
    board = np.ones((board_height + 2 * border, board_width + 2 * border), dtype=np.uint8) * 255
    
    # Create chessboard pattern (alternating black and white squares)
    for i in range(chessboard_size[1] + 1):
        for j in range(chessboard_size[0] + 1):
            if (i + j) % 2 == 1:  # Black squares
                y1 = border + i * square_size
                y2 = border + (i + 1) * square_size
                x1 = border + j * square_size
                x2 = border + (j + 1) * square_size
                board[y1:y2, x1:x2] = 0
    
    # Resize to image size
    board = cv2.resize(board, image_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to BGR
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    
    return board_bgr


class TestCalibrator:
    """Test suite for Calibrator class"""
    
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration"""
        config = CalibratorConfig(
            calibration_file_path="calibration.json",
            chessboard_size=(9, 6),
            square_size_mm=25.0,
            perspective_transform_enabled=False
        )
        calibrator = Calibrator(config)
        assert calibrator.config == config
    
    def test_calibrate_with_sufficient_images(self):
        """Test calibration with 10+ valid chessboard images"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            square_size_mm=25.0
        )
        calibrator = Calibrator(config)
        
        # Create 12 synthetic chessboard images
        images = [create_chessboard_image((9, 6)) for _ in range(12)]
        
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        # Verify calibration parameters
        assert params.camera_matrix is not None
        assert params.camera_matrix.shape == (3, 3)
        assert params.distortion_coefficients is not None
        assert len(params.distortion_coefficients) == 5
        assert params.calibration_error >= 0
        assert params.calibration_date != ""
    
    def test_calibrate_with_insufficient_images(self):
        """Test calibration fails with fewer than 10 valid images"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Create only 5 images
        images = [create_chessboard_image((9, 6)) for _ in range(5)]
        
        with pytest.raises(ValueError, match="Insufficient valid calibration images"):
            calibrator.calibrate(images, chessboard_size=(9, 6))
    
    def test_calibrate_with_invalid_chessboard_size(self):
        """Test calibration with images that don't match chessboard size"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Create images with different chessboard size
        images = [create_chessboard_image((7, 5)) for _ in range(12)]
        
        # Should fail because corners won't be detected
        with pytest.raises(ValueError, match="Insufficient valid calibration images"):
            calibrator.calibrate(images, chessboard_size=(9, 6))
    
    def test_calibrate_computes_homography_when_enabled(self):
        """Test that homography matrix is computed when perspective transform enabled"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            perspective_transform_enabled=True
        )
        calibrator = Calibrator(config)
        
        images = [create_chessboard_image((9, 6)) for _ in range(12)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        assert params.homography_matrix is not None
        assert params.homography_matrix.shape == (3, 3)
    
    def test_calibrate_no_homography_when_disabled(self):
        """Test that homography matrix is None when perspective transform disabled"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            perspective_transform_enabled=False
        )
        calibrator = Calibrator(config)
        
        images = [create_chessboard_image((9, 6)) for _ in range(12)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        assert params.homography_matrix is None
    
    def test_calibrate_computes_pixels_per_meter(self):
        """Test that pixels_per_meter is computed"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            square_size_mm=25.0
        )
        calibrator = Calibrator(config)
        
        images = [create_chessboard_image((9, 6)) for _ in range(12)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        assert params.pixels_per_meter is not None
        assert params.pixels_per_meter > 0
    
    def test_save_calibration(self):
        """Test saving calibration parameters to JSON file"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        # Create sample calibration parameters
        params = CalibrationParameters(
            camera_matrix=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32),
            distortion_coefficients=np.array([0.1, -0.2, 0.01, 0.02, 0.05], dtype=np.float32),
            homography_matrix=np.eye(3, dtype=np.float32),
            pixels_per_meter=42.5,
            calibration_error=0.5,
            calibration_date="2024-01-15T10:30:00"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_calibration.json"
            calibrator.save_calibration(params, str(filepath))
            
            # Verify file exists and contains valid JSON
            assert filepath.exists()
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert "camera_matrix" in data
            assert "distortion_coefficients" in data
            assert "homography_matrix" in data
            assert "pixels_per_meter" in data
            assert data["pixels_per_meter"] == 42.5
    
    def test_load_calibration(self):
        """Test loading calibration parameters from JSON file"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        # Create and save calibration parameters
        original_params = CalibrationParameters(
            camera_matrix=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32),
            distortion_coefficients=np.array([0.1, -0.2, 0.01, 0.02, 0.05], dtype=np.float32),
            homography_matrix=np.eye(3, dtype=np.float32),
            pixels_per_meter=42.5,
            calibration_error=0.5,
            calibration_date="2024-01-15T10:30:00"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_calibration.json"
            calibrator.save_calibration(original_params, str(filepath))
            
            # Load and verify
            loaded_params = calibrator.load_calibration(str(filepath))
            
            assert np.allclose(loaded_params.camera_matrix, original_params.camera_matrix)
            assert np.allclose(loaded_params.distortion_coefficients, original_params.distortion_coefficients)
            assert np.allclose(loaded_params.homography_matrix, original_params.homography_matrix)
            assert loaded_params.pixels_per_meter == original_params.pixels_per_meter
            assert loaded_params.calibration_error == original_params.calibration_error
            assert loaded_params.calibration_date == original_params.calibration_date
    
    def test_load_calibration_file_not_found(self):
        """Test loading calibration from non-existent file raises error"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        with pytest.raises(FileNotFoundError):
            calibrator.load_calibration("nonexistent_file.json")
    
    def test_load_calibration_invalid_json(self):
        """Test loading calibration from invalid JSON file raises error"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.json"
            with open(filepath, 'w') as f:
                f.write("invalid json content {")
            
            with pytest.raises(ValueError, match="Invalid calibration file"):
                calibrator.load_calibration(str(filepath))
    
    def test_load_calibration_missing_fields(self):
        """Test loading calibration with missing required fields raises error"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "incomplete.json"
            with open(filepath, 'w') as f:
                json.dump({"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}, f)
            
            with pytest.raises(ValueError, match="Invalid calibration file"):
                calibrator.load_calibration(str(filepath))
    
    def test_calibration_round_trip(self):
        """Test that save and load produce equivalent parameters"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Calibrate from images
        images = [create_chessboard_image((9, 6)) for _ in range(12)]
        original_params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "calibration.json"
            
            # Save and load
            calibrator.save_calibration(original_params, str(filepath))
            loaded_params = calibrator.load_calibration(str(filepath))
            
            # Verify equivalence within numerical precision
            assert np.allclose(loaded_params.camera_matrix, original_params.camera_matrix, rtol=1e-5)
            assert np.allclose(loaded_params.distortion_coefficients, 
                             original_params.distortion_coefficients, rtol=1e-5)
    
    def test_undistort_frame(self):
        """Test undistorting a frame with calibration parameters"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        # Create sample frame and calibration parameters
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        params = CalibrationParameters(
            camera_matrix=np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32),
            distortion_coefficients=np.array([0.1, -0.2, 0.01, 0.02, 0.05], dtype=np.float32)
        )
        
        result = calibrator.undistort(frame, params)
        
        # Result should have same dimensions as input
        assert result.shape == frame.shape
        assert result.dtype == frame.dtype
    
    def test_undistort_preserves_dimensions(self):
        """Test that undistortion preserves frame dimensions"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        # Test with different resolutions
        for height, width in [(480, 640), (720, 1280), (1080, 1920)]:
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            params = CalibrationParameters(
                camera_matrix=np.array([[1000, 0, width//2], [0, 1000, height//2], [0, 0, 1]], dtype=np.float32),
                distortion_coefficients=np.zeros(5, dtype=np.float32)
            )
            
            result = calibrator.undistort(frame, params)
            assert result.shape == (height, width, 3)
    
    def test_apply_perspective_transform(self):
        """Test applying perspective transform with homography matrix"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        params = CalibrationParameters(
            camera_matrix=np.eye(3, dtype=np.float32),
            distortion_coefficients=np.zeros(5, dtype=np.float32),
            homography_matrix=np.eye(3, dtype=np.float32)
        )
        
        result = calibrator.apply_perspective_transform(frame, params)
        
        # Result should have same dimensions
        assert result.shape == frame.shape
    
    def test_apply_perspective_transform_without_homography(self):
        """Test that perspective transform raises error without homography matrix"""
        config = CalibratorConfig()
        calibrator = Calibrator(config)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        params = CalibrationParameters(
            camera_matrix=np.eye(3, dtype=np.float32),
            distortion_coefficients=np.zeros(5, dtype=np.float32),
            homography_matrix=None
        )
        
        with pytest.raises(ValueError, match="Homography matrix not available"):
            calibrator.apply_perspective_transform(frame, params)
