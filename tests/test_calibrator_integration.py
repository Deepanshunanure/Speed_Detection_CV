"""Integration tests for Calibrator module"""
import pytest
import numpy as np
import cv2
import tempfile
from pathlib import Path
from src.components.calibrator import Calibrator
from src.config.models import CalibratorConfig
from src.models import CalibrationParameters


def create_chessboard_image_with_variation(chessboard_size=(9, 6), image_size=(640, 480), angle=0):
    """Create synthetic chessboard with rotation for more realistic calibration"""
    square_size = 50
    border = square_size
    board_width = (chessboard_size[0] + 1) * square_size
    board_height = (chessboard_size[1] + 1) * square_size
    
    # Create white background
    board = np.ones((board_height + 2 * border, board_width + 2 * border), dtype=np.uint8) * 255
    
    # Create chessboard pattern
    for i in range(chessboard_size[1] + 1):
        for j in range(chessboard_size[0] + 1):
            if (i + j) % 2 == 1:
                y1 = border + i * square_size
                y2 = border + (i + 1) * square_size
                x1 = border + j * square_size
                x2 = border + (j + 1) * square_size
                board[y1:y2, x1:x2] = 0
    
    # Apply rotation if specified
    if angle != 0:
        center = (board.shape[1] // 2, board.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        board = cv2.warpAffine(board, rotation_matrix, (board.shape[1], board.shape[0]), 
                              borderValue=255)
    
    # Resize to image size
    board = cv2.resize(board, image_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to BGR
    board_bgr = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    
    return board_bgr


class TestCalibratorIntegration:
    """Integration tests for Calibrator workflow"""
    
    def test_complete_calibration_workflow(self):
        """Test complete calibration workflow: calibrate, save, load, undistort"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            square_size_mm=25.0,
            perspective_transform_enabled=False
        )
        calibrator = Calibrator(config)
        
        # Step 1: Create calibration images with variations
        images = []
        for angle in range(-15, 16, 3):  # Different angles for better calibration
            images.append(create_chessboard_image_with_variation((9, 6), angle=angle))
        
        # Step 2: Calibrate
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        assert params.camera_matrix is not None
        assert params.distortion_coefficients is not None
        assert params.calibration_error >= 0
        
        # Step 3: Save calibration
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "calibration.json"
            calibrator.save_calibration(params, str(filepath))
            assert filepath.exists()
            
            # Step 4: Load calibration
            loaded_params = calibrator.load_calibration(str(filepath))
            assert np.allclose(loaded_params.camera_matrix, params.camera_matrix)
            
            # Step 5: Use calibration to undistort a frame
            test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            undistorted = calibrator.undistort(test_frame, loaded_params)
            
            assert undistorted.shape == test_frame.shape
            assert undistorted is not None
    
    def test_calibration_with_perspective_transform(self):
        """Test calibration workflow with perspective transform enabled"""
        config = CalibratorConfig(
            chessboard_size=(9, 6),
            square_size_mm=25.0,
            perspective_transform_enabled=True
        )
        calibrator = Calibrator(config)
        
        # Create calibration images
        images = [create_chessboard_image_with_variation((9, 6), angle=a) 
                 for a in range(-10, 11, 2)]
        
        # Calibrate
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        # Verify homography is computed
        assert params.homography_matrix is not None
        assert params.homography_matrix.shape == (3, 3)
        
        # Apply perspective transform
        test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        transformed = calibrator.apply_perspective_transform(test_frame, params)
        
        assert transformed.shape == test_frame.shape
    
    def test_undistort_multiple_frames(self):
        """Test undistorting multiple frames with same calibration"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Calibrate
        images = [create_chessboard_image_with_variation((9, 6), angle=a) 
                 for a in range(-12, 13, 2)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        # Undistort multiple frames
        for i in range(5):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            undistorted = calibrator.undistort(frame, params)
            
            assert undistorted.shape == frame.shape
            assert undistorted.dtype == frame.dtype
    
    def test_calibration_with_different_resolutions(self):
        """Test calibration works with different image resolutions"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Test with different resolutions
        resolutions = [(640, 480), (800, 600), (1280, 720)]
        
        for res in resolutions:
            images = [create_chessboard_image_with_variation((9, 6), image_size=res, angle=a)
                     for a in range(-10, 11, 2)]
            
            params = calibrator.calibrate(images, chessboard_size=(9, 6))
            
            assert params.camera_matrix is not None
            assert params.distortion_coefficients is not None
            
            # Test undistortion with same resolution
            test_frame = np.random.randint(0, 256, (res[1], res[0], 3), dtype=np.uint8)
            undistorted = calibrator.undistort(test_frame, params)
            assert undistorted.shape == test_frame.shape
    
    def test_calibration_persistence_across_sessions(self):
        """Test that calibration can be saved and reused in different sessions"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "persistent_calibration.json"
            
            # Session 1: Calibrate and save
            calibrator1 = Calibrator(config)
            images = [create_chessboard_image_with_variation((9, 6), angle=a)
                     for a in range(-12, 13, 2)]
            params1 = calibrator1.calibrate(images, chessboard_size=(9, 6))
            calibrator1.save_calibration(params1, str(filepath))
            
            # Session 2: Load and use
            calibrator2 = Calibrator(config)
            params2 = calibrator2.load_calibration(str(filepath))
            
            # Verify parameters are equivalent
            assert np.allclose(params1.camera_matrix, params2.camera_matrix)
            assert np.allclose(params1.distortion_coefficients, params2.distortion_coefficients)
            
            # Verify both produce same undistortion
            test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            undistorted1 = calibrator1.undistort(test_frame, params1)
            undistorted2 = calibrator2.undistort(test_frame, params2)
            
            assert np.allclose(undistorted1, undistorted2)
    
    def test_calibration_quality_metrics(self):
        """Test that calibration provides quality metrics"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        images = [create_chessboard_image_with_variation((9, 6), angle=a)
                 for a in range(-10, 11, 2)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        # Verify quality metrics are present
        assert params.calibration_error >= 0
        assert params.calibration_date != ""
        assert params.pixels_per_meter is not None
        assert params.pixels_per_meter > 0
    
    def test_undistort_with_grayscale_frame(self):
        """Test undistortion works with grayscale frames"""
        config = CalibratorConfig(chessboard_size=(9, 6))
        calibrator = Calibrator(config)
        
        # Calibrate
        images = [create_chessboard_image_with_variation((9, 6), angle=a)
                 for a in range(-10, 11, 2)]
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        
        # Test with grayscale frame
        gray_frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        undistorted = calibrator.undistort(gray_frame, params)
        
        assert undistorted.shape == gray_frame.shape
        assert undistorted.ndim == 2
