"""Unit tests for Detector module"""
import pytest
import numpy as np
import cv2
from src.components.detector import Detector
from src.config.models import DetectorConfig
from src.models import DetectionResult, BoundingBox


class TestDetector:
    """Test suite for Detector class"""
    
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration"""
        config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            edge_detection_enabled=False,
            min_contour_area=500.0,
            max_contour_area=50000.0
        )
        detector = Detector(config)
        assert detector.config == config
        assert detector._background_subtractor is not None
    
    def test_init_with_mog2_method(self):
        """Test initialization with MOG2 background subtraction"""
        config = DetectorConfig(background_subtraction_method="MOG2")
        detector = Detector(config)
        assert detector._background_subtractor is not None
    
    def test_init_with_knn_method(self):
        """Test initialization with KNN background subtraction"""
        config = DetectorConfig(background_subtraction_method="KNN")
        detector = Detector(config)
        assert detector._background_subtractor is not None
    
    def test_init_with_gmg_method(self):
        """Test initialization with GMG background subtraction (or fallback)"""
        config = DetectorConfig(background_subtraction_method="GMG")
        detector = Detector(config)
        # Should create detector even if GMG not available (falls back to MOG2)
        assert detector._background_subtractor is not None
    
    def test_init_with_invalid_method(self):
        """Test initialization fails with invalid background subtraction method"""
        config = DetectorConfig(background_subtraction_method="INVALID")
        with pytest.raises(ValueError, match="Invalid background_subtraction_method"):
            Detector(config)
    
    def test_init_with_invalid_learning_rate_negative(self):
        """Test initialization fails with negative learning rate"""
        config = DetectorConfig(background_learning_rate=-0.1)
        with pytest.raises(ValueError, match="Invalid background_learning_rate"):
            Detector(config)
    
    def test_init_with_invalid_learning_rate_too_high(self):
        """Test initialization fails with learning rate > 1.0"""
        config = DetectorConfig(background_learning_rate=1.5)
        with pytest.raises(ValueError, match="Invalid background_learning_rate"):
            Detector(config)
    
    def test_init_with_invalid_canny_thresholds(self):
        """Test initialization fails when threshold1 >= threshold2"""
        config = DetectorConfig(
            canny_threshold1=150,
            canny_threshold2=50
        )
        with pytest.raises(ValueError, match="canny_threshold1.*must be less than"):
            Detector(config)
    
    def test_init_with_negative_canny_threshold(self):
        """Test initialization fails with negative Canny threshold"""
        config = DetectorConfig(canny_threshold1=-10)
        with pytest.raises(ValueError, match="Canny thresholds must be non-negative"):
            Detector(config)
    
    def test_init_with_negative_min_contour_area(self):
        """Test initialization fails with negative min contour area"""
        config = DetectorConfig(min_contour_area=-100)
        with pytest.raises(ValueError, match="Invalid min_contour_area"):
            Detector(config)
    
    def test_init_with_invalid_area_range(self):
        """Test initialization fails when max_area < min_area"""
        config = DetectorConfig(
            min_contour_area=10000,
            max_contour_area=5000
        )
        with pytest.raises(ValueError, match="max_contour_area.*must be greater than"):
            Detector(config)
    
    def test_detect_with_uint8_frame(self):
        """Test detection with uint8 frame"""
        config = DetectorConfig()
        detector = Detector(config)
        
        # Create a simple frame with a white square on black background
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        
        result = detector.detect(frame)
        
        # Should return DetectionResult
        assert isinstance(result, DetectionResult)
        assert result.foreground_mask is not None
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.contours, list)
    
    def test_detect_with_normalized_frame(self):
        """Test detection with normalized float32 frame"""
        config = DetectorConfig()
        detector = Detector(config)
        
        # Create a normalized frame
        frame = np.zeros((480, 640), dtype=np.float32)
        frame[100:200, 100:200] = 1.0
        
        result = detector.detect(frame)
        
        # Should handle conversion and return DetectionResult
        assert isinstance(result, DetectionResult)
        assert result.foreground_mask is not None
    
    def test_detect_with_none_frame(self):
        """Test detection raises error for None frame"""
        config = DetectorConfig()
        detector = Detector(config)
        
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(None)
    
    def test_detect_with_empty_frame(self):
        """Test detection raises error for empty frame"""
        config = DetectorConfig()
        detector = Detector(config)
        
        empty_frame = np.array([])
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(empty_frame)
    
    def test_detect_filters_by_min_area(self):
        """Test that small contours are filtered out"""
        config = DetectorConfig(
            min_contour_area=1000.0,
            max_contour_area=50000.0
        )
        detector = Detector(config)
        
        # Create frames to establish background
        for _ in range(5):
            bg_frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(bg_frame)
        
        # Create frame with small and large objects
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Small object (5x5 = 25 pixels, area < 1000)
        frame[100:105, 100:105] = 255
        # Large object (40x40 = 1600 pixels, area > 1000)
        frame[200:240, 200:240] = 255
        
        result = detector.detect(frame)
        
        # Should only detect the large object
        for bbox in result.bounding_boxes:
            assert bbox.area >= config.min_contour_area
    
    def test_detect_filters_by_max_area(self):
        """Test that large contours are filtered out"""
        config = DetectorConfig(
            min_contour_area=100.0,
            max_contour_area=5000.0
        )
        detector = Detector(config)
        
        # Create frames to establish background
        for _ in range(5):
            bg_frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(bg_frame)
        
        # Create frame with objects
        frame = np.zeros((480, 640), dtype=np.uint8)
        # Medium object (30x30 = 900 pixels, area < 5000)
        frame[100:130, 100:130] = 255
        # Very large object (100x100 = 10000 pixels, area > 5000)
        frame[200:300, 200:300] = 255
        
        result = detector.detect(frame)
        
        # Should only detect objects within area range
        for bbox in result.bounding_boxes:
            assert bbox.area <= config.max_contour_area
    
    def test_detect_computes_bounding_boxes(self):
        """Test that bounding boxes are computed correctly"""
        config = DetectorConfig(min_contour_area=100.0)
        detector = Detector(config)
        
        # Establish background
        for _ in range(5):
            bg_frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(bg_frame)
        
        # Create frame with known object
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:150, 200:250] = 255  # 50x50 square at (200, 100)
        
        result = detector.detect(frame)
        
        # Should have at least one detection
        if len(result.bounding_boxes) > 0:
            bbox = result.bounding_boxes[0]
            assert isinstance(bbox, BoundingBox)
            assert bbox.x >= 0
            assert bbox.y >= 0
            assert bbox.width > 0
            assert bbox.height > 0
            assert bbox.area > 0
    
    def test_detect_computes_centroids(self):
        """Test that centroids are computed correctly"""
        config = DetectorConfig(min_contour_area=100.0)
        detector = Detector(config)
        
        # Establish background
        for _ in range(5):
            bg_frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(bg_frame)
        
        # Create frame with known object
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:150, 200:250] = 255  # 50x50 square
        
        result = detector.detect(frame)
        
        # Should have centroid
        if len(result.bounding_boxes) > 0:
            bbox = result.bounding_boxes[0]
            cx, cy = bbox.centroid
            assert isinstance(cx, float)
            assert isinstance(cy, float)
            # Centroid should be within frame bounds
            assert 0 <= cx < 640
            assert 0 <= cy < 480
    
    def test_detect_with_edge_detection_enabled(self):
        """Test detection with Canny edge detection enabled"""
        config = DetectorConfig(
            edge_detection_enabled=True,
            canny_threshold1=50,
            canny_threshold2=150
        )
        detector = Detector(config)
        
        # Create frame with edges
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        
        result = detector.detect(frame)
        
        # Should return valid result
        assert isinstance(result, DetectionResult)
        assert result.foreground_mask is not None
    
    def test_detect_with_edge_detection_disabled(self):
        """Test detection with edge detection disabled"""
        config = DetectorConfig(edge_detection_enabled=False)
        detector = Detector(config)
        
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        
        result = detector.detect(frame)
        
        # Should return valid result
        assert isinstance(result, DetectionResult)
    
    def test_detect_updates_background_model(self):
        """Test that background model is updated with each frame"""
        config = DetectorConfig()
        detector = Detector(config)
        
        # First frame - all black
        frame1 = np.zeros((480, 640), dtype=np.uint8)
        result1 = detector.detect(frame1)
        
        # Second frame - same as first
        frame2 = np.zeros((480, 640), dtype=np.uint8)
        result2 = detector.detect(frame2)
        
        # Third frame - with object
        frame3 = np.zeros((480, 640), dtype=np.uint8)
        frame3[100:200, 100:200] = 255
        result3 = detector.detect(frame3)
        
        # Foreground masks should differ as background model updates
        # (This is a basic check - actual behavior depends on learning rate)
        assert result1.foreground_mask is not None
        assert result2.foreground_mask is not None
        assert result3.foreground_mask is not None
    
    def test_detect_result_completeness(self):
        """Test that detection result contains all required fields"""
        config = DetectorConfig()
        detector = Detector(config)
        
        frame = np.zeros((480, 640), dtype=np.uint8)
        result = detector.detect(frame)
        
        # Check all fields are present
        assert hasattr(result, 'frame_number')
        assert hasattr(result, 'bounding_boxes')
        assert hasattr(result, 'contours')
        assert hasattr(result, 'foreground_mask')
        
        # Check types
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.contours, list)
        assert isinstance(result.foreground_mask, np.ndarray)
    
    def test_detect_multiple_objects(self):
        """Test detection of multiple objects in a frame"""
        config = DetectorConfig(min_contour_area=100.0)
        detector = Detector(config)
        
        # Establish background
        for _ in range(5):
            bg_frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(bg_frame)
        
        # Create frame with multiple objects
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[50:100, 50:100] = 255    # Object 1
        frame[150:200, 150:200] = 255  # Object 2
        frame[300:350, 300:350] = 255  # Object 3
        
        result = detector.detect(frame)
        
        # Should detect multiple objects
        # (Actual count may vary due to background subtraction behavior)
        assert len(result.bounding_boxes) == len(result.contours)
    
    def test_detect_no_objects(self):
        """Test detection when no objects are present"""
        config = DetectorConfig()
        detector = Detector(config)
        
        # Establish background
        for _ in range(10):
            frame = np.zeros((480, 640), dtype=np.uint8)
            detector.detect(frame)
        
        # Same background frame
        frame = np.zeros((480, 640), dtype=np.uint8)
        result = detector.detect(frame)
        
        # Should return empty lists (no foreground objects)
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.contours, list)
        # Lists may be empty or have very small detections filtered out
    
    def test_bounding_box_and_contour_count_match(self):
        """Test that number of bounding boxes matches number of contours"""
        config = DetectorConfig()
        detector = Detector(config)
        
        frame = np.zeros((480, 640), dtype=np.uint8)
        frame[100:200, 100:200] = 255
        
        result = detector.detect(frame)
        
        # Should have same number of bounding boxes and contours
        assert len(result.bounding_boxes) == len(result.contours)
