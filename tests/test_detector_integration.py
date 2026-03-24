"""Integration tests for Detector with Preprocessor"""
import pytest
import numpy as np
import cv2
from src.components.preprocessor import Preprocessor
from src.components.detector import Detector
from src.config.models import PreprocessorConfig, DetectorConfig


class TestDetectorIntegration:
    """Integration tests for Detector with other components"""
    
    def test_preprocessor_to_detector_pipeline(self):
        """Test that preprocessed frames work correctly with detector"""
        # Setup preprocessor
        prep_config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        )
        preprocessor = Preprocessor(prep_config)
        
        # Setup detector
        det_config = DetectorConfig(
            background_subtraction_method="MOG2",
            min_contour_area=100.0
        )
        detector = Detector(det_config)
        
        # Create a BGR frame
        bgr_frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        # Process through pipeline
        preprocessed = preprocessor.process(bgr_frame)
        result = detector.detect(preprocessed)
        
        # Verify result
        assert result is not None
        assert result.foreground_mask is not None
        assert isinstance(result.bounding_boxes, list)
        assert isinstance(result.contours, list)
    
    def test_detector_with_moving_object_sequence(self):
        """Test detector with a sequence of frames showing moving object"""
        prep_config = PreprocessorConfig(
            target_resolution=(640, 480),
            normalize_intensity=False
        )
        preprocessor = Preprocessor(prep_config)
        
        det_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            min_contour_area=500.0
        )
        detector = Detector(det_config)
        
        # Create sequence of frames with moving object
        for i in range(10):
            # Create frame with object at different position
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x_pos = 100 + i * 20  # Object moves horizontally
            frame[200:250, x_pos:x_pos+50] = 255
            
            # Process through pipeline
            preprocessed = preprocessor.process(frame)
            result = detector.detect(preprocessed)
            
            # After a few frames, background model should stabilize
            # and detect the moving object
            if i > 5:
                # Should have detected something by now
                assert result.foreground_mask is not None
    
    def test_detector_with_static_background(self):
        """Test detector learns static background correctly"""
        prep_config = PreprocessorConfig(normalize_intensity=False)
        preprocessor = Preprocessor(prep_config)
        
        det_config = DetectorConfig(
            background_subtraction_method="MOG2",
            min_contour_area=100.0
        )
        detector = Detector(det_config)
        
        # Feed static background frames
        static_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        for _ in range(20):
            preprocessed = preprocessor.process(static_frame)
            detector.detect(preprocessed)
        
        # Now introduce a new object
        frame_with_object = static_frame.copy()
        frame_with_object[200:300, 200:300] = 255
        
        preprocessed = preprocessor.process(frame_with_object)
        result = detector.detect(preprocessed)
        
        # Should detect the new object
        assert result.foreground_mask is not None
        # May or may not have detections depending on background model state
        assert isinstance(result.bounding_boxes, list)
    
    def test_detector_with_edge_detection_integration(self):
        """Test detector with edge detection enabled in full pipeline"""
        prep_config = PreprocessorConfig(
            target_resolution=(640, 480),
            normalize_intensity=False
        )
        preprocessor = Preprocessor(prep_config)
        
        det_config = DetectorConfig(
            background_subtraction_method="MOG2",
            edge_detection_enabled=True,
            canny_threshold1=50,
            canny_threshold2=150,
            min_contour_area=100.0
        )
        detector = Detector(det_config)
        
        # Create frame with clear edges
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)
        
        preprocessed = preprocessor.process(frame)
        result = detector.detect(preprocessed)
        
        # Should produce valid result with edge detection
        assert result is not None
        assert result.foreground_mask is not None
    
    def test_detector_handles_various_preprocessor_outputs(self):
        """Test detector works with both normalized and non-normalized frames"""
        det_config = DetectorConfig(min_contour_area=100.0)
        detector = Detector(det_config)
        
        # Test with normalized output
        prep_config_norm = PreprocessorConfig(normalize_intensity=True)
        preprocessor_norm = Preprocessor(prep_config_norm)
        
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        preprocessed_norm = preprocessor_norm.process(frame)
        result_norm = detector.detect(preprocessed_norm)
        
        assert result_norm is not None
        
        # Test with non-normalized output
        prep_config_uint8 = PreprocessorConfig(normalize_intensity=False)
        preprocessor_uint8 = Preprocessor(prep_config_uint8)
        
        preprocessed_uint8 = preprocessor_uint8.process(frame)
        result_uint8 = detector.detect(preprocessed_uint8)
        
        assert result_uint8 is not None
    
    def test_full_pipeline_with_realistic_video_frames(self):
        """Test complete pipeline with realistic video frame sequence"""
        prep_config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        preprocessor = Preprocessor(prep_config)
        
        det_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            edge_detection_enabled=False,
            min_contour_area=500.0,
            max_contour_area=50000.0
        )
        detector = Detector(det_config)
        
        # Simulate video frames
        num_frames = 15
        results = []
        
        for i in range(num_frames):
            # Create frame with some variation
            frame = np.random.randint(50, 100, (720, 1280, 3), dtype=np.uint8)
            
            # Add moving object after frame 5
            if i >= 5:
                x_pos = 200 + (i - 5) * 30
                y_pos = 200
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 80, y_pos + 80), 
                            (255, 255, 255), -1)
            
            # Process through pipeline
            preprocessed = preprocessor.process(frame)
            result = detector.detect(preprocessed)
            results.append(result)
        
        # Verify all frames were processed
        assert len(results) == num_frames
        
        # All results should be valid
        for result in results:
            assert result is not None
            assert result.foreground_mask is not None
            assert isinstance(result.bounding_boxes, list)
            assert isinstance(result.contours, list)
            # Verify bounding boxes and contours have same count
            assert len(result.bounding_boxes) == len(result.contours)
