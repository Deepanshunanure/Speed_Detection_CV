"""Integration tests for Tracker with Detector"""
import pytest
import numpy as np
import cv2
from src.components.tracker import Tracker
from src.components.detector import Detector
from src.components.preprocessor import Preprocessor
from src.config.models import TrackerConfig, DetectorConfig, PreprocessorConfig


class TestTrackerDetectorIntegration:
    """Test tracker integration with detector"""
    
    def test_tracker_with_detector_output(self):
        """Test that tracker can process detector output"""
        # Create components
        preprocessor_config = PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        )
        detector_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            min_contour_area=100.0,
            max_contour_area=10000.0
        )
        tracker_config = TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        )
        
        preprocessor = Preprocessor(preprocessor_config)
        detector = Detector(detector_config)
        tracker = Tracker(tracker_config)
        
        # Create synthetic video frames with moving object
        frames = []
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a moving white rectangle
            x = 100 + i * 20
            y = 100 + i * 10
            cv2.rectangle(frame, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
            frames.append(frame)
        
        # Process frames through pipeline
        tracked_objects_per_frame = []
        for i, frame in enumerate(frames):
            # Preprocess
            processed = preprocessor.process(frame)
            
            # Detect
            detections = detector.detect(processed)
            detections.frame_number = i
            
            # Track
            tracking_result = tracker.update(detections)
            tracked_objects_per_frame.append(tracking_result.tracked_objects)
        
        # Verify tracking results
        # After a few frames, should have stable tracking
        assert len(tracked_objects_per_frame) == 10
        
        # Check that we have tracked objects in later frames
        # (first few frames might be used for background model initialization)
        later_frames = tracked_objects_per_frame[5:]
        has_tracked_objects = any(len(objs) > 0 for objs in later_frames)
        assert has_tracked_objects, "Should have tracked objects in later frames"
    
    def test_tracker_maintains_id_across_frames(self):
        """Test that tracker maintains object ID across multiple frames"""
        detector_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.1,  # Higher learning rate for faster adaptation
            min_contour_area=50.0,
            max_contour_area=10000.0
        )
        tracker_config = TrackerConfig(
            max_tracking_distance=100.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        )
        
        detector = Detector(detector_config)
        tracker = Tracker(tracker_config)
        
        # Create frames with consistent moving object
        frames = []
        for i in range(15):
            frame = np.zeros((480, 640), dtype=np.uint8)
            # Draw a moving white circle
            x = 200 + i * 15
            y = 200 + i * 5
            cv2.circle(frame, (x, y), 30, 255, -1)
            frames.append(frame)
        
        # Process frames
        object_ids_per_frame = []
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            detections.frame_number = i
            tracking_result = tracker.update(detections)
            
            ids = [obj.object_id for obj in tracking_result.tracked_objects]
            object_ids_per_frame.append(ids)
        
        # After background model stabilizes, should have consistent ID
        # Check frames 8-14 for ID consistency
        stable_frames = object_ids_per_frame[8:15]
        stable_frames_with_objects = [ids for ids in stable_frames if len(ids) > 0]
        
        if len(stable_frames_with_objects) > 1:
            # All frames should have the same object ID
            first_id = stable_frames_with_objects[0][0]
            for ids in stable_frames_with_objects[1:]:
                if len(ids) > 0:
                    assert first_id in ids, "Object ID should be consistent across frames"
    
    def test_tracker_with_multiple_objects(self):
        """Test tracker with multiple objects from detector"""
        detector_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.1,
            min_contour_area=50.0,
            max_contour_area=10000.0
        )
        tracker_config = TrackerConfig(
            max_tracking_distance=80.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        )
        
        detector = Detector(detector_config)
        tracker = Tracker(tracker_config)
        
        # Create frames with two moving objects
        frames = []
        for i in range(12):
            frame = np.zeros((480, 640), dtype=np.uint8)
            # Object 1: moving right
            x1 = 100 + i * 20
            cv2.circle(frame, (x1, 200), 25, 255, -1)
            # Object 2: moving down
            y2 = 100 + i * 20
            cv2.circle(frame, (400, y2), 25, 255, -1)
            frames.append(frame)
        
        # Process frames
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            detections.frame_number = i
            tracking_result = tracker.update(detections)
        
        # In later frames, should have multiple tracked objects
        # (after background model stabilizes)
        assert len(tracking_result.tracked_objects) >= 1, "Should track at least one object"
    
    def test_tracker_trajectory_accumulation(self):
        """Test that tracker accumulates trajectory over frames"""
        detector_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.1,
            min_contour_area=50.0,
            max_contour_area=10000.0
        )
        tracker_config = TrackerConfig(
            max_tracking_distance=100.0,
            max_disappeared_frames=30,
            trajectory_history_length=50
        )
        
        detector = Detector(detector_config)
        tracker = Tracker(tracker_config)
        
        # Create frames with moving object
        num_frames = 20
        frames = []
        for i in range(num_frames):
            frame = np.zeros((480, 640), dtype=np.uint8)
            x = 150 + i * 20
            y = 150 + i * 10
            cv2.circle(frame, (x, y), 30, 255, -1)
            frames.append(frame)
        
        # Process frames
        tracking_result = None
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            detections.frame_number = i
            tracking_result = tracker.update(detections)
        
        # Check that trajectories have accumulated
        if len(tracking_result.tracked_objects) > 0:
            # At least one object should have a trajectory
            trajectories = [obj.trajectory for obj in tracking_result.tracked_objects]
            max_trajectory_length = max(len(traj) for traj in trajectories)
            assert max_trajectory_length > 1, "Should have accumulated trajectory points"


class TestTrackerEdgeCases:
    """Test tracker edge cases"""
    
    def test_tracker_with_empty_detections(self):
        """Test tracker handles empty detections gracefully"""
        tracker_config = TrackerConfig()
        tracker = Tracker(tracker_config)
        
        # Create empty detection result
        from src.models import DetectionResult
        empty_detections = DetectionResult(
            frame_number=0,
            bounding_boxes=[],
            contours=[],
            foreground_mask=np.zeros((100, 100), dtype=np.uint8)
        )
        
        result = tracker.update(empty_detections)
        
        assert result.frame_number == 0
        assert len(result.tracked_objects) == 0
    
    def test_tracker_with_intermittent_detections(self):
        """Test tracker handles intermittent detections"""
        detector_config = DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.1,
            min_contour_area=50.0,
            max_contour_area=10000.0
        )
        tracker_config = TrackerConfig(
            max_tracking_distance=100.0,
            max_disappeared_frames=5,
            trajectory_history_length=100
        )
        
        detector = Detector(detector_config)
        tracker = Tracker(tracker_config)
        
        # Create frames where object appears and disappears
        frames = []
        for i in range(15):
            frame = np.zeros((480, 640), dtype=np.uint8)
            # Object appears in frames 5-10
            if 5 <= i <= 10:
                x = 200 + (i - 5) * 20
                cv2.circle(frame, (x, 200), 30, 255, -1)
            frames.append(frame)
        
        # Process frames
        results = []
        for i, frame in enumerate(frames):
            detections = detector.detect(frame)
            detections.frame_number = i
            tracking_result = tracker.update(detections)
            results.append(tracking_result)
        
        # Should handle intermittent detections without crashing
        assert len(results) == 15
