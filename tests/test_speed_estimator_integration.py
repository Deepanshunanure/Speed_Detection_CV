"""Integration tests for SpeedEstimator component"""
import pytest
import numpy as np
from src.components.speed_estimator import SpeedEstimator
from src.components.tracker import Tracker
from src.components.detector import Detector
from src.models import DetectionResult, BoundingBox, CalibrationParameters
from src.config.models import SpeedEstimatorConfig, TrackerConfig, DetectorConfig


@pytest.fixture
def speed_estimator():
    """Create speed estimator with default config"""
    config = SpeedEstimatorConfig(
        averaging_window_frames=5,
        min_trajectory_length=2,
        output_unit="m/s"
    )
    return SpeedEstimator(config)


@pytest.fixture
def tracker():
    """Create tracker with default config"""
    config = TrackerConfig(
        max_tracking_distance=50.0,
        max_disappeared_frames=30,
        trajectory_history_length=100
    )
    return Tracker(config)


@pytest.fixture
def calibration():
    """Create calibration parameters"""
    return CalibrationParameters(
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        pixels_per_meter=40.0  # 40 pixels = 1 meter
    )


def create_detection_result(frame_number: int, bboxes: list) -> DetectionResult:
    """Helper to create detection result"""
    boxes = []
    contours = []
    
    for x, y, w, h in bboxes:
        bbox = BoundingBox(
            x=x, y=y, width=w, height=h,
            area=w * h,
            centroid=(x + w/2, y + h/2)
        )
        boxes.append(bbox)
        # Create simple rectangular contour
        contour = np.array([
            [[x, y]], [[x+w, y]], [[x+w, y+h]], [[x, y+h]]
        ])
        contours.append(contour)
    
    return DetectionResult(
        frame_number=frame_number,
        bounding_boxes=boxes,
        contours=contours,
        foreground_mask=np.zeros((480, 640), dtype=np.uint8)
    )


class TestSpeedEstimatorIntegration:
    """Integration tests with tracker"""
    
    def test_single_object_constant_speed(self, speed_estimator, tracker, calibration):
        """Test speed estimation for single object moving at constant speed"""
        # Simulate object moving 20 pixels right per frame
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(120, 100, 50, 50)]),
            create_detection_result(3, [(140, 100, 50, 50)]),
            create_detection_result(4, [(160, 100, 50, 50)]),
            create_detection_result(5, [(180, 100, 50, 50)]),
        ]
        
        fps = 30.0
        speeds_over_time = []
        
        for detection in detections:
            tracking = tracker.update(detection)
            if len(tracking.tracked_objects) > 0:
                speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
                if speeds:
                    speeds_over_time.append(speeds[0])
        
        # Should have speed results after first 2 frames
        assert len(speeds_over_time) >= 3
        
        # All speeds should be similar (constant velocity)
        instantaneous_speeds = [s.instantaneous_speed for s in speeds_over_time]
        avg_speed = np.mean(instantaneous_speeds)
        
        # Expected: 20 px/frame * 30 fps / 40 px/m = 15 m/s
        assert abs(avg_speed - 15.0) < 1.0
        
        # Check that all results are calibrated
        assert all(s.calibrated for s in speeds_over_time)
        assert all(s.unit == "m/s" for s in speeds_over_time)
    
    def test_single_object_accelerating(self, speed_estimator, tracker, calibration):
        """Test speed estimation for accelerating object"""
        # Simulate object accelerating (5, 10, 15, 20 pixels per frame)
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(105, 100, 50, 50)]),
            create_detection_result(3, [(115, 100, 50, 50)]),
            create_detection_result(4, [(130, 100, 50, 50)]),
            create_detection_result(5, [(150, 100, 50, 50)]),
        ]
        
        fps = 30.0
        speeds_over_time = []
        
        for detection in detections:
            tracking = tracker.update(detection)
            if len(tracking.tracked_objects) > 0:
                speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
                if speeds:
                    speeds_over_time.append(speeds[0])
        
        assert len(speeds_over_time) >= 3
        
        # Instantaneous speeds should increase over time
        instantaneous_speeds = [s.instantaneous_speed for s in speeds_over_time]
        assert instantaneous_speeds[-1] > instantaneous_speeds[0]
    
    def test_multiple_objects_different_speeds(self, speed_estimator, tracker, calibration):
        """Test speed estimation for multiple objects moving at different speeds"""
        # Object 1: slow (5 px/frame), Object 2: fast (20 px/frame)
        detections = [
            create_detection_result(1, [(100, 100, 50, 50), (300, 100, 50, 50)]),
            create_detection_result(2, [(105, 100, 50, 50), (320, 100, 50, 50)]),
            create_detection_result(3, [(110, 100, 50, 50), (340, 100, 50, 50)]),
            create_detection_result(4, [(115, 100, 50, 50), (360, 100, 50, 50)]),
        ]
        
        fps = 30.0
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        # Get final speeds
        speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
        
        assert len(speeds) == 2
        
        # Object 1: 5 px/frame * 30 fps / 40 px/m = 3.75 m/s
        # Object 2: 20 px/frame * 30 fps / 40 px/m = 15 m/s
        speeds_sorted = sorted(speeds, key=lambda s: s.instantaneous_speed)
        assert abs(speeds_sorted[0].instantaneous_speed - 3.75) < 1.0
        assert abs(speeds_sorted[1].instantaneous_speed - 15.0) < 1.0
    
    def test_object_stopping_and_starting(self, speed_estimator, tracker, calibration):
        """Test speed estimation for object that stops and starts"""
        # Object moves, stops, then moves again
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(120, 100, 50, 50)]),  # moving
            create_detection_result(3, [(120, 100, 50, 50)]),  # stopped
            create_detection_result(4, [(120, 100, 50, 50)]),  # stopped
            create_detection_result(5, [(140, 100, 50, 50)]),  # moving again
        ]
        
        fps = 30.0
        speeds_over_time = []
        
        for detection in detections:
            tracking = tracker.update(detection)
            if len(tracking.tracked_objects) > 0:
                speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
                if speeds:
                    speeds_over_time.append(speeds[0])
        
        assert len(speeds_over_time) >= 4
        
        # Check that speed drops to zero when stopped
        instantaneous_speeds = [s.instantaneous_speed for s in speeds_over_time]
        assert min(instantaneous_speeds) == 0.0
        assert max(instantaneous_speeds) > 0.0
    
    def test_without_calibration(self, speed_estimator, tracker):
        """Test speed estimation without calibration (pixel-based)"""
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(110, 100, 50, 50)]),
            create_detection_result(3, [(120, 100, 50, 50)]),
        ]
        
        fps = 30.0
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        speeds = speed_estimator.estimate_speeds(tracking, None, fps)
        
        assert len(speeds) == 1
        assert speeds[0].calibrated is False
        assert speeds[0].unit == "px/s"
        # 10 px/frame * 30 fps = 300 px/s
        assert abs(speeds[0].instantaneous_speed - 300.0) < 10.0


class TestSpeedEstimatorWithDifferentUnits:
    """Test speed estimation with different output units"""
    
    def test_kmh_output(self, tracker, calibration):
        """Test speed output in km/h"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=5,
            min_trajectory_length=2,
            output_unit="km/h"
        )
        estimator = SpeedEstimator(config)
        
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(120, 100, 50, 50)]),
        ]
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        speeds = estimator.estimate_speeds(tracking, calibration, fps=30.0)
        
        assert len(speeds) == 1
        assert speeds[0].unit == "km/h"
        # 20 px/frame * 30 fps / 40 px/m = 15 m/s = 54 km/h
        assert abs(speeds[0].instantaneous_speed - 54.0) < 5.0
    
    def test_mph_output(self, tracker, calibration):
        """Test speed output in mph"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=5,
            min_trajectory_length=2,
            output_unit="mph"
        )
        estimator = SpeedEstimator(config)
        
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(120, 100, 50, 50)]),
        ]
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        speeds = estimator.estimate_speeds(tracking, calibration, fps=30.0)
        
        assert len(speeds) == 1
        assert speeds[0].unit == "mph"
        # 20 px/frame * 30 fps / 40 px/m = 15 m/s = 33.55 mph
        assert abs(speeds[0].instantaneous_speed - 33.55) < 5.0


class TestSpeedEstimatorEdgeCases:
    """Test edge cases in integration scenarios"""
    
    def test_object_appearing_and_disappearing(self, speed_estimator, tracker, calibration):
        """Test speed estimation when object temporarily disappears"""
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(110, 100, 50, 50)]),
            create_detection_result(3, []),  # Object disappears
            create_detection_result(4, []),  # Still gone
            create_detection_result(5, [(130, 100, 50, 50)]),  # Reappears
        ]
        
        fps = 30.0
        
        for detection in detections:
            tracking = tracker.update(detection)
            speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
            # Should handle gracefully without errors
            assert isinstance(speeds, list)
    
    def test_very_high_fps(self, speed_estimator, tracker, calibration):
        """Test speed estimation with very high frame rate"""
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(101, 100, 50, 50)]),  # Small movement
        ]
        
        fps = 240.0  # High-speed camera
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
        
        assert len(speeds) == 1
        # 1 px/frame * 240 fps / 40 px/m = 6 m/s
        assert abs(speeds[0].instantaneous_speed - 6.0) < 1.0
    
    def test_very_low_fps(self, speed_estimator, tracker, calibration):
        """Test speed estimation with low frame rate"""
        detections = [
            create_detection_result(1, [(100, 100, 50, 50)]),
            create_detection_result(2, [(150, 100, 50, 50)]),  # Large movement
        ]
        
        fps = 5.0  # Low frame rate
        
        for detection in detections:
            tracking = tracker.update(detection)
        
        speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
        
        assert len(speeds) == 1
        # 50 px/frame * 5 fps / 40 px/m = 6.25 m/s
        assert abs(speeds[0].instantaneous_speed - 6.25) < 1.0
