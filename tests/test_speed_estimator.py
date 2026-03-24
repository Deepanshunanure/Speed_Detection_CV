"""Unit tests for SpeedEstimator component"""
import pytest
import numpy as np
from src.components.speed_estimator import SpeedEstimator
from src.models import TrackingResult, TrackedObject, BoundingBox, CalibrationParameters, SpeedResult
from src.config.models import SpeedEstimatorConfig


@pytest.fixture
def default_config():
    """Default speed estimator configuration"""
    return SpeedEstimatorConfig(
        averaging_window_frames=10,
        min_trajectory_length=2,
        output_unit="m/s"
    )


@pytest.fixture
def calibration_params():
    """Sample calibration parameters"""
    return CalibrationParameters(
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        pixels_per_meter=50.0  # 50 pixels = 1 meter
    )


@pytest.fixture
def sample_trajectory():
    """Sample trajectory with 5 points"""
    return [
        (100.0, 100.0),
        (110.0, 100.0),  # moved 10 pixels right
        (120.0, 100.0),  # moved 10 pixels right
        (130.0, 100.0),  # moved 10 pixels right
        (140.0, 100.0),  # moved 10 pixels right
    ]


def create_tracked_object(object_id: int, trajectory: list) -> TrackedObject:
    """Helper to create a tracked object"""
    bbox = BoundingBox(x=100, y=100, width=50, height=50, area=2500, centroid=(125.0, 125.0))
    return TrackedObject(
        object_id=object_id,
        position=trajectory[-1],
        bounding_box=bbox,
        trajectory=trajectory,
        age=len(trajectory),
        disappeared_count=0
    )


class TestSpeedEstimatorInit:
    """Tests for SpeedEstimator initialization"""
    
    def test_init_with_valid_config(self, default_config):
        """Test initialization with valid configuration"""
        estimator = SpeedEstimator(default_config)
        assert estimator.config == default_config
    
    def test_init_with_invalid_window_frames(self):
        """Test initialization fails with invalid averaging window"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=0,
            min_trajectory_length=2,
            output_unit="m/s"
        )
        with pytest.raises(ValueError, match="averaging_window_frames must be at least 1"):
            SpeedEstimator(config)
    
    def test_init_with_invalid_min_trajectory(self):
        """Test initialization fails with invalid min trajectory length"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=1,
            output_unit="m/s"
        )
        with pytest.raises(ValueError, match="min_trajectory_length must be at least 2"):
            SpeedEstimator(config)
    
    def test_init_with_invalid_unit(self):
        """Test initialization fails with invalid output unit"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="invalid"
        )
        with pytest.raises(ValueError, match="output_unit must be one of"):
            SpeedEstimator(config)


class TestEstimateSpeeds:
    """Tests for estimate_speeds method"""
    
    def test_estimate_speeds_with_calibration(self, default_config, calibration_params, sample_trajectory):
        """Test speed estimation with calibration parameters"""
        estimator = SpeedEstimator(default_config)
        
        tracked_obj = create_tracked_object(1, sample_trajectory)
        tracking = TrackingResult(frame_number=5, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        result = results[0]
        assert result.object_id == 1
        assert result.calibrated is True
        assert result.unit == "m/s"
        assert result.instantaneous_speed > 0
        assert result.average_speed > 0
    
    def test_estimate_speeds_without_calibration(self, default_config, sample_trajectory):
        """Test speed estimation without calibration (pixel-based)"""
        estimator = SpeedEstimator(default_config)
        
        tracked_obj = create_tracked_object(1, sample_trajectory)
        tracking = TrackingResult(frame_number=5, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, None, fps=30.0)
        
        assert len(results) == 1
        result = results[0]
        assert result.object_id == 1
        assert result.calibrated is False
        assert result.unit == "px/s"
        assert result.instantaneous_speed > 0
        assert result.average_speed > 0
    
    def test_estimate_speeds_insufficient_trajectory(self, default_config):
        """Test that objects with insufficient trajectory are skipped"""
        estimator = SpeedEstimator(default_config)
        
        # Only 1 point in trajectory (need at least 2)
        tracked_obj = create_tracked_object(1, [(100.0, 100.0)])
        tracking = TrackingResult(frame_number=1, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, None, fps=30.0)
        
        assert len(results) == 0
    
    def test_estimate_speeds_multiple_objects(self, default_config, calibration_params):
        """Test speed estimation for multiple tracked objects"""
        estimator = SpeedEstimator(default_config)
        
        obj1 = create_tracked_object(1, [(100.0, 100.0), (110.0, 100.0)])
        obj2 = create_tracked_object(2, [(200.0, 200.0), (210.0, 200.0)])
        tracking = TrackingResult(frame_number=2, tracked_objects=[obj1, obj2])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 2
        assert results[0].object_id == 1
        assert results[1].object_id == 2
    
    def test_estimate_speeds_invalid_fps(self, default_config, sample_trajectory):
        """Test that invalid fps raises error"""
        estimator = SpeedEstimator(default_config)
        
        tracked_obj = create_tracked_object(1, sample_trajectory)
        tracking = TrackingResult(frame_number=5, tracked_objects=[tracked_obj])
        
        with pytest.raises(ValueError, match="fps must be positive"):
            estimator.estimate_speeds(tracking, None, fps=0.0)


class TestInstantaneousSpeed:
    """Tests for instantaneous speed calculation"""
    
    def test_instantaneous_speed_horizontal_movement(self, default_config, calibration_params):
        """Test instantaneous speed for horizontal movement"""
        estimator = SpeedEstimator(default_config)
        
        # Move 10 pixels right at 30 fps with 50 px/m calibration
        # Expected: 10 px * 30 fps / 50 px/m = 6 m/s
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        assert abs(results[0].instantaneous_speed - 6.0) < 0.01
    
    def test_instantaneous_speed_diagonal_movement(self, default_config, calibration_params):
        """Test instantaneous speed for diagonal movement"""
        estimator = SpeedEstimator(default_config)
        
        # Move 3 pixels right, 4 pixels up (5 pixels total) at 30 fps with 50 px/m
        # Expected: 5 px * 30 fps / 50 px/m = 3 m/s
        trajectory = [(100.0, 100.0), (103.0, 104.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        assert abs(results[0].instantaneous_speed - 3.0) < 0.01
    
    def test_instantaneous_speed_no_movement(self, default_config, calibration_params):
        """Test instantaneous speed when object is stationary"""
        estimator = SpeedEstimator(default_config)
        
        trajectory = [(100.0, 100.0), (100.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        assert results[0].instantaneous_speed == 0.0


class TestAverageSpeed:
    """Tests for average speed calculation"""
    
    def test_average_speed_constant_velocity(self, default_config, calibration_params):
        """Test average speed with constant velocity"""
        estimator = SpeedEstimator(default_config)
        
        # Constant 10 px/frame movement
        trajectory = [(100.0, 100.0), (110.0, 100.0), (120.0, 100.0), (130.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=4, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        # All segments are 10 px, so average should be 6 m/s
        assert abs(results[0].average_speed - 6.0) < 0.01
    
    def test_average_speed_varying_velocity(self, default_config, calibration_params):
        """Test average speed with varying velocity"""
        estimator = SpeedEstimator(default_config)
        
        # Varying movement: 10px, 20px, 30px
        trajectory = [(100.0, 100.0), (110.0, 100.0), (130.0, 100.0), (160.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=4, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        # Average of 10, 20, 30 px = 20 px/frame = 12 m/s
        assert abs(results[0].average_speed - 12.0) < 0.01
    
    def test_average_speed_window_limit(self, default_config, calibration_params):
        """Test that averaging window is limited to configured size"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=3,  # Only use last 3 frames
            min_trajectory_length=2,
            output_unit="m/s"
        )
        estimator = SpeedEstimator(config)
        
        # 5 points, but only last 3 should be used for averaging
        trajectory = [(100.0, 100.0), (110.0, 100.0), (120.0, 100.0), (130.0, 100.0), (140.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=5, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        # Should only average over last 3 points (2 segments)
        assert results[0].average_speed > 0


class TestUnitConversion:
    """Tests for unit conversion"""
    
    def test_speed_in_meters_per_second(self, calibration_params):
        """Test speed output in m/s"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="m/s"
        )
        estimator = SpeedEstimator(config)
        
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].unit == "m/s"
        assert abs(results[0].instantaneous_speed - 6.0) < 0.01
    
    def test_speed_in_kilometers_per_hour(self, calibration_params):
        """Test speed output in km/h"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="km/h"
        )
        estimator = SpeedEstimator(config)
        
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].unit == "km/h"
        # 6 m/s * 3.6 = 21.6 km/h
        assert abs(results[0].instantaneous_speed - 21.6) < 0.01
    
    def test_speed_in_miles_per_hour(self, calibration_params):
        """Test speed output in mph"""
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="mph"
        )
        estimator = SpeedEstimator(config)
        
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].unit == "mph"
        # 6 m/s * 2.23694 = 13.42 mph
        assert abs(results[0].instantaneous_speed - 13.42) < 0.1


class TestDisplacementVector:
    """Tests for displacement vector calculation"""
    
    def test_displacement_vector_horizontal(self, default_config, calibration_params):
        """Test displacement vector for horizontal movement"""
        estimator = SpeedEstimator(default_config)
        
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].displacement_vector == (10.0, 0.0)
    
    def test_displacement_vector_vertical(self, default_config, calibration_params):
        """Test displacement vector for vertical movement"""
        estimator = SpeedEstimator(default_config)
        
        trajectory = [(100.0, 100.0), (100.0, 110.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].displacement_vector == (0.0, 10.0)
    
    def test_displacement_vector_diagonal(self, default_config, calibration_params):
        """Test displacement vector for diagonal movement"""
        estimator = SpeedEstimator(default_config)
        
        trajectory = [(100.0, 100.0), (103.0, 104.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert results[0].displacement_vector == (3.0, 4.0)


class TestEdgeCases:
    """Tests for edge cases and error handling"""
    
    def test_empty_tracking_result(self, default_config, calibration_params):
        """Test with no tracked objects"""
        estimator = SpeedEstimator(default_config)
        
        tracking = TrackingResult(frame_number=1, tracked_objects=[])
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 0
    
    def test_calibration_without_pixels_per_meter(self, default_config):
        """Test with calibration but no pixels_per_meter"""
        estimator = SpeedEstimator(default_config)
        
        calibration = CalibrationParameters(
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
            pixels_per_meter=None  # No spatial scaling
        )
        
        trajectory = [(100.0, 100.0), (110.0, 100.0)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration, fps=30.0)
        
        assert len(results) == 1
        assert results[0].calibrated is False
        assert results[0].unit == "px/s"
    
    def test_very_small_displacement(self, default_config, calibration_params):
        """Test with very small displacement"""
        estimator = SpeedEstimator(default_config)
        
        trajectory = [(100.0, 100.0), (100.01, 100.01)]
        tracked_obj = create_tracked_object(1, trajectory)
        tracking = TrackingResult(frame_number=2, tracked_objects=[tracked_obj])
        
        results = estimator.estimate_speeds(tracking, calibration_params, fps=30.0)
        
        assert len(results) == 1
        assert results[0].instantaneous_speed >= 0
        assert results[0].instantaneous_speed < 1.0
