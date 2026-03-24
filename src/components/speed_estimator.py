"""Speed estimation component for tracked objects"""
import logging
from typing import List, Optional, Tuple
import numpy as np

from src.models import TrackingResult, SpeedResult, CalibrationParameters
from src.config.models import SpeedEstimatorConfig


logger = logging.getLogger(__name__)


class SpeedEstimator:
    """Estimates object speeds from tracking data and calibration parameters"""
    
    def __init__(self, config: SpeedEstimatorConfig):
        """
        Initialize speed estimator with configuration
        
        Args:
            config: Speed estimator configuration parameters
        """
        self.config = config
        self._validate_config()
        logger.info(
            f"SpeedEstimator initialized: window={config.averaging_window_frames}, "
            f"min_trajectory={config.min_trajectory_length}, unit={config.output_unit}"
        )
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.averaging_window_frames < 1:
            raise ValueError("averaging_window_frames must be at least 1")
        
        if self.config.min_trajectory_length < 2:
            raise ValueError("min_trajectory_length must be at least 2")
        
        valid_units = ["m/s", "km/h", "mph"]
        if self.config.output_unit not in valid_units:
            raise ValueError(f"output_unit must be one of {valid_units}")
    
    def estimate_speeds(
        self,
        tracking: TrackingResult,
        calibration: Optional[CalibrationParameters],
        fps: float
    ) -> List[SpeedResult]:
        """
        Estimate speeds for all tracked objects
        
        Args:
            tracking: Current tracking results with trajectories
            calibration: Optional calibration parameters for spatial scaling
            fps: Video frame rate for temporal scaling
            
        Returns:
            List of speed estimates for each tracked object
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        
        results = []
        
        for tracked_obj in tracking.tracked_objects:
            # Check if trajectory has enough points
            if len(tracked_obj.trajectory) < self.config.min_trajectory_length:
                logger.debug(
                    f"Object {tracked_obj.object_id}: insufficient trajectory points "
                    f"({len(tracked_obj.trajectory)} < {self.config.min_trajectory_length})"
                )
                continue
            
            # Calculate instantaneous speed from last two points
            instantaneous_speed = self._calculate_instantaneous_speed(
                tracked_obj.trajectory, fps, calibration
            )
            
            # Calculate average speed over window
            average_speed = self._calculate_average_speed(
                tracked_obj.trajectory, fps, calibration
            )
            
            # Calculate displacement vector
            displacement_vector = self._calculate_displacement_vector(
                tracked_obj.trajectory
            )
            
            # Determine unit and calibration status
            if calibration is not None and calibration.pixels_per_meter is not None:
                unit = self.config.output_unit
                calibrated = True
            else:
                unit = "px/s"
                calibrated = False
                if calibration is None:
                    logger.debug(
                        f"Object {tracked_obj.object_id}: no calibration available, "
                        f"using pixel-based speed"
                    )
            
            # Create speed result
            result = SpeedResult(
                object_id=tracked_obj.object_id,
                instantaneous_speed=instantaneous_speed,
                average_speed=average_speed,
                displacement_vector=displacement_vector,
                unit=unit,
                calibrated=calibrated,
                confidence=1.0
            )
            
            results.append(result)
            
            logger.debug(
                f"Object {tracked_obj.object_id}: instant={instantaneous_speed:.2f} {unit}, "
                f"avg={average_speed:.2f} {unit}"
            )
        
        return results
    
    def _calculate_instantaneous_speed(
        self,
        trajectory: List[Tuple[float, float]],
        fps: float,
        calibration: Optional[CalibrationParameters]
    ) -> float:
        """
        Calculate instantaneous speed from last two trajectory points
        
        Args:
            trajectory: List of (x, y) positions
            fps: Frame rate for temporal scaling
            calibration: Optional calibration for spatial scaling
            
        Returns:
            Instantaneous speed in configured units
        """
        # Get last two points
        pos1 = trajectory[-2]
        pos2 = trajectory[-1]
        
        # Calculate pixel displacement
        pixel_displacement = self._calculate_distance(pos1, pos2)
        
        # Apply temporal scaling (displacement per frame -> displacement per second)
        velocity_px_per_sec = pixel_displacement * fps
        
        # Apply spatial scaling if calibration available
        if calibration is not None and calibration.pixels_per_meter is not None:
            velocity_m_per_sec = velocity_px_per_sec / calibration.pixels_per_meter
            # Convert to configured unit
            velocity = self._convert_unit(velocity_m_per_sec, "m/s", self.config.output_unit)
        else:
            velocity = velocity_px_per_sec
        
        return velocity
    
    def _calculate_average_speed(
        self,
        trajectory: List[Tuple[float, float]],
        fps: float,
        calibration: Optional[CalibrationParameters]
    ) -> float:
        """
        Calculate average speed over averaging window
        
        Args:
            trajectory: List of (x, y) positions
            fps: Frame rate for temporal scaling
            calibration: Optional calibration for spatial scaling
            
        Returns:
            Average speed in configured units
        """
        # Determine window size (limited by trajectory length)
        window_size = min(self.config.averaging_window_frames, len(trajectory))
        
        # Get trajectory points within window
        window_trajectory = trajectory[-window_size:]
        
        # Calculate speeds between consecutive points
        speeds = []
        for i in range(len(window_trajectory) - 1):
            pos1 = window_trajectory[i]
            pos2 = window_trajectory[i + 1]
            
            # Calculate pixel displacement
            pixel_displacement = self._calculate_distance(pos1, pos2)
            
            # Apply temporal scaling
            velocity_px_per_sec = pixel_displacement * fps
            
            # Apply spatial scaling if calibration available
            if calibration is not None and calibration.pixels_per_meter is not None:
                velocity_m_per_sec = velocity_px_per_sec / calibration.pixels_per_meter
                # Convert to configured unit
                velocity = self._convert_unit(velocity_m_per_sec, "m/s", self.config.output_unit)
            else:
                velocity = velocity_px_per_sec
            
            speeds.append(velocity)
        
        # Return mean speed
        if speeds:
            return float(np.mean(speeds))
        else:
            return 0.0
    
    def _calculate_displacement_vector(
        self,
        trajectory: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        Calculate displacement vector from last two points
        
        Args:
            trajectory: List of (x, y) positions
            
        Returns:
            Displacement vector (dx, dy)
        """
        pos1 = trajectory[-2]
        pos2 = trajectory[-1]
        
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        
        return (dx, dy)
    
    def _calculate_distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)
            
        Returns:
            Euclidean distance
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return float(np.sqrt(dx**2 + dy**2))
    
    def _convert_unit(self, speed_m_per_sec: float, from_unit: str, to_unit: str) -> float:
        """
        Convert speed between units
        
        Args:
            speed_m_per_sec: Speed in meters per second
            from_unit: Source unit (should be "m/s")
            to_unit: Target unit ("m/s", "km/h", "mph")
            
        Returns:
            Speed in target unit
        """
        if from_unit != "m/s":
            raise ValueError("from_unit must be 'm/s'")
        
        if to_unit == "m/s":
            return speed_m_per_sec
        elif to_unit == "km/h":
            return speed_m_per_sec * 3.6
        elif to_unit == "mph":
            return speed_m_per_sec * 2.23694
        else:
            raise ValueError(f"Unknown unit: {to_unit}")
