"""Configuration data models"""
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PreprocessorConfig:
    """Configuration for video preprocessor"""
    target_resolution: Tuple[int, int] = (1280, 720)
    noise_reduction_method: str = "gaussian"
    noise_reduction_kernel_size: int = 5
    normalize_intensity: bool = True


@dataclass
class DetectorConfig:
    """Configuration for object detector"""
    background_subtraction_method: str = "MOG2"
    background_learning_rate: float = 0.01
    edge_detection_enabled: bool = False
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    min_contour_area: float = 500.0
    max_contour_area: float = 50000.0


@dataclass
class TrackerConfig:
    """Configuration for object tracker"""
    max_tracking_distance: float = 50.0
    max_disappeared_frames: int = 30
    trajectory_history_length: int = 100


@dataclass
class CalibratorConfig:
    """Configuration for camera calibrator"""
    calibration_file_path: str = "calibration.json"
    chessboard_size: Tuple[int, int] = (9, 6)
    square_size_mm: float = 25.0
    perspective_transform_enabled: bool = False


@dataclass
class SpeedEstimatorConfig:
    """Configuration for speed estimator"""
    averaging_window_frames: int = 10
    min_trajectory_length: int = 2
    output_unit: str = "m/s"


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    file_path: Optional[str] = None
    log_to_file: bool = False


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""
    enabled_components: List[str]
    preprocessor: PreprocessorConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    calibrator: CalibratorConfig
    speed_estimator: SpeedEstimatorConfig
    logging: LoggingConfig
