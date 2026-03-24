"""Core data models for the video processing pipeline"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class VideoFrame:
    """Represents a single video frame"""
    data: np.ndarray
    frame_number: int
    timestamp: float
    resolution: Tuple[int, int]


@dataclass
class BoundingBox:
    """Bounding box for detected object"""
    x: int
    y: int
    width: int
    height: int
    area: float
    centroid: Tuple[float, float]


@dataclass
class DetectionResult:
    """Results from object detection"""
    frame_number: int
    bounding_boxes: List[BoundingBox]
    contours: List[np.ndarray]
    foreground_mask: np.ndarray


@dataclass
class TrackedObject:
    """Tracked object with identity and history"""
    object_id: int
    position: Tuple[float, float]
    bounding_box: BoundingBox
    trajectory: List[Tuple[float, float]]
    age: int
    disappeared_count: int


@dataclass
class TrackingResult:
    """Results from object tracking"""
    frame_number: int
    tracked_objects: List[TrackedObject]


@dataclass
class CalibrationParameters:
    """Camera calibration parameters"""
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    homography_matrix: Optional[np.ndarray] = None
    pixels_per_meter: Optional[float] = None
    calibration_error: float = 0.0
    calibration_date: str = ""


@dataclass
class SpeedResult:
    """Speed estimation result for tracked object"""
    object_id: int
    instantaneous_speed: float
    average_speed: float
    displacement_vector: Tuple[float, float]
    unit: str
    calibrated: bool
    confidence: float = 1.0


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    frame_number: int
    timestamp: float
    annotated_frame: np.ndarray
    detections: Optional[DetectionResult]
    tracking: Optional[TrackingResult]
    speeds: Optional[List[SpeedResult]]
    processing_time_ms: float
    component_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComponentStats:
    """Statistics for a pipeline component"""
    component_name: str
    average_time_ms: float
    max_time_ms: float
    error_count: int


@dataclass
class PipelineSummary:
    """Summary statistics for video processing"""
    total_frames: int
    processed_frames: int
    average_fps: float
    total_objects_detected: int
    unique_objects_tracked: int
    average_speed: Optional[float]
    max_speed: Optional[float]
    processing_errors: List[str]
    component_statistics: Dict[str, ComponentStats]
