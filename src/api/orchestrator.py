"""Pipeline orchestrator for coordinating component execution"""
import logging
import json
import time
import psutil
import gc
from typing import Optional, Dict, List
import cv2
import numpy as np

from src.config.models import PipelineConfig
from src.models import (
    PipelineResult, PipelineSummary, ComponentStats,
    DetectionResult, TrackingResult, SpeedResult, CalibrationParameters
)
from src.components.preprocessor import Preprocessor
from src.components.detector import Detector
from src.components.tracker import Tracker
from src.components.speed_estimator import SpeedEstimator


logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates execution of all pipeline components"""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize all enabled components from configuration
        
        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.calibration: Optional[CalibrationParameters] = None
        
        # Initialize enabled components
        self.preprocessor = None
        self.detector = None
        self.tracker = None
        self.speed_estimator = None
        
        try:
            if "preprocessor" in config.enabled_components:
                self.preprocessor = Preprocessor(config.preprocessor)
                logger.info(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "component_initialized",
                    "component": "preprocessor"
                }))
            
            if "detector" in config.enabled_components:
                self.detector = Detector(config.detector)
                logger.info(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "component_initialized",
                    "component": "detector"
                }))
            
            if "tracker" in config.enabled_components:
                self.tracker = Tracker(config.tracker)
                logger.info(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "component_initialized",
                    "component": "tracker"
                }))
            
            if "speed_estimator" in config.enabled_components:
                self.speed_estimator = SpeedEstimator(config.speed_estimator)
                logger.info(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "component_initialized",
                    "component": "speed_estimator"
                }))
            
            logger.info(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "initialized",
                "enabled_components": config.enabled_components
            }))
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "initialization_failed",
                "error": str(e)
            }))
            raise
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> PipelineResult:
        """
        Process single frame through enabled components
        
        Args:
            frame: Input video frame (BGR format)
            frame_number: Frame sequence number
            
        Returns:
            Complete pipeline result with annotations and timing
        """
        start_time = time.time()
        component_times: Dict[str, float] = {}
        
        # Initialize result variables
        processed_frame = frame.copy()
        detections: Optional[DetectionResult] = None
        tracking: Optional[TrackingResult] = None
        speeds: Optional[List[SpeedResult]] = None
        
        # Step 1: Preprocessing
        if self.preprocessor is not None:
            try:
                comp_start = time.time()
                processed_frame = self.preprocessor.process(frame)
                component_times["preprocessor"] = (time.time() - comp_start) * 1000
                logger.debug(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "preprocessing_complete",
                    "frame_number": frame_number,
                    "processing_time_ms": round(component_times["preprocessor"], 2)
                }))
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "preprocessing_failed",
                    "frame_number": frame_number,
                    "error": str(e)
                }), exc_info=True)
                processed_frame = frame.copy()
        
        # Step 2: Detection
        if self.detector is not None:
            try:
                comp_start = time.time()
                detections = self.detector.detect(processed_frame)
                detections.frame_number = frame_number
                component_times["detector"] = (time.time() - comp_start) * 1000
                logger.debug(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "detection_complete",
                    "frame_number": frame_number,
                    "num_detections": len(detections.bounding_boxes),
                    "processing_time_ms": round(component_times["detector"], 2)
                }))
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "detection_failed",
                    "frame_number": frame_number,
                    "error": str(e)
                }), exc_info=True)
                detections = None
        
        # Step 3: Tracking
        if self.tracker is not None and detections is not None:
            try:
                comp_start = time.time()
                tracking = self.tracker.update(detections)
                component_times["tracker"] = (time.time() - comp_start) * 1000
                logger.debug(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "tracking_complete",
                    "frame_number": frame_number,
                    "num_tracked_objects": len(tracking.tracked_objects),
                    "processing_time_ms": round(component_times["tracker"], 2)
                }))
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "tracking_failed",
                    "frame_number": frame_number,
                    "error": str(e)
                }), exc_info=True)
                tracking = None
        
        # Step 4: Speed Estimation
        if self.speed_estimator is not None and tracking is not None:
            try:
                comp_start = time.time()
                # Assume 30 fps if not specified (will be overridden in process_video)
                speeds = self.speed_estimator.estimate_speeds(tracking, self.calibration, fps=30.0)
                component_times["speed_estimator"] = (time.time() - comp_start) * 1000
                logger.debug(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "speed_estimation_complete",
                    "frame_number": frame_number,
                    "num_speeds": len(speeds),
                    "processing_time_ms": round(component_times["speed_estimator"], 2)
                }))
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "speed_estimation_failed",
                    "frame_number": frame_number,
                    "error": str(e)
                }), exc_info=True)
                speeds = None
        
        # Step 5: Frame Annotation
        try:
            annotated_frame = self._annotate_frame(
                frame, detections, tracking, speeds
            )
        except Exception as e:
            logger.warning(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "annotation_failed",
                "frame_number": frame_number,
                "error": str(e)
            }))
            annotated_frame = frame.copy()
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create pipeline result
        result = PipelineResult(
            frame_number=frame_number,
            timestamp=frame_number / 30.0,  # Placeholder, will be updated in process_video
            annotated_frame=annotated_frame,
            detections=detections,
            tracking=tracking,
            speeds=speeds,
            processing_time_ms=processing_time_ms,
            component_times=component_times
        )
        
        return result
    
    def _check_memory_usage(self, frame_number: int) -> None:
        """
        Monitor memory usage and log warnings if threshold exceeded
        
        Args:
            frame_number: Current frame number for logging context
        """
        try:
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent >= 95:
                logger.critical(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "memory_critical",
                    "frame_number": frame_number,
                    "memory_usage_percent": memory_percent
                }))
                raise MemoryError(f"Memory usage critical: {memory_percent}%")
            elif memory_percent >= 85:
                logger.warning(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "memory_high",
                    "frame_number": frame_number,
                    "memory_usage_percent": memory_percent
                }))
                # Trigger garbage collection
                gc.collect()
            elif memory_percent >= 80:
                logger.warning(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "memory_warning",
                    "frame_number": frame_number,
                    "memory_usage_percent": memory_percent
                }))
        except Exception as e:
            logger.debug(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "memory_check_failed",
                "frame_number": frame_number,
                "error": str(e)
            }))
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: Optional[DetectionResult],
        tracking: Optional[TrackingResult],
        speeds: Optional[List[SpeedResult]]
    ) -> np.ndarray:
        """
        Annotate frame with bounding boxes, IDs, and speeds
        
        Args:
            frame: Original input frame
            detections: Detection results (optional)
            tracking: Tracking results (optional)
            speeds: Speed results (optional)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # If we have tracking results, use them for annotation
        if tracking is not None:
            # Create speed lookup dictionary
            speed_lookup = {}
            if speeds is not None:
                speed_lookup = {s.object_id: s for s in speeds}
            
            for tracked_obj in tracking.tracked_objects:
                bbox = tracked_obj.bounding_box
                obj_id = tracked_obj.object_id
                
                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (bbox.x, bbox.y),
                    (bbox.x + bbox.width, bbox.y + bbox.height),
                    (0, 255, 0),  # Green
                    2
                )
                
                # Draw object ID
                label = f"ID: {obj_id}"
                
                # Add speed if available
                if obj_id in speed_lookup:
                    speed_result = speed_lookup[obj_id]
                    label += f" | {speed_result.instantaneous_speed:.1f} {speed_result.unit}"
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (bbox.x, bbox.y - label_height - baseline - 5),
                    (bbox.x + label_width, bbox.y),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (bbox.x, bbox.y - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Black text
                    1
                )
                
                # Draw trajectory
                if len(tracked_obj.trajectory) > 1:
                    for i in range(len(tracked_obj.trajectory) - 1):
                        pt1 = (int(tracked_obj.trajectory[i][0]), int(tracked_obj.trajectory[i][1]))
                        pt2 = (int(tracked_obj.trajectory[i + 1][0]), int(tracked_obj.trajectory[i + 1][1]))
                        cv2.line(annotated, pt1, pt2, (255, 0, 0), 1)  # Blue trajectory
        
        # If no tracking but we have detections, draw detection boxes
        elif detections is not None:
            for bbox in detections.bounding_boxes:
                cv2.rectangle(
                    annotated,
                    (bbox.x, bbox.y),
                    (bbox.x + bbox.width, bbox.y + bbox.height),
                    (0, 0, 255),  # Red for untracked detections
                    2
                )
        
        return annotated
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> PipelineSummary:
        """
        Process entire video file and generate summary
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save annotated video
            
        Returns:
            Summary statistics for the video processing
        """
        logger.info(json.dumps({
            "component_name": "PipelineOrchestrator",
            "event": "video_processing_started",
            "video_path": video_path,
            "output_path": output_path
        }))
        
        # Open video file
        cap = None
        writer = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                error_msg = f"Failed to open video file: {video_path}"
                logger.error(json.dumps({
                    "component_name": "PipelineOrchestrator",
                    "event": "video_open_failed",
                    "video_path": video_path,
                    "error": error_msg
                }))
                raise ValueError(error_msg)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "video_properties",
                "total_frames": total_frames,
                "fps": fps,
                "resolution": f"{frame_width}x{frame_height}"
            }))
            
            # Initialize video writer if output path specified
            if output_path is not None:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                    logger.info(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_writer_initialized",
                        "output_path": output_path
                    }))
                except Exception as e:
                    logger.error(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_writer_initialization_failed",
                        "output_path": output_path,
                        "error": str(e)
                    }))
                    writer = None
            
            # Processing statistics
            processed_frames = 0
            processing_errors: List[str] = []
            component_times_accumulator: Dict[str, List[float]] = {
                "preprocessor": [],
                "detector": [],
                "tracker": [],
                "speed_estimator": []
            }
            
            all_speeds: List[float] = []
            unique_object_ids = set()
            total_detections = 0
            
            start_time = time.time()
            frame_number = 0
            
            # Process each frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Check memory usage every 100 frames
                    if frame_number % 100 == 0:
                        self._check_memory_usage(frame_number)
                    
                    # Process frame with actual fps
                    result = self.process_frame(frame, frame_number)
                    
                    # Update timestamp with actual fps
                    result.timestamp = frame_number / fps
                    
                    # Update statistics
                    processed_frames += 1
                    
                    # Accumulate component times
                    for component, comp_time in result.component_times.items():
                        if component in component_times_accumulator:
                            component_times_accumulator[component].append(comp_time)
                    
                    # Count detections
                    if result.detections is not None:
                        total_detections += len(result.detections.bounding_boxes)
                    
                    # Track unique objects
                    if result.tracking is not None:
                        for tracked_obj in result.tracking.tracked_objects:
                            unique_object_ids.add(tracked_obj.object_id)
                    
                    # Collect speeds
                    if result.speeds is not None:
                        for speed_result in result.speeds:
                            all_speeds.append(speed_result.instantaneous_speed)
                    
                    # Write annotated frame to output video
                    if writer is not None:
                        try:
                            writer.write(result.annotated_frame)
                        except Exception as e:
                            logger.warning(json.dumps({
                                "component_name": "PipelineOrchestrator",
                                "event": "frame_write_failed",
                                "frame_number": frame_number,
                                "error": str(e)
                            }))
                    
                    # Log progress every 100 frames
                    if (frame_number + 1) % 100 == 0:
                        elapsed = time.time() - start_time
                        current_fps = (frame_number + 1) / elapsed
                        logger.info(json.dumps({
                            "component_name": "PipelineOrchestrator",
                            "event": "processing_progress",
                            "frames_processed": frame_number + 1,
                            "total_frames": total_frames,
                            "current_fps": round(current_fps, 2)
                        }))
                
                except Exception as e:
                    error_msg = f"Frame {frame_number}: processing failed: {e}"
                    logger.error(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "frame_processing_failed",
                        "frame_number": frame_number,
                        "error": str(e)
                    }), exc_info=True)
                    processing_errors.append(error_msg)
                
                frame_number += 1
            
            # Calculate summary statistics
            total_time = time.time() - start_time
            average_fps = processed_frames / total_time if total_time > 0 else 0.0
            
            # Calculate component statistics
            component_statistics: Dict[str, ComponentStats] = {}
            for component, times in component_times_accumulator.items():
                if times:
                    component_statistics[component] = ComponentStats(
                        component_name=component,
                        average_time_ms=float(np.mean(times)),
                        max_time_ms=float(np.max(times)),
                        error_count=0  # Errors are logged but not tracked per component
                    )
            
            # Calculate speed statistics
            average_speed = float(np.mean(all_speeds)) if all_speeds else None
            max_speed = float(np.max(all_speeds)) if all_speeds else None
            
            # Create summary
            summary = PipelineSummary(
                total_frames=total_frames,
                processed_frames=processed_frames,
                average_fps=average_fps,
                total_objects_detected=total_detections,
                unique_objects_tracked=len(unique_object_ids),
                average_speed=average_speed,
                max_speed=max_speed,
                processing_errors=processing_errors,
                component_statistics=component_statistics
            )
            
            logger.info(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "video_processing_complete",
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "average_fps": round(average_fps, 2),
                "unique_objects_tracked": len(unique_object_ids),
                "error_count": len(processing_errors)
            }))
            
            return summary
            
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "PipelineOrchestrator",
                "event": "video_processing_failed",
                "video_path": video_path,
                "error": str(e)
            }), exc_info=True)
            raise
        finally:
            # Resource cleanup
            if cap is not None:
                try:
                    cap.release()
                    logger.debug(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_capture_released"
                    }))
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_capture_release_failed",
                        "error": str(e)
                    }))
            
            if writer is not None:
                try:
                    writer.release()
                    logger.debug(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_writer_released"
                    }))
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "PipelineOrchestrator",
                        "event": "video_writer_release_failed",
                        "error": str(e)
                    }))
