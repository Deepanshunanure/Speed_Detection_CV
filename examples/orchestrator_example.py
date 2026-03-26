"""Example usage of PipelineOrchestrator for video processing"""
import cv2
import numpy as np
import logging
from pathlib import Path

from src.api.orchestrator import PipelineOrchestrator
from src.config.models import (
    PipelineConfig, PreprocessorConfig, DetectorConfig,
    TrackerConfig, SpeedEstimatorConfig, LoggingConfig, CalibratorConfig
)
from src.models import CalibrationParameters


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_demo_video(output_path: str, num_frames: int = 100):
    """
    Create a demo video with moving objects for testing
    
    Args:
        output_path: Path to save the demo video
        num_frames: Number of frames to generate
    """
    logger.info(f"Creating demo video: {output_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
    
    for i in range(num_frames):
        # Create frame with gradient background
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        
        # Add moving objects
        # Object 1: Moving left to right
        x1 = 100 + i * 8
        if x1 < 1200:
            cv2.rectangle(frame, (x1, 200), (x1 + 60, 260), (255, 100, 100), -1)
        
        # Object 2: Moving right to left
        x2 = 1100 - i * 6
        if x2 > 50:
            cv2.circle(frame, (x2, 400), 30, (100, 255, 100), -1)
        
        # Object 3: Moving diagonally
        x3 = 200 + i * 5
        y3 = 500 + i * 2
        if x3 < 1200 and y3 < 680:
            cv2.rectangle(frame, (x3, y3), (x3 + 50, y3 + 50), (100, 100, 255), -1)
        
        writer.write(frame)
    
    writer.release()
    logger.info(f"Demo video created with {num_frames} frames")


def example_basic_pipeline():
    """Example: Basic pipeline with all components"""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Pipeline with All Components")
    logger.info("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        enabled_components=["preprocessor", "detector", "tracker", "speed_estimator"],
        preprocessor=PreprocessorConfig(
            target_resolution=(1280, 720),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        ),
        detector=DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            edge_detection_enabled=False,
            min_contour_area=500.0,
            max_contour_area=50000.0
        ),
        tracker=TrackerConfig(
            max_tracking_distance=100.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        ),
        calibrator=CalibratorConfig(),
        speed_estimator=SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="m/s"
        ),
        logging=LoggingConfig(level="INFO")
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Create demo video
    demo_video_path = "demo_input.mp4"
    create_demo_video(demo_video_path, num_frames=100)
    
    # Process video
    output_path = "demo_output_basic.mp4"
    logger.info(f"Processing video: {demo_video_path}")
    
    summary = orchestrator.process_video(demo_video_path, output_path)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Total frames: {summary.total_frames}")
    logger.info(f"Processed frames: {summary.processed_frames}")
    logger.info(f"Average FPS: {summary.average_fps:.2f}")
    logger.info(f"Total objects detected: {summary.total_objects_detected}")
    logger.info(f"Unique objects tracked: {summary.unique_objects_tracked}")
    
    if summary.average_speed is not None:
        logger.info(f"Average speed: {summary.average_speed:.2f} px/s")
    if summary.max_speed is not None:
        logger.info(f"Max speed: {summary.max_speed:.2f} px/s")
    
    logger.info("\nComponent Statistics:")
    for component, stats in summary.component_statistics.items():
        logger.info(f"  {component}:")
        logger.info(f"    Average time: {stats.average_time_ms:.2f} ms")
        logger.info(f"    Max time: {stats.max_time_ms:.2f} ms")
    
    if summary.processing_errors:
        logger.warning(f"\nProcessing errors: {len(summary.processing_errors)}")
        for error in summary.processing_errors[:5]:  # Show first 5 errors
            logger.warning(f"  {error}")
    
    logger.info(f"\nOutput video saved to: {output_path}")


def example_with_calibration():
    """Example: Pipeline with camera calibration"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Pipeline with Camera Calibration")
    logger.info("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        enabled_components=["preprocessor", "detector", "tracker", "speed_estimator"],
        preprocessor=PreprocessorConfig(
            target_resolution=(1280, 720),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        ),
        detector=DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            min_contour_area=500.0,
            max_contour_area=50000.0
        ),
        tracker=TrackerConfig(
            max_tracking_distance=100.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        ),
        calibrator=CalibratorConfig(),
        speed_estimator=SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit="km/h"  # Use km/h with calibration
        ),
        logging=LoggingConfig(level="INFO")
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Set calibration parameters (simulated)
    # In real usage, load from calibration file
    orchestrator.calibration = CalibrationParameters(
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        homography_matrix=None,
        pixels_per_meter=50.0,  # 50 pixels = 1 meter
        calibration_error=0.5,
        calibration_date="2024-01-15"
    )
    
    logger.info("Calibration parameters set:")
    logger.info(f"  Pixels per meter: {orchestrator.calibration.pixels_per_meter}")
    logger.info(f"  Output unit: km/h")
    
    # Process video
    demo_video_path = "demo_input.mp4"
    output_path = "demo_output_calibrated.mp4"
    
    summary = orchestrator.process_video(demo_video_path, output_path)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary (with Calibration)")
    logger.info("=" * 60)
    logger.info(f"Processed frames: {summary.processed_frames}")
    logger.info(f"Unique objects tracked: {summary.unique_objects_tracked}")
    
    if summary.average_speed is not None:
        logger.info(f"Average speed: {summary.average_speed:.2f} km/h")
    if summary.max_speed is not None:
        logger.info(f"Max speed: {summary.max_speed:.2f} km/h")
    
    logger.info(f"\nOutput video saved to: {output_path}")


def example_single_frame_processing():
    """Example: Process single frames"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Single Frame Processing")
    logger.info("=" * 60)
    
    # Create configuration
    config = PipelineConfig(
        enabled_components=["preprocessor", "detector", "tracker"],
        preprocessor=PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False
        ),
        detector=DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            min_contour_area=200.0,
            max_contour_area=10000.0
        ),
        tracker=TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=30,
            trajectory_history_length=50
        ),
        calibrator=CalibratorConfig(),
        speed_estimator=SpeedEstimatorConfig(),
        logging=LoggingConfig(level="INFO")
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Process individual frames
    logger.info("Processing 10 frames individually...")
    
    for i in range(10):
        # Create frame with moving object
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        x = 100 + i * 30
        cv2.rectangle(frame, (x, 200), (x + 50, 250), (255, 255, 255), -1)
        
        # Process frame
        result = orchestrator.process_frame(frame, frame_number=i)
        
        # Display results
        logger.info(f"\nFrame {i}:")
        logger.info(f"  Processing time: {result.processing_time_ms:.2f} ms")
        
        if result.detections:
            logger.info(f"  Detections: {len(result.detections.bounding_boxes)}")
        
        if result.tracking:
            logger.info(f"  Tracked objects: {len(result.tracking.tracked_objects)}")
            for obj in result.tracking.tracked_objects:
                logger.info(f"    Object {obj.object_id}: pos={obj.position}, age={obj.age}")


def example_minimal_pipeline():
    """Example: Minimal pipeline with only preprocessing"""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Minimal Pipeline (Preprocessing Only)")
    logger.info("=" * 60)
    
    # Create minimal configuration
    config = PipelineConfig(
        enabled_components=["preprocessor"],
        preprocessor=PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="bilateral",
            noise_reduction_kernel_size=5,
            normalize_intensity=True
        ),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        calibrator=CalibratorConfig(),
        speed_estimator=SpeedEstimatorConfig(),
        logging=LoggingConfig(level="INFO")
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    # Process a single frame
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = orchestrator.process_frame(frame, frame_number=0)
    
    logger.info(f"Processing time: {result.processing_time_ms:.2f} ms")
    logger.info(f"Output shape: {result.annotated_frame.shape}")
    logger.info(f"Components executed: {list(result.component_times.keys())}")


if __name__ == "__main__":
    # Run examples
    example_basic_pipeline()
    example_with_calibration()
    example_single_frame_processing()
    example_minimal_pipeline()
    
    logger.info("\n" + "=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
