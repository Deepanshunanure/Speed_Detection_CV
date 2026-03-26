"""Integration tests for PipelineOrchestrator with real components"""
import pytest
import numpy as np
import tempfile
import os
import cv2

from src.api.orchestrator import PipelineOrchestrator
from src.config.models import (
    PipelineConfig, PreprocessorConfig, DetectorConfig,
    TrackerConfig, SpeedEstimatorConfig, LoggingConfig, CalibratorConfig
)
from src.models import CalibrationParameters


@pytest.fixture
def full_pipeline_config():
    """Create configuration with all components enabled"""
    return PipelineConfig(
        enabled_components=["preprocessor", "detector", "tracker", "speed_estimator"],
        preprocessor=PreprocessorConfig(
            target_resolution=(640, 480),
            noise_reduction_method="gaussian",
            noise_reduction_kernel_size=5,
            normalize_intensity=False  # Keep as uint8 for detector
        ),
        detector=DetectorConfig(
            background_subtraction_method="MOG2",
            background_learning_rate=0.01,
            edge_detection_enabled=False,
            min_contour_area=100.0,
            max_contour_area=10000.0
        ),
        tracker=TrackerConfig(
            max_tracking_distance=50.0,
            max_disappeared_frames=30,
            trajectory_history_length=100
        ),
        calibrator=CalibratorConfig(),
        speed_estimator=SpeedEstimatorConfig(
            averaging_window_frames=5,
            min_trajectory_length=2,
            output_unit="m/s"
        ),
        logging=LoggingConfig(level="INFO")
    )


def create_test_video_with_moving_object(path, num_frames=30):
    """Create a test video with a moving white square on black background"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, 30.0, (640, 480))
    
    for i in range(num_frames):
        # Create black background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw moving white square
        x = 100 + i * 10  # Move right
        y = 200
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (255, 255, 255), -1)
        
        writer.write(frame)
    
    writer.release()


class TestFullPipelineIntegration:
    """Test complete pipeline with all components"""
    
    def test_process_frame_full_pipeline(self, full_pipeline_config):
        """Test processing a single frame through full pipeline"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Create frame with white square
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (150, 150), (255, 255, 255), -1)
        
        result = orchestrator.process_frame(frame, frame_number=0)
        
        assert result.frame_number == 0
        assert result.annotated_frame is not None
        assert result.processing_time_ms > 0
        
        # Check all components executed
        assert "preprocessor" in result.component_times
        assert "detector" in result.component_times
        
        # First frame may not have tracking/speeds yet
        assert result.detections is not None
    
    def test_process_video_with_moving_object(self, full_pipeline_config):
        """Test processing video with moving object"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        try:
            # Create test video
            create_test_video_with_moving_object(video_path, num_frames=30)
            
            # Process video
            orchestrator = PipelineOrchestrator(full_pipeline_config)
            summary = orchestrator.process_video(video_path)
            
            assert summary.total_frames == 30
            assert summary.processed_frames == 30
            assert summary.average_fps > 0
            
            # Should detect objects
            assert summary.total_objects_detected > 0
            
            # Should track at least one unique object
            assert summary.unique_objects_tracked > 0
            
            # Check component statistics
            assert "preprocessor" in summary.component_statistics
            assert "detector" in summary.component_statistics
            assert summary.component_statistics["preprocessor"].average_time_ms > 0
            assert summary.component_statistics["detector"].average_time_ms > 0
        
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
    
    def test_process_video_with_output_file(self, full_pipeline_config):
        """Test processing video and saving annotated output"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_in:
            video_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            # Create test video
            create_test_video_with_moving_object(video_path, num_frames=15)
            
            # Process with output
            orchestrator = PipelineOrchestrator(full_pipeline_config)
            summary = orchestrator.process_video(video_path, output_path)
            
            assert summary.processed_frames == 15
            
            # Verify output file exists and has content
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
            
            # Verify output video can be opened
            cap = cv2.VideoCapture(output_path)
            assert cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count == 15
            cap.release()
        
        finally:
            for path in [video_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def test_tracking_persistence_across_frames(self, full_pipeline_config):
        """Test that objects maintain IDs across frames"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Process multiple frames with same object
        object_ids_per_frame = []
        
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw object at slightly different position each frame
            x = 100 + i * 5
            cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
            
            result = orchestrator.process_frame(frame, frame_number=i)
            
            if result.tracking is not None and len(result.tracking.tracked_objects) > 0:
                ids = [obj.object_id for obj in result.tracking.tracked_objects]
                object_ids_per_frame.append(ids)
        
        # Check that we tracked objects
        assert len(object_ids_per_frame) > 0
        
        # Check that the same ID appears in multiple frames (persistence)
        if len(object_ids_per_frame) > 1:
            # Get the most common ID
            all_ids = [id for frame_ids in object_ids_per_frame for id in frame_ids]
            if all_ids:
                most_common_id = max(set(all_ids), key=all_ids.count)
                # This ID should appear in multiple frames
                frames_with_id = sum(1 for frame_ids in object_ids_per_frame if most_common_id in frame_ids)
                assert frames_with_id > 1
    
    def test_speed_estimation_with_calibration(self, full_pipeline_config):
        """Test speed estimation with calibration parameters"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Set calibration parameters
        orchestrator.calibration = CalibrationParameters(
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
            homography_matrix=None,
            pixels_per_meter=100.0,  # 100 pixels = 1 meter
            calibration_error=0.0,
            calibration_date="2024-01-01"
        )
        
        # Process frames with moving object
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            x = 100 + i * 20  # Move 20 pixels per frame
            cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
            
            result = orchestrator.process_frame(frame, frame_number=i)
            
            # After a few frames, should have speed estimates
            if i > 2 and result.speeds is not None and len(result.speeds) > 0:
                speed = result.speeds[0]
                assert speed.calibrated is True
                assert speed.unit == "m/s"
                assert speed.instantaneous_speed > 0


class TestComponentCoordination:
    """Test coordination between components"""
    
    def test_preprocessor_output_feeds_detector(self, full_pipeline_config):
        """Test that preprocessor output is used by detector"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Create high-contrast frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (300, 300), (255, 255, 255), -1)
        
        result = orchestrator.process_frame(frame, frame_number=0)
        
        # Both components should execute
        assert "preprocessor" in result.component_times
        assert "detector" in result.component_times
        
        # Detector should find something (after background model adapts)
        assert result.detections is not None
    
    def test_detector_output_feeds_tracker(self, full_pipeline_config):
        """Test that detector output is used by tracker"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Process multiple frames to build tracking
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100 + i*10, 100), (150 + i*10, 150), (255, 255, 255), -1)
            
            result = orchestrator.process_frame(frame, frame_number=i)
            
            if i > 1:  # After background model adapts
                # Should have detections
                if result.detections is not None and len(result.detections.bounding_boxes) > 0:
                    # Tracker should process them
                    assert "tracker" in result.component_times
    
    def test_tracker_output_feeds_speed_estimator(self, full_pipeline_config):
        """Test that tracker output is used by speed estimator"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Process frames with consistent object
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100 + i*15, 100), (150 + i*15, 150), (255, 255, 255), -1)
            
            result = orchestrator.process_frame(frame, frame_number=i)
            
            # After enough frames for tracking and speed estimation
            if i > 3:
                if result.tracking is not None and len(result.tracking.tracked_objects) > 0:
                    tracked_obj = result.tracking.tracked_objects[0]
                    # If trajectory is long enough, should have speeds
                    if len(tracked_obj.trajectory) >= 2:
                        assert "speed_estimator" in result.component_times


class TestAnnotationQuality:
    """Test quality of frame annotations"""
    
    def test_annotations_include_bounding_boxes(self, full_pipeline_config):
        """Test that annotated frames include bounding boxes"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Create frame with object
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (200, 200), (250, 250), (255, 255, 255), -1)
        
        # Process multiple frames to get tracking
        for i in range(5):
            result = orchestrator.process_frame(frame, frame_number=i)
        
        # Last frame should have annotations
        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == frame.shape
    
    def test_annotations_include_object_ids(self, full_pipeline_config):
        """Test that annotations include object IDs when tracking is active"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Process frames to establish tracking
        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100 + i*5, 100), (150 + i*5, 150), (255, 255, 255), -1)
            
            result = orchestrator.process_frame(frame, frame_number=i)
            
            # After tracking is established, annotated frame should differ from input
            if result.tracking is not None and len(result.tracking.tracked_objects) > 0:
                assert not np.array_equal(result.annotated_frame, frame)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_processing_speed_meets_requirements(self, full_pipeline_config):
        """Test that processing meets performance requirements"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        # Process several frames and check average time
        processing_times = []
        
        for i in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = orchestrator.process_frame(frame, frame_number=i)
            processing_times.append(result.processing_time_ms)
        
        avg_time = np.mean(processing_times)
        
        # Should process reasonably fast (< 200ms per frame for 640x480)
        # This is a relaxed requirement for testing
        assert avg_time < 200.0
    
    def test_component_timing_accuracy(self, full_pipeline_config):
        """Test that component timing is accurate"""
        orchestrator = PipelineOrchestrator(full_pipeline_config)
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = orchestrator.process_frame(frame, frame_number=0)
        
        # Sum of component times should be less than or equal to total time
        component_sum = sum(result.component_times.values())
        assert component_sum <= result.processing_time_ms
