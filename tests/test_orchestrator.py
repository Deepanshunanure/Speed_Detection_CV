"""Unit tests for PipelineOrchestrator"""
import pytest
import numpy as np
import tempfile
import os
import cv2
from unittest.mock import Mock, patch, MagicMock

from src.api.orchestrator import PipelineOrchestrator
from src.config.models import (
    PipelineConfig, PreprocessorConfig, DetectorConfig,
    TrackerConfig, SpeedEstimatorConfig, LoggingConfig
)
from src.models import (
    DetectionResult, TrackingResult, TrackedObject,
    BoundingBox, SpeedResult, PipelineSummary
)


@pytest.fixture
def minimal_config():
    """Create minimal pipeline configuration"""
    return PipelineConfig(
        enabled_components=["preprocessor"],
        preprocessor=PreprocessorConfig(),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        calibrator=None,
        speed_estimator=SpeedEstimatorConfig(),
        logging=LoggingConfig()
    )


@pytest.fixture
def full_config():
    """Create full pipeline configuration with all components"""
    return PipelineConfig(
        enabled_components=["preprocessor", "detector", "tracker", "speed_estimator"],
        preprocessor=PreprocessorConfig(),
        detector=DetectorConfig(),
        tracker=TrackerConfig(),
        calibrator=None,
        speed_estimator=SpeedEstimatorConfig(),
        logging=LoggingConfig()
    )


@pytest.fixture
def sample_frame():
    """Create a sample video frame"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Create sample detection result"""
    bbox = BoundingBox(
        x=100, y=100, width=50, height=50,
        area=2500.0, centroid=(125.0, 125.0)
    )
    return DetectionResult(
        frame_number=0,
        bounding_boxes=[bbox],
        contours=[np.array([[100, 100], [150, 100], [150, 150], [100, 150]])],
        foreground_mask=np.zeros((480, 640), dtype=np.uint8)
    )


@pytest.fixture
def sample_tracking():
    """Create sample tracking result"""
    bbox = BoundingBox(
        x=100, y=100, width=50, height=50,
        area=2500.0, centroid=(125.0, 125.0)
    )
    tracked_obj = TrackedObject(
        object_id=1,
        position=(125.0, 125.0),
        bounding_box=bbox,
        trajectory=[(120.0, 120.0), (125.0, 125.0)],
        age=2,
        disappeared_count=0
    )
    return TrackingResult(
        frame_number=0,
        tracked_objects=[tracked_obj]
    )


class TestPipelineOrchestratorInitialization:
    """Test orchestrator initialization"""
    
    def test_init_with_minimal_components(self, minimal_config):
        """Test initialization with only preprocessor enabled"""
        orchestrator = PipelineOrchestrator(minimal_config)
        
        assert orchestrator.preprocessor is not None
        assert orchestrator.detector is None
        assert orchestrator.tracker is None
        assert orchestrator.speed_estimator is None
    
    def test_init_with_all_components(self, full_config):
        """Test initialization with all components enabled"""
        orchestrator = PipelineOrchestrator(full_config)
        
        assert orchestrator.preprocessor is not None
        assert orchestrator.detector is not None
        assert orchestrator.tracker is not None
        assert orchestrator.speed_estimator is not None
    
    def test_init_with_no_components(self):
        """Test initialization with no components enabled"""
        config = PipelineConfig(
            enabled_components=[],
            preprocessor=PreprocessorConfig(),
            detector=DetectorConfig(),
            tracker=TrackerConfig(),
            calibrator=None,
            speed_estimator=SpeedEstimatorConfig(),
            logging=LoggingConfig()
        )
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.preprocessor is None
        assert orchestrator.detector is None
        assert orchestrator.tracker is None
        assert orchestrator.speed_estimator is None


class TestProcessFrame:
    """Test single frame processing"""
    
    def test_process_frame_with_preprocessor_only(self, minimal_config, sample_frame):
        """Test frame processing with only preprocessor"""
        orchestrator = PipelineOrchestrator(minimal_config)
        result = orchestrator.process_frame(sample_frame, frame_number=0)
        
        assert result.frame_number == 0
        assert result.annotated_frame is not None
        assert result.annotated_frame.shape == sample_frame.shape
        assert result.processing_time_ms > 0
        assert "preprocessor" in result.component_times
        assert result.detections is None
        assert result.tracking is None
        assert result.speeds is None
    
    def test_process_frame_with_all_components(self, full_config, sample_frame):
        """Test frame processing with all components"""
        orchestrator = PipelineOrchestrator(full_config)
        result = orchestrator.process_frame(sample_frame, frame_number=5)
        
        assert result.frame_number == 5
        assert result.annotated_frame is not None
        assert result.processing_time_ms > 0
        assert "preprocessor" in result.component_times
        assert "detector" in result.component_times
        # Tracker and speed estimator may not have timing if no detections
    
    def test_process_frame_component_timing(self, full_config, sample_frame):
        """Test that component timing is measured"""
        orchestrator = PipelineOrchestrator(full_config)
        result = orchestrator.process_frame(sample_frame, frame_number=0)
        
        # Check that timing is positive for executed components
        for component, time_ms in result.component_times.items():
            assert time_ms >= 0
    
    def test_process_frame_graceful_degradation(self, full_config, sample_frame):
        """Test that processing continues when a component fails"""
        orchestrator = PipelineOrchestrator(full_config)
        
        # Mock preprocessor to raise an exception
        orchestrator.preprocessor.process = Mock(side_effect=Exception("Test error"))
        
        # Should not raise, should continue with other components
        result = orchestrator.process_frame(sample_frame, frame_number=0)
        
        assert result is not None
        assert result.annotated_frame is not None


class TestAnnotateFrame:
    """Test frame annotation"""
    
    def test_annotate_with_tracking_and_speeds(
        self, full_config, sample_frame, sample_tracking
    ):
        """Test annotation with tracking and speed results"""
        orchestrator = PipelineOrchestrator(full_config)
        
        speeds = [SpeedResult(
            object_id=1,
            instantaneous_speed=5.5,
            average_speed=5.0,
            displacement_vector=(1.0, 2.0),
            unit="m/s",
            calibrated=True,
            confidence=1.0
        )]
        
        annotated = orchestrator._annotate_frame(
            sample_frame, None, sample_tracking, speeds
        )
        
        assert annotated.shape == sample_frame.shape
        # Frame should be modified (not identical to input)
        assert not np.array_equal(annotated, sample_frame)
    
    def test_annotate_with_detections_only(
        self, full_config, sample_frame, sample_detection
    ):
        """Test annotation with only detection results"""
        orchestrator = PipelineOrchestrator(full_config)
        
        annotated = orchestrator._annotate_frame(
            sample_frame, sample_detection, None, None
        )
        
        assert annotated.shape == sample_frame.shape
        assert not np.array_equal(annotated, sample_frame)
    
    def test_annotate_with_no_results(self, full_config, sample_frame):
        """Test annotation with no detection or tracking results"""
        orchestrator = PipelineOrchestrator(full_config)
        
        annotated = orchestrator._annotate_frame(
            sample_frame, None, None, None
        )
        
        assert annotated.shape == sample_frame.shape
        # Should return a copy even with no annotations
        assert annotated is not sample_frame


class TestProcessVideo:
    """Test video file processing"""
    
    def test_process_video_basic(self, minimal_config):
        """Test basic video processing"""
        # Create a temporary test video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        try:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            
            # Write 10 frames
            for i in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            
            # Process the video
            orchestrator = PipelineOrchestrator(minimal_config)
            summary = orchestrator.process_video(video_path)
            
            assert isinstance(summary, PipelineSummary)
            assert summary.total_frames == 10
            assert summary.processed_frames == 10
            assert summary.average_fps > 0
            assert len(summary.processing_errors) == 0
        
        finally:
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
    
    def test_process_video_with_output(self, minimal_config):
        """Test video processing with output file"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_in:
            video_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        try:
            # Create test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            for i in range(5):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            
            # Process with output
            orchestrator = PipelineOrchestrator(minimal_config)
            summary = orchestrator.process_video(video_path, output_path)
            
            assert summary.processed_frames == 5
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        
        finally:
            # Clean up
            for path in [video_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
    
    def test_process_video_invalid_path(self, minimal_config):
        """Test processing with invalid video path"""
        orchestrator = PipelineOrchestrator(minimal_config)
        
        with pytest.raises(ValueError, match="Failed to open video file"):
            orchestrator.process_video("nonexistent_video.mp4")
    
    def test_process_video_summary_statistics(self, full_config):
        """Test that summary contains correct statistics"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        try:
            # Create test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            for i in range(15):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            
            # Process video
            orchestrator = PipelineOrchestrator(full_config)
            summary = orchestrator.process_video(video_path)
            
            assert summary.total_frames == 15
            assert summary.processed_frames == 15
            assert summary.average_fps > 0
            assert "preprocessor" in summary.component_statistics
            assert summary.component_statistics["preprocessor"].average_time_ms > 0
        
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)


class TestErrorHandling:
    """Test error handling and graceful degradation"""
    
    def test_component_failure_isolation(self, full_config, sample_frame):
        """Test that component failure doesn't stop pipeline"""
        orchestrator = PipelineOrchestrator(full_config)
        
        # Make detector fail
        orchestrator.detector.detect = Mock(side_effect=Exception("Detector error"))
        
        # Should still complete processing
        result = orchestrator.process_frame(sample_frame, frame_number=0)
        
        assert result is not None
        assert result.detections is None
        # Preprocessor should still have run
        assert "preprocessor" in result.component_times
    
    def test_video_processing_with_frame_errors(self, minimal_config):
        """Test video processing continues despite frame errors"""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        try:
            # Create test video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            for i in range(10):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                writer.write(frame)
            writer.release()
            
            orchestrator = PipelineOrchestrator(minimal_config)
            
            # Mock process_frame to fail on some frames
            original_process = orchestrator.process_frame
            call_count = [0]
            
            def mock_process(frame, frame_number):
                call_count[0] += 1
                if call_count[0] == 3:  # Fail on 3rd frame
                    raise Exception("Test error")
                return original_process(frame, frame_number)
            
            orchestrator.process_frame = mock_process
            
            summary = orchestrator.process_video(video_path)
            
            # Should process 9 frames successfully (1 failed)
            assert summary.processed_frames == 9
            assert len(summary.processing_errors) == 1
        
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)
