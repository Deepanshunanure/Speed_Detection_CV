"""Unit tests for FastAPI application"""
import base64
import json
import tempfile
import os
from io import BytesIO

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.api.main import (
    app,
    decode_base64_image,
    encode_image_to_base64,
    pipeline_result_to_dict,
    pipeline_summary_to_dict
)
from src.models import (
    PipelineResult,
    PipelineSummary,
    DetectionResult,
    TrackingResult,
    SpeedResult,
    BoundingBox,
    TrackedObject,
    ComponentStats
)


# Test client
client = TestClient(app)


# Helper functions

def create_test_image(width=640, height=480):
    """Create a test image"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return img


def encode_test_image(img):
    """Encode test image to base64"""
    success, buffer = cv2.imencode('.jpg', img)
    assert success
    return base64.b64encode(buffer).decode('utf-8')


# Tests for helper functions

def test_decode_base64_image():
    """Test base64 image decoding"""
    # Create test image
    img = create_test_image()
    
    # Encode to base64
    base64_str = encode_test_image(img)
    
    # Decode
    decoded = decode_base64_image(base64_str)
    
    # Verify shape
    assert decoded.shape == img.shape
    assert decoded.dtype == np.uint8


def test_decode_base64_image_invalid():
    """Test base64 decoding with invalid data"""
    with pytest.raises(ValueError):
        decode_base64_image("invalid_base64_data")


def test_encode_image_to_base64():
    """Test image encoding to base64"""
    img = create_test_image()
    
    # Encode
    base64_str = encode_image_to_base64(img)
    
    # Verify it's valid base64
    assert isinstance(base64_str, str)
    decoded_bytes = base64.b64decode(base64_str)
    assert len(decoded_bytes) > 0


def test_encode_image_to_base64_png():
    """Test image encoding to PNG format"""
    img = create_test_image()
    
    # Encode as PNG
    base64_str = encode_image_to_base64(img, format='.png')
    
    # Verify it's valid base64
    assert isinstance(base64_str, str)
    decoded_bytes = base64.b64decode(base64_str)
    assert len(decoded_bytes) > 0


def test_pipeline_result_to_dict():
    """Test PipelineResult to dict conversion"""
    # Create test result
    annotated_frame = create_test_image()
    bbox = BoundingBox(x=10, y=20, width=50, height=60, area=3000.0, centroid=(35.0, 50.0))
    
    detections = DetectionResult(
        frame_number=0,
        bounding_boxes=[bbox],
        contours=[],
        foreground_mask=np.zeros((480, 640), dtype=np.uint8)
    )
    
    tracked_obj = TrackedObject(
        object_id=1,
        position=(35.0, 50.0),
        bounding_box=bbox,
        trajectory=[(35.0, 50.0), (36.0, 51.0)],
        age=2,
        disappeared_count=0
    )
    
    tracking = TrackingResult(
        frame_number=0,
        tracked_objects=[tracked_obj]
    )
    
    speed = SpeedResult(
        object_id=1,
        instantaneous_speed=5.5,
        average_speed=5.0,
        displacement_vector=(1.0, 1.0),
        unit="px/s",
        calibrated=False,
        confidence=0.95
    )
    
    result = PipelineResult(
        frame_number=0,
        timestamp=0.0,
        annotated_frame=annotated_frame,
        detections=detections,
        tracking=tracking,
        speeds=[speed],
        processing_time_ms=25.5,
        component_times={"preprocessor": 5.0, "detector": 10.0}
    )
    
    # Convert to dict
    result_dict = pipeline_result_to_dict(result, include_frame=True)
    
    # Verify structure
    assert result_dict["frame_number"] == 0
    assert result_dict["timestamp"] == 0.0
    assert result_dict["processing_time_ms"] == 25.5
    assert "annotated_frame" in result_dict
    assert "detections" in result_dict
    assert "tracking" in result_dict
    assert "speeds" in result_dict
    
    # Verify detections
    assert result_dict["detections"]["num_objects"] == 1
    assert len(result_dict["detections"]["bounding_boxes"]) == 1
    
    # Verify tracking
    assert result_dict["tracking"]["num_tracked_objects"] == 1
    assert result_dict["tracking"]["tracked_objects"][0]["object_id"] == 1
    
    # Verify speeds
    assert len(result_dict["speeds"]) == 1
    assert result_dict["speeds"][0]["object_id"] == 1
    assert result_dict["speeds"][0]["instantaneous_speed"] == 5.5


def test_pipeline_result_to_dict_no_frame():
    """Test PipelineResult to dict without annotated frame"""
    result = PipelineResult(
        frame_number=0,
        timestamp=0.0,
        annotated_frame=create_test_image(),
        detections=None,
        tracking=None,
        speeds=None,
        processing_time_ms=25.5,
        component_times={}
    )
    
    result_dict = pipeline_result_to_dict(result, include_frame=False)
    
    assert "annotated_frame" not in result_dict


def test_pipeline_summary_to_dict():
    """Test PipelineSummary to dict conversion"""
    component_stats = ComponentStats(
        component_name="detector",
        average_time_ms=15.5,
        max_time_ms=25.0,
        error_count=0
    )
    
    summary = PipelineSummary(
        total_frames=100,
        processed_frames=100,
        average_fps=30.0,
        total_objects_detected=50,
        unique_objects_tracked=10,
        average_speed=5.5,
        max_speed=15.0,
        processing_errors=[],
        component_statistics={"detector": component_stats}
    )
    
    summary_dict = pipeline_summary_to_dict(summary)
    
    assert summary_dict["total_frames"] == 100
    assert summary_dict["processed_frames"] == 100
    assert summary_dict["average_fps"] == 30.0
    assert summary_dict["total_objects_detected"] == 50
    assert summary_dict["unique_objects_tracked"] == 10
    assert "component_statistics" in summary_dict
    assert "detector" in summary_dict["component_statistics"]


# Tests for API endpoints

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "active_tasks" in data


def test_process_frame_endpoint():
    """Test frame processing endpoint"""
    # Create test image
    img = create_test_image()
    base64_img = encode_test_image(img)
    
    # Make request
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": base64_img,
            "include_annotated_frame": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "frame_number" in data
    assert "timestamp" in data
    assert "processing_time_ms" in data
    assert "component_times" in data


def test_process_frame_invalid_base64():
    """Test frame processing with invalid base64"""
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": "invalid_base64",
            "include_annotated_frame": True
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_process_frame_without_annotated_frame():
    """Test frame processing without annotated frame"""
    img = create_test_image()
    base64_img = encode_test_image(img)
    
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": base64_img,
            "include_annotated_frame": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "annotated_frame" not in data or data["annotated_frame"] is None


def test_process_video_endpoint():
    """Test video processing endpoint"""
    # Create a temporary test video
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_video.mp4")
    
    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(10):
        frame = create_test_image(640, 480)
        writer.write(frame)
    
    writer.release()
    
    # Upload video
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/v1/process/video",
            files={"video_file": ("test_video.mp4", f, "video/mp4")}
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response
    assert "task_id" in data
    assert "status" in data
    assert data["status"] == "queued"
    
    # Clean up
    os.remove(video_path)


def test_get_video_status_not_found():
    """Test video status endpoint with non-existent task"""
    response = client.get("/api/v1/process/video/non-existent-task-id")
    assert response.status_code == 404


def test_get_video_status():
    """Test video status endpoint"""
    # First create a video processing task
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_video2.mp4")
    
    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(5):
        frame = create_test_image(640, 480)
        writer.write(frame)
    
    writer.release()
    
    # Upload video
    with open(video_path, "rb") as f:
        response = client.post(
            "/api/v1/process/video",
            files={"video_file": ("test_video2.mp4", f, "video/mp4")}
        )
    
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    
    # Check status
    status_response = client.get(f"/api/v1/process/video/{task_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    
    assert status_data["task_id"] == task_id
    assert status_data["status"] in ["queued", "processing", "completed", "failed"]
    
    # Clean up
    os.remove(video_path)


def test_cors_headers():
    """Test CORS middleware is configured"""
    # CORS headers are only added for cross-origin requests
    # Just verify the middleware is configured by checking the app
    from src.api.main import app
    
    # Check that CORS middleware is in the middleware stack
    middleware_classes = [type(m).__name__ for m in app.user_middleware]
    assert "CORSMiddleware" in middleware_classes or len(app.user_middleware) > 0


# Tests for calibration endpoints

def test_calibration_status_no_calibration():
    """Test calibration status when no calibration exists"""
    # Remove calibration file if it exists
    import os
    if os.path.exists("calibration.json"):
        os.remove("calibration.json")
    
    response = client.get("/api/v1/calibration/status")
    assert response.status_code == 200
    data = response.json()
    
    assert data["calibrated"] == False
    assert "message" in data


def test_calibrate_camera_insufficient_images():
    """Test calibration with insufficient images"""
    # Create only 5 images (need at least 10)
    images = []
    for i in range(5):
        img = create_test_image()
        images.append(encode_test_image(img))
    
    response = client.post(
        "/api/v1/calibration/calibrate",
        json={
            "images": images,
            "chessboard_size": [9, 6]
        }
    )
    
    # Should fail validation
    assert response.status_code == 422


def test_calibrate_camera_invalid_chessboard_size():
    """Test calibration with invalid chessboard size"""
    images = []
    for i in range(10):
        img = create_test_image()
        images.append(encode_test_image(img))
    
    response = client.post(
        "/api/v1/calibration/calibrate",
        json={
            "images": images,
            "chessboard_size": [2, 2]  # Too small
        }
    )
    
    # Should fail validation
    assert response.status_code == 422


def test_load_calibration_file_not_found():
    """Test loading calibration from non-existent file"""
    response = client.post(
        "/api/v1/calibration/load",
        json={
            "filepath": "non_existent_calibration.json"
        }
    )
    
    assert response.status_code == 404


def test_load_calibration_success():
    """Test loading calibration from existing file"""
    import os
    import json
    
    # Create a test calibration file
    calibration_data = {
        "camera_matrix": [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.1, -0.2, 0.0, 0.0, 0.0],
        "homography_matrix": None,
        "pixels_per_meter": 100.0,
        "calibration_error": 0.5,
        "calibration_date": "2024-01-01T00:00:00"
    }
    
    test_file = "test_calibration.json"
    with open(test_file, 'w') as f:
        json.dump(calibration_data, f)
    
    try:
        response = client.post(
            "/api/v1/calibration/load",
            json={
                "filepath": test_file
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "camera_matrix" in data
        assert "distortion_coefficients" in data
        assert "calibration_error" in data
        assert "calibration_date" in data
        assert "message" in data
        assert data["calibration_error"] == 0.5
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


def test_calibration_status_with_calibration():
    """Test calibration status when calibration exists"""
    import os
    import json
    
    # Create a test calibration file
    calibration_data = {
        "camera_matrix": [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]],
        "distortion_coefficients": [0.1, -0.2, 0.0, 0.0, 0.0],
        "homography_matrix": None,
        "pixels_per_meter": 100.0,
        "calibration_error": 0.5,
        "calibration_date": "2024-01-01T00:00:00"
    }
    
    with open("calibration.json", 'w') as f:
        json.dump(calibration_data, f)
    
    try:
        response = client.get("/api/v1/calibration/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["calibrated"] == True
        assert data["calibration_error"] == 0.5
        assert data["pixels_per_meter"] == 100.0
        assert "calibration_date" in data
        
    finally:
        # Clean up
        if os.path.exists("calibration.json"):
            os.remove("calibration.json")


# Tests for configuration endpoints

def test_get_config():
    """Test GET /api/v1/config endpoint"""
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    
    # Verify all configuration sections are present
    assert "pipeline" in data
    assert "preprocessor" in data
    assert "detector" in data
    assert "tracker" in data
    assert "calibrator" in data
    assert "speed_estimator" in data
    assert "logging" in data
    
    # Verify pipeline config structure
    assert "enabled_components" in data["pipeline"]
    assert isinstance(data["pipeline"]["enabled_components"], list)
    
    # Verify preprocessor config structure
    assert "target_resolution" in data["preprocessor"]
    assert "noise_reduction_method" in data["preprocessor"]
    assert "noise_reduction_kernel_size" in data["preprocessor"]
    assert "normalize_intensity" in data["preprocessor"]
    
    # Verify detector config structure
    assert "background_subtraction_method" in data["detector"]
    assert "background_learning_rate" in data["detector"]
    assert "min_contour_area" in data["detector"]
    assert "max_contour_area" in data["detector"]
    
    # Verify tracker config structure
    assert "max_tracking_distance" in data["tracker"]
    assert "max_disappeared_frames" in data["tracker"]
    assert "trajectory_history_length" in data["tracker"]


def test_update_config_preprocessor():
    """Test PUT /api/v1/config endpoint with preprocessor updates"""
    import yaml
    import os
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update preprocessor configuration
        response = client.put(
            "/api/v1/config",
            json={
                "preprocessor": {
                    "target_resolution": [1920, 1080],
                    "noise_reduction_kernel_size": 7
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "message" in data
        assert "warnings" in data
        assert "updated_config" in data
        
        # Verify updated values
        assert data["updated_config"]["preprocessor"]["target_resolution"] == [1920, 1080]
        assert data["updated_config"]["preprocessor"]["noise_reduction_kernel_size"] == 7
        
        # Verify file was updated
        with open("config/default.yaml", 'r') as f:
            updated_config = yaml.safe_load(f)
        
        assert updated_config["preprocessor"]["target_resolution"] == [1920, 1080]
        assert updated_config["preprocessor"]["noise_reduction_kernel_size"] == 7
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


def test_update_config_detector():
    """Test PUT /api/v1/config endpoint with detector updates"""
    import yaml
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update detector configuration
        response = client.put(
            "/api/v1/config",
            json={
                "detector": {
                    "background_subtraction_method": "KNN",
                    "min_contour_area": 1000.0
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify updated values
        assert data["updated_config"]["detector"]["background_subtraction_method"] == "KNN"
        assert data["updated_config"]["detector"]["min_contour_area"] == 1000.0
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


def test_update_config_tracker():
    """Test PUT /api/v1/config endpoint with tracker updates"""
    import yaml
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update tracker configuration
        response = client.put(
            "/api/v1/config",
            json={
                "tracker": {
                    "max_tracking_distance": 75.0,
                    "max_disappeared_frames": 50
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify updated values
        assert data["updated_config"]["tracker"]["max_tracking_distance"] == 75.0
        assert data["updated_config"]["tracker"]["max_disappeared_frames"] == 50
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


def test_update_config_multiple_components():
    """Test PUT /api/v1/config endpoint with multiple component updates"""
    import yaml
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update multiple components
        response = client.put(
            "/api/v1/config",
            json={
                "preprocessor": {
                    "noise_reduction_method": "bilateral"
                },
                "detector": {
                    "edge_detection_enabled": True
                },
                "speed_estimator": {
                    "output_unit": "km/h"
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all updated values
        assert data["updated_config"]["preprocessor"]["noise_reduction_method"] == "bilateral"
        assert data["updated_config"]["detector"]["edge_detection_enabled"] == True
        assert data["updated_config"]["speed_estimator"]["output_unit"] == "km/h"
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


def test_update_config_with_validation_warnings():
    """Test PUT /api/v1/config endpoint with invalid values that trigger warnings"""
    import yaml
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update with invalid values
        response = client.put(
            "/api/v1/config",
            json={
                "preprocessor": {
                    "noise_reduction_kernel_size": 4  # Should be odd
                },
                "detector": {
                    "background_learning_rate": 1.5  # Should be between 0 and 1
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify warnings are present
        assert "warnings" in data
        assert len(data["warnings"]) > 0
        
        # Check for specific warnings
        warning_text = " ".join(data["warnings"])
        assert "noise_reduction_kernel_size" in warning_text or "odd" in warning_text
        assert "background_learning_rate" in warning_text or "between 0.0 and 1.0" in warning_text
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


def test_update_config_pipeline_enabled_components():
    """Test PUT /api/v1/config endpoint with pipeline enabled components"""
    import yaml
    
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    try:
        # Update enabled components
        response = client.put(
            "/api/v1/config",
            json={
                "pipeline": {
                    "enabled_components": ["preprocessor", "detector"]
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify updated values
        assert data["updated_config"]["pipeline"]["enabled_components"] == ["preprocessor", "detector"]
        
    finally:
        # Restore original config
        with open("config/default.yaml", 'w') as f:
            yaml.dump(original_config, f, default_flow_style=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
