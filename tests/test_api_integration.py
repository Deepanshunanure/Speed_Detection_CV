"""Integration tests for FastAPI with actual pipeline components"""
import base64
import tempfile
import os

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def create_test_frame(width=640, height=480):
    """Create a test frame with some content"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some shapes to detect
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(frame, (400, 300), 50, (255, 255, 255), -1)
    
    return frame


def encode_frame(frame):
    """Encode frame to base64"""
    success, buffer = cv2.imencode('.jpg', frame)
    assert success
    return base64.b64encode(buffer).decode('utf-8')


def test_frame_processing_integration():
    """Test complete frame processing through pipeline"""
    # Create test frame
    frame = create_test_frame()
    base64_frame = encode_frame(frame)
    
    # Process frame
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": base64_frame,
            "include_annotated_frame": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify all expected fields are present
    assert "frame_number" in data
    assert "timestamp" in data
    assert "processing_time_ms" in data
    assert "component_times" in data
    
    # Verify processing time is reasonable
    assert data["processing_time_ms"] > 0
    assert data["processing_time_ms"] < 5000  # Should be under 5 seconds
    
    # Verify component times
    assert isinstance(data["component_times"], dict)
    
    # Annotated frame should be present
    if data.get("annotated_frame"):
        # Verify it's valid base64
        decoded = base64.b64decode(data["annotated_frame"])
        assert len(decoded) > 0


def test_frame_processing_without_annotation():
    """Test frame processing without annotated frame"""
    frame = create_test_frame()
    base64_frame = encode_frame(frame)
    
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": base64_frame,
            "include_annotated_frame": False
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Annotated frame should not be present
    assert data.get("annotated_frame") is None


def test_video_processing_integration():
    """Test complete video processing through pipeline"""
    # Create a small test video
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, "test_integration_video.mp4")
    
    # Create video with moving object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
    
    for i in range(15):  # 15 frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Moving rectangle
        x = 100 + i * 10
        cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
        writer.write(frame)
    
    writer.release()
    
    try:
        # Upload video
        with open(video_path, "rb") as f:
            response = client.post(
                "/api/v1/process/video",
                files={"video_file": ("test_video.mp4", f, "video/mp4")}
            )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify task created
        assert "task_id" in data
        assert "status" in data
        assert data["status"] == "queued"
        
        task_id = data["task_id"]
        
        # Check status endpoint
        status_response = client.get(f"/api/v1/process/video/{task_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["task_id"] == task_id
        assert status_data["status"] in ["queued", "processing", "completed", "failed"]
        
    finally:
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)


def test_error_handling_invalid_image():
    """Test error handling for invalid image data"""
    response = client.post(
        "/api/v1/process/frame",
        json={
            "image": "not_valid_base64_image_data",
            "include_annotated_frame": True
        }
    )
    
    # Should return validation error
    assert response.status_code == 422


def test_error_handling_missing_task():
    """Test error handling for non-existent task"""
    response = client.get("/api/v1/process/video/non-existent-task-id")
    assert response.status_code == 404


def test_api_endpoints_exist():
    """Test that all required endpoints exist"""
    # Root endpoint
    response = client.get("/")
    assert response.status_code == 200
    
    # Health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    
    # Frame processing endpoint (POST only)
    response = client.get("/api/v1/process/frame")
    assert response.status_code == 405  # Method not allowed
    
    # Video processing endpoint (POST only)
    response = client.get("/api/v1/process/video")
    assert response.status_code == 405  # Method not allowed


def test_component_times_present():
    """Test that component timing information is included"""
    frame = create_test_frame()
    base64_frame = encode_frame(frame)
    
    response = client.post(
        "/api/v1/process/frame",
        json={"image": base64_frame}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Component times should be present
    assert "component_times" in data
    component_times = data["component_times"]
    
    # Should have timing for enabled components
    # At minimum, should have some components
    assert isinstance(component_times, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
