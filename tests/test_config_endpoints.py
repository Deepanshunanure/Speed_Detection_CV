"""Integration tests for configuration endpoints"""
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


@pytest.fixture
def backup_config():
    """Backup and restore configuration file"""
    # Backup original config
    with open("config/default.yaml", 'r') as f:
        original_config = yaml.safe_load(f)
    
    yield
    
    # Restore original config
    with open("config/default.yaml", 'w') as f:
        yaml.dump(original_config, f, default_flow_style=False)


def test_config_endpoints_in_root(backup_config):
    """Test that configuration endpoints are listed in root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    
    assert "endpoints" in data
    assert "config_get" in data["endpoints"]
    assert "config_update" in data["endpoints"]
    assert data["endpoints"]["config_get"] == "/api/v1/config"
    assert data["endpoints"]["config_update"] == "/api/v1/config"


def test_get_config_complete_structure(backup_config):
    """Test GET /api/v1/config returns complete configuration structure"""
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    
    # Verify all top-level sections
    required_sections = ["pipeline", "preprocessor", "detector", "tracker", 
                        "calibrator", "speed_estimator", "logging"]
    for section in required_sections:
        assert section in data, f"Missing section: {section}"
    
    # Verify pipeline section
    assert "enabled_components" in data["pipeline"]
    assert isinstance(data["pipeline"]["enabled_components"], list)
    
    # Verify preprocessor section
    preprocessor_keys = ["target_resolution", "noise_reduction_method", 
                         "noise_reduction_kernel_size", "normalize_intensity"]
    for key in preprocessor_keys:
        assert key in data["preprocessor"], f"Missing preprocessor key: {key}"
    
    # Verify detector section
    detector_keys = ["background_subtraction_method", "background_learning_rate",
                    "edge_detection_enabled", "canny_threshold1", "canny_threshold2",
                    "min_contour_area", "max_contour_area"]
    for key in detector_keys:
        assert key in data["detector"], f"Missing detector key: {key}"
    
    # Verify tracker section
    tracker_keys = ["max_tracking_distance", "max_disappeared_frames", 
                   "trajectory_history_length"]
    for key in tracker_keys:
        assert key in data["tracker"], f"Missing tracker key: {key}"
    
    # Verify calibrator section
    calibrator_keys = ["calibration_file_path", "chessboard_size", 
                      "square_size_mm", "perspective_transform_enabled"]
    for key in calibrator_keys:
        assert key in data["calibrator"], f"Missing calibrator key: {key}"
    
    # Verify speed_estimator section
    speed_keys = ["averaging_window_frames", "min_trajectory_length", "output_unit"]
    for key in speed_keys:
        assert key in data["speed_estimator"], f"Missing speed_estimator key: {key}"
    
    # Verify logging section
    logging_keys = ["level", "file_path", "log_to_file"]
    for key in logging_keys:
        assert key in data["logging"], f"Missing logging key: {key}"


def test_update_config_partial_update(backup_config):
    """Test PUT /api/v1/config with partial updates"""
    # Update only one field in preprocessor
    response = client.put(
        "/api/v1/config",
        json={
            "preprocessor": {
                "noise_reduction_kernel_size": 9
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "message" in data
    assert "warnings" in data
    assert "updated_config" in data
    
    # Verify only the specified field was updated
    assert data["updated_config"]["preprocessor"]["noise_reduction_kernel_size"] == 9
    
    # Verify other fields remain unchanged
    assert "target_resolution" in data["updated_config"]["preprocessor"]
    assert "noise_reduction_method" in data["updated_config"]["preprocessor"]


def test_update_config_persistence(backup_config):
    """Test that configuration updates persist to file"""
    # Update configuration
    response = client.put(
        "/api/v1/config",
        json={
            "detector": {
                "min_contour_area": 2000.0
            }
        }
    )
    
    assert response.status_code == 200
    
    # Read file directly
    with open("config/default.yaml", 'r') as f:
        file_config = yaml.safe_load(f)
    
    # Verify update persisted
    assert file_config["detector"]["min_contour_area"] == 2000.0
    
    # Verify GET endpoint returns updated value
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert data["detector"]["min_contour_area"] == 2000.0


def test_update_config_validation_warnings(backup_config):
    """Test that validation warnings are returned for invalid values"""
    # Update with multiple invalid values
    response = client.put(
        "/api/v1/config",
        json={
            "preprocessor": {
                "noise_reduction_kernel_size": 2,  # Too small and even
                "target_resolution": [50, 50]  # Too small
            },
            "detector": {
                "background_learning_rate": 2.0,  # Out of range
                "min_contour_area": -10.0  # Negative
            }
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify warnings are present
    assert "warnings" in data
    assert len(data["warnings"]) > 0
    
    # Check for specific warnings
    warnings_text = " ".join(data["warnings"]).lower()
    assert any(keyword in warnings_text for keyword in ["kernel", "resolution", "learning", "contour"])


def test_update_config_empty_request(backup_config):
    """Test PUT /api/v1/config with empty request body"""
    response = client.put(
        "/api/v1/config",
        json={}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should succeed but not change anything
    assert "message" in data
    assert "updated_config" in data


def test_update_config_all_components(backup_config):
    """Test updating all components at once"""
    update_data = {
        "pipeline": {
            "enabled_components": ["preprocessor", "detector"]
        },
        "preprocessor": {
            "target_resolution": [800, 600],
            "noise_reduction_method": "median"
        },
        "detector": {
            "background_subtraction_method": "GMG",
            "min_contour_area": 300.0
        },
        "tracker": {
            "max_tracking_distance": 100.0
        },
        "calibrator": {
            "square_size_mm": 30.0
        },
        "speed_estimator": {
            "output_unit": "mph"
        },
        "logging": {
            "level": "DEBUG"
        }
    }
    
    response = client.put(
        "/api/v1/config",
        json=update_data
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify all updates
    config = data["updated_config"]
    assert config["pipeline"]["enabled_components"] == ["preprocessor", "detector"]
    assert config["preprocessor"]["target_resolution"] == [800, 600]
    assert config["preprocessor"]["noise_reduction_method"] == "median"
    assert config["detector"]["background_subtraction_method"] == "GMG"
    assert config["detector"]["min_contour_area"] == 300.0
    assert config["tracker"]["max_tracking_distance"] == 100.0
    assert config["calibrator"]["square_size_mm"] == 30.0
    assert config["speed_estimator"]["output_unit"] == "mph"
    assert config["logging"]["level"] == "DEBUG"


def test_config_roundtrip(backup_config):
    """Test GET -> PUT -> GET roundtrip"""
    # Get initial config
    response1 = client.get("/api/v1/config")
    assert response1.status_code == 200
    initial_config = response1.json()
    
    # Update some values
    response2 = client.put(
        "/api/v1/config",
        json={
            "preprocessor": {
                "noise_reduction_kernel_size": 11
            },
            "tracker": {
                "max_disappeared_frames": 60
            }
        }
    )
    assert response2.status_code == 200
    
    # Get updated config
    response3 = client.get("/api/v1/config")
    assert response3.status_code == 200
    updated_config = response3.json()
    
    # Verify changes
    assert updated_config["preprocessor"]["noise_reduction_kernel_size"] == 11
    assert updated_config["tracker"]["max_disappeared_frames"] == 60
    
    # Verify other values unchanged
    assert updated_config["preprocessor"]["noise_reduction_method"] == \
           initial_config["preprocessor"]["noise_reduction_method"]
    assert updated_config["detector"]["background_subtraction_method"] == \
           initial_config["detector"]["background_subtraction_method"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
