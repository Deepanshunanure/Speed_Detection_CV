"""
Example demonstrating configuration API endpoints

This example shows how to:
1. Retrieve current pipeline configuration
2. Update configuration parameters
3. Handle validation warnings
"""

import requests
import json


def main():
    # API base URL (adjust if running on different host/port)
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Configuration API Example")
    print("=" * 60)
    
    # 1. Get current configuration
    print("\n1. Retrieving current configuration...")
    response = requests.get(f"{base_url}/api/v1/config")
    
    if response.status_code == 200:
        config = response.json()
        print("✓ Configuration retrieved successfully")
        print(f"\nCurrent preprocessor settings:")
        print(f"  - Target resolution: {config['preprocessor']['target_resolution']}")
        print(f"  - Noise reduction method: {config['preprocessor']['noise_reduction_method']}")
        print(f"  - Kernel size: {config['preprocessor']['noise_reduction_kernel_size']}")
        
        print(f"\nCurrent detector settings:")
        print(f"  - Background subtraction: {config['detector']['background_subtraction_method']}")
        print(f"  - Learning rate: {config['detector']['background_learning_rate']}")
        print(f"  - Min contour area: {config['detector']['min_contour_area']}")
        
        print(f"\nCurrent tracker settings:")
        print(f"  - Max tracking distance: {config['tracker']['max_tracking_distance']}")
        print(f"  - Max disappeared frames: {config['tracker']['max_disappeared_frames']}")
        print(f"  - Trajectory history: {config['tracker']['trajectory_history_length']}")
    else:
        print(f"✗ Failed to retrieve configuration: {response.status_code}")
        print(response.text)
        return
    
    # 2. Update preprocessor configuration
    print("\n2. Updating preprocessor configuration...")
    update_data = {
        "preprocessor": {
            "target_resolution": [1920, 1080],
            "noise_reduction_method": "bilateral",
            "noise_reduction_kernel_size": 7
        }
    }
    
    response = requests.put(
        f"{base_url}/api/v1/config",
        json=update_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Configuration updated successfully")
        print(f"  Message: {result['message']}")
        
        if result['warnings']:
            print(f"\n  Validation warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")
        else:
            print("  No validation warnings")
        
        print(f"\nUpdated preprocessor settings:")
        updated_config = result['updated_config']
        print(f"  - Target resolution: {updated_config['preprocessor']['target_resolution']}")
        print(f"  - Noise reduction method: {updated_config['preprocessor']['noise_reduction_method']}")
        print(f"  - Kernel size: {updated_config['preprocessor']['noise_reduction_kernel_size']}")
    else:
        print(f"✗ Failed to update configuration: {response.status_code}")
        print(response.text)
        return
    
    # 3. Update multiple components
    print("\n3. Updating multiple components...")
    update_data = {
        "detector": {
            "background_subtraction_method": "KNN",
            "min_contour_area": 1000.0
        },
        "tracker": {
            "max_tracking_distance": 75.0,
            "max_disappeared_frames": 50
        },
        "speed_estimator": {
            "output_unit": "km/h",
            "averaging_window_frames": 15
        }
    }
    
    response = requests.put(
        f"{base_url}/api/v1/config",
        json=update_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Multiple components updated successfully")
        
        updated_config = result['updated_config']
        print(f"\nUpdated detector settings:")
        print(f"  - Background subtraction: {updated_config['detector']['background_subtraction_method']}")
        print(f"  - Min contour area: {updated_config['detector']['min_contour_area']}")
        
        print(f"\nUpdated tracker settings:")
        print(f"  - Max tracking distance: {updated_config['tracker']['max_tracking_distance']}")
        print(f"  - Max disappeared frames: {updated_config['tracker']['max_disappeared_frames']}")
        
        print(f"\nUpdated speed estimator settings:")
        print(f"  - Output unit: {updated_config['speed_estimator']['output_unit']}")
        print(f"  - Averaging window: {updated_config['speed_estimator']['averaging_window_frames']}")
    else:
        print(f"✗ Failed to update configuration: {response.status_code}")
        print(response.text)
        return
    
    # 4. Update with invalid values to demonstrate validation
    print("\n4. Testing validation with invalid values...")
    update_data = {
        "preprocessor": {
            "noise_reduction_kernel_size": 4  # Should be odd
        },
        "detector": {
            "background_learning_rate": 1.5  # Should be between 0 and 1
        }
    }
    
    response = requests.put(
        f"{base_url}/api/v1/config",
        json=update_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Configuration updated (with warnings)")
        
        if result['warnings']:
            print(f"\n  Validation warnings detected:")
            for warning in result['warnings']:
                print(f"    - {warning}")
        else:
            print("  No validation warnings (unexpected)")
    else:
        print(f"✗ Failed to update configuration: {response.status_code}")
        print(response.text)
    
    # 5. Update pipeline enabled components
    print("\n5. Updating pipeline enabled components...")
    update_data = {
        "pipeline": {
            "enabled_components": ["preprocessor", "detector", "tracker"]
        }
    }
    
    response = requests.put(
        f"{base_url}/api/v1/config",
        json=update_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✓ Pipeline configuration updated")
        
        updated_config = result['updated_config']
        print(f"\nEnabled components: {updated_config['pipeline']['enabled_components']}")
    else:
        print(f"✗ Failed to update configuration: {response.status_code}")
        print(response.text)
    
    print("\n" + "=" * 60)
    print("Configuration API example completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\nNOTE: This example requires the API server to be running.")
    print("Start the server with: uvicorn src.api.main:app --reload")
    print("Press Ctrl+C to cancel, or Enter to continue...")
    
    try:
        input()
        main()
    except KeyboardInterrupt:
        print("\n\nExample cancelled by user")
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server")
        print("Please start the server with: uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n✗ Error: {e}")
