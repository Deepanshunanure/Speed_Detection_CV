"""Example usage of calibration API endpoints"""
import base64
import requests
import cv2
import numpy as np
import glob


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def calibrate_camera_example():
    """Example: Calibrate camera using API"""
    print("=== Camera Calibration Example ===\n")
    
    # API endpoint
    base_url = "http://localhost:8000"
    
    # Load calibration images (assuming you have chessboard images)
    # For this example, we'll show the structure
    calibration_images = []
    
    # In a real scenario, you would load actual chessboard images:
    # image_files = glob.glob("calibration_images/*.jpg")
    # for img_file in image_files[:15]:  # Use 15 images
    #     calibration_images.append(encode_image_to_base64(img_file))
    
    # For demonstration, we'll create synthetic images
    print("Note: This example shows the API structure.")
    print("In practice, you need real chessboard calibration images.\n")
    
    # Example request structure
    calibration_request = {
        "images": calibration_images,  # List of base64-encoded images
        "chessboard_size": [9, 6]  # Inner corners: 9 columns, 6 rows
    }
    
    print("1. Calibrating camera...")
    print(f"   - Number of images: {len(calibration_images)}")
    print(f"   - Chessboard size: {calibration_request['chessboard_size']}")
    
    # Uncomment to make actual API call:
    # response = requests.post(
    #     f"{base_url}/api/v1/calibration/calibrate",
    #     json=calibration_request
    # )
    # 
    # if response.status_code == 200:
    #     result = response.json()
    #     print(f"   ✓ Calibration successful!")
    #     print(f"   - Calibration error: {result['calibration_error']:.4f}")
    #     print(f"   - Pixels per meter: {result.get('pixels_per_meter', 'N/A')}")
    #     print(f"   - Date: {result['calibration_date']}")
    # else:
    #     print(f"   ✗ Calibration failed: {response.json()}")


def check_calibration_status():
    """Example: Check calibration status"""
    print("\n=== Check Calibration Status ===\n")
    
    base_url = "http://localhost:8000"
    
    print("2. Checking calibration status...")
    
    # Make API call
    response = requests.get(f"{base_url}/api/v1/calibration/status")
    
    if response.status_code == 200:
        result = response.json()
        
        if result['calibrated']:
            print(f"   ✓ Camera is calibrated")
            print(f"   - Calibration date: {result.get('calibration_date', 'N/A')}")
            print(f"   - Calibration error: {result.get('calibration_error', 'N/A')}")
            print(f"   - Pixels per meter: {result.get('pixels_per_meter', 'N/A')}")
        else:
            print(f"   ✗ Camera is not calibrated")
            print(f"   - Message: {result['message']}")
    else:
        print(f"   ✗ Status check failed: {response.json()}")


def load_calibration_example():
    """Example: Load calibration from file"""
    print("\n=== Load Calibration from File ===\n")
    
    base_url = "http://localhost:8000"
    
    print("3. Loading calibration from file...")
    
    # Request to load calibration
    load_request = {
        "filepath": "calibration.json"
    }
    
    response = requests.post(
        f"{base_url}/api/v1/calibration/load",
        json=load_request
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Calibration loaded successfully")
        print(f"   - Calibration error: {result['calibration_error']:.4f}")
        print(f"   - Calibration date: {result['calibration_date']}")
        
        # Display camera matrix
        print(f"   - Camera matrix shape: {len(result['camera_matrix'])}x{len(result['camera_matrix'][0])}")
        print(f"   - Distortion coefficients: {len(result['distortion_coefficients'])} values")
    elif response.status_code == 404:
        print(f"   ✗ Calibration file not found")
    else:
        print(f"   ✗ Load failed: {response.json()}")


def main():
    """Run all calibration API examples"""
    print("=" * 50)
    print("Calibration API Examples")
    print("=" * 50)
    print("\nMake sure the API server is running:")
    print("  python -m uvicorn src.api.main:app --reload\n")
    
    try:
        # Check if API is running
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("Error: API server is not responding")
            return
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API server")
        print("Please start the server first:")
        print("  python -m uvicorn src.api.main:app --reload")
        return
    
    # Run examples
    calibrate_camera_example()
    check_calibration_status()
    load_calibration_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
