"""Example usage of the FastAPI video processing API"""
import base64
import time
import requests
import cv2
import numpy as np


# API base URL (adjust if running on different host/port)
API_BASE_URL = "http://localhost:8000"


def encode_image_to_base64(image_path):
    """Encode image file to base64 string"""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')


def encode_frame_to_base64(frame):
    """Encode OpenCV frame to base64 string"""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer).decode('utf-8')


def example_process_frame():
    """Example: Process a single frame"""
    print("\n=== Example 1: Process Single Frame ===")
    
    # Create a test frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Encode to base64
    base64_image = encode_frame_to_base64(frame)
    
    # Make API request
    response = requests.post(
        f"{API_BASE_URL}/api/v1/process/frame",
        json={
            "image": base64_image,
            "include_annotated_frame": True
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Frame processed successfully")
        print(f"  Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"  Component times: {result['component_times']}")
        
        if result.get('detections'):
            print(f"  Detections: {result['detections']['num_objects']} objects")
        
        if result.get('tracking'):
            print(f"  Tracking: {result['tracking']['num_tracked_objects']} tracked objects")
        
        if result.get('speeds'):
            print(f"  Speeds: {len(result['speeds'])} speed measurements")
    else:
        print(f"✗ Error: {response.status_code} - {response.text}")


def example_process_video():
    """Example: Process a video file asynchronously"""
    print("\n=== Example 2: Process Video File ===")
    
    # Use existing demo video
    video_path = "demo_input.mp4"
    
    try:
        # Upload video for processing
        with open(video_path, "rb") as f:
            files = {"video_file": ("demo_input.mp4", f, "video/mp4")}
            response = requests.post(
                f"{API_BASE_URL}/api/v1/process/video",
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            task_id = result["task_id"]
            print(f"✓ Video processing task created")
            print(f"  Task ID: {task_id}")
            print(f"  Status: {result['status']}")
            
            # Poll for completion
            print("\n  Polling for completion...")
            max_attempts = 60
            attempt = 0
            
            while attempt < max_attempts:
                time.sleep(2)
                
                status_response = requests.get(
                    f"{API_BASE_URL}/api/v1/process/video/{task_id}"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"  Status: {status['status']}", end="")
                    
                    if status.get('progress') is not None:
                        print(f" ({status['progress']*100:.1f}%)")
                    else:
                        print()
                    
                    if status['status'] == 'completed':
                        print("\n✓ Video processing completed!")
                        summary = status['summary']
                        print(f"  Total frames: {summary['total_frames']}")
                        print(f"  Processed frames: {summary['processed_frames']}")
                        print(f"  Average FPS: {summary['average_fps']:.2f}")
                        print(f"  Objects detected: {summary['total_objects_detected']}")
                        print(f"  Unique objects tracked: {summary['unique_objects_tracked']}")
                        break
                    elif status['status'] == 'failed':
                        print(f"\n✗ Processing failed: {status.get('error')}")
                        break
                
                attempt += 1
            
            if attempt >= max_attempts:
                print("\n✗ Timeout waiting for processing to complete")
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
    
    except FileNotFoundError:
        print(f"✗ Video file not found: {video_path}")
        print("  Please ensure demo_input.mp4 exists in the current directory")


def example_health_check():
    """Example: Check API health"""
    print("\n=== Example 3: Health Check ===")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        health = response.json()
        print(f"✓ API is healthy")
        print(f"  Status: {health['status']}")
        print(f"  Active tasks: {health['active_tasks']}")
        print(f"  Timestamp: {health['timestamp']}")
    else:
        print(f"✗ Health check failed: {response.status_code}")


def main():
    """Run all examples"""
    print("=" * 60)
    print("FastAPI Video Processing Pipeline - API Examples")
    print("=" * 60)
    print("\nMake sure the API server is running:")
    print("  python -m uvicorn src.api.main:app --reload")
    print("\nOr:")
    print("  python src/api/main.py")
    
    try:
        # Check if API is available
        response = requests.get(API_BASE_URL, timeout=2)
        if response.status_code != 200:
            print("\n✗ API server is not responding correctly")
            return
    except requests.exceptions.RequestException:
        print("\n✗ Cannot connect to API server")
        print(f"  Please start the server at {API_BASE_URL}")
        return
    
    print("\n✓ API server is running")
    
    # Run examples
    example_health_check()
    example_process_frame()
    example_process_video()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
