"""Example demonstrating the Tracker component"""
import numpy as np
import cv2
from src.components.tracker import Tracker
from src.components.detector import Detector
from src.components.preprocessor import Preprocessor
from src.config.models import TrackerConfig, DetectorConfig, PreprocessorConfig


def create_synthetic_video_frame(frame_num, num_objects=2):
    """Create a synthetic video frame with moving objects"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Object 1: Moving horizontally
    x1 = 100 + frame_num * 15
    y1 = 200
    cv2.circle(frame, (x1, y1), 30, (0, 255, 0), -1)
    
    # Object 2: Moving diagonally
    if num_objects > 1:
        x2 = 400 - frame_num * 10
        y2 = 100 + frame_num * 15
        cv2.circle(frame, (x2, y2), 25, (0, 0, 255), -1)
    
    return frame


def main():
    """Demonstrate tracker functionality"""
    print("Tracker Example")
    print("=" * 50)
    
    # Configure components
    preprocessor_config = PreprocessorConfig(
        target_resolution=(640, 480),
        noise_reduction_method="gaussian",
        noise_reduction_kernel_size=5,
        normalize_intensity=False
    )
    
    detector_config = DetectorConfig(
        background_subtraction_method="MOG2",
        background_learning_rate=0.1,
        min_contour_area=100.0,
        max_contour_area=10000.0
    )
    
    tracker_config = TrackerConfig(
        max_tracking_distance=100.0,
        max_disappeared_frames=30,
        trajectory_history_length=50
    )
    
    # Initialize components
    preprocessor = Preprocessor(preprocessor_config)
    detector = Detector(detector_config)
    tracker = Tracker(tracker_config)
    
    print(f"\nTracker Configuration:")
    print(f"  Max Tracking Distance: {tracker_config.max_tracking_distance} pixels")
    print(f"  Max Disappeared Frames: {tracker_config.max_disappeared_frames}")
    print(f"  Trajectory History Length: {tracker_config.trajectory_history_length}")
    
    # Process synthetic video frames
    num_frames = 20
    print(f"\nProcessing {num_frames} synthetic video frames...")
    
    for frame_num in range(num_frames):
        # Create synthetic frame
        frame = create_synthetic_video_frame(frame_num, num_objects=2)
        
        # Preprocess
        processed = preprocessor.process(frame)
        
        # Detect objects
        detections = detector.detect(processed)
        detections.frame_number = frame_num
        
        # Track objects
        tracking_result = tracker.update(detections)
        
        # Display results
        print(f"\nFrame {frame_num}:")
        print(f"  Detections: {len(detections.bounding_boxes)}")
        print(f"  Tracked Objects: {len(tracking_result.tracked_objects)}")
        
        for obj in tracking_result.tracked_objects:
            print(f"    Object ID {obj.object_id}:")
            print(f"      Position: ({obj.position[0]:.1f}, {obj.position[1]:.1f})")
            print(f"      Age: {obj.age} frames")
            print(f"      Trajectory Length: {len(obj.trajectory)} points")
            print(f"      Disappeared Count: {obj.disappeared_count}")
    
    print("\n" + "=" * 50)
    print("Tracker example completed successfully!")
    
    # Summary
    final_result = tracking_result
    print(f"\nFinal Summary:")
    print(f"  Total Tracked Objects: {len(final_result.tracked_objects)}")
    
    if len(final_result.tracked_objects) > 0:
        print(f"\n  Object Details:")
        for obj in final_result.tracked_objects:
            print(f"    ID {obj.object_id}: Age={obj.age}, Trajectory={len(obj.trajectory)} points")


if __name__ == "__main__":
    main()
