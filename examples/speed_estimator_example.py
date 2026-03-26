"""Example usage of SpeedEstimator component"""
import numpy as np
from src.components.speed_estimator import SpeedEstimator
from src.components.tracker import Tracker
from src.components.detector import Detector
from src.components.preprocessor import Preprocessor
from src.models import CalibrationParameters
from src.config.models import (
    SpeedEstimatorConfig,
    TrackerConfig,
    DetectorConfig,
    PreprocessorConfig
)
import cv2


def main():
    """Demonstrate speed estimation on a video"""
    
    # Initialize components
    preprocessor_config = PreprocessorConfig(
        target_resolution=(640, 480),
        noise_reduction_method="gaussian",
        noise_reduction_kernel_size=5,
        normalize_intensity=False
    )
    
    detector_config = DetectorConfig(
        background_subtraction_method="MOG2",
        background_learning_rate=0.01,
        min_contour_area=500.0,
        max_contour_area=50000.0
    )
    
    tracker_config = TrackerConfig(
        max_tracking_distance=50.0,
        max_disappeared_frames=30,
        trajectory_history_length=100
    )
    
    speed_estimator_config = SpeedEstimatorConfig(
        averaging_window_frames=10,
        min_trajectory_length=2,
        output_unit="km/h"  # Output in km/h
    )
    
    preprocessor = Preprocessor(preprocessor_config)
    detector = Detector(detector_config)
    tracker = Tracker(tracker_config)
    speed_estimator = SpeedEstimator(speed_estimator_config)
    
    # Load calibration (if available)
    try:
        import json
        with open("calibration.json", "r") as f:
            calib_data = json.load(f)
        
        calibration = CalibrationParameters(
            camera_matrix=np.array(calib_data["camera_matrix"]),
            distortion_coefficients=np.array(calib_data["distortion_coefficients"]),
            pixels_per_meter=calib_data.get("pixels_per_meter")
        )
        print(f"Loaded calibration: {calibration.pixels_per_meter} px/m")
    except FileNotFoundError:
        calibration = None
        print("No calibration found, using pixel-based speed")
    
    # Create synthetic video for demonstration
    print("\nGenerating synthetic video with moving objects...")
    
    # Video parameters
    width, height = 640, 480
    fps = 30.0
    num_frames = 100
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_speed_demo.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    
    for i in range(num_frames):
        # Create blank frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Draw moving objects
        # Object 1: Moving right at constant speed
        obj1_x = 50 + i * 3
        if obj1_x < width - 50:
            cv2.rectangle(frame, (obj1_x, 100), (obj1_x + 40, 140), (0, 0, 255), -1)
        
        # Object 2: Moving diagonally
        obj2_x = 50 + i * 2
        obj2_y = 300 + i * 1
        if obj2_x < width - 50 and obj2_y < height - 50:
            cv2.circle(frame, (obj2_x, obj2_y), 20, (0, 255, 0), -1)
        
        # Process frame through pipeline
        processed = preprocessor.process(frame)
        detections = detector.detect(processed)
        tracking = tracker.update(detections)
        speeds = speed_estimator.estimate_speeds(tracking, calibration, fps)
        
        # Annotate frame with tracking and speed info
        for tracked_obj in tracking.tracked_objects:
            # Draw bounding box
            bbox = tracked_obj.bounding_box
            cv2.rectangle(
                frame,
                (bbox.x, bbox.y),
                (bbox.x + bbox.width, bbox.y + bbox.height),
                (255, 0, 0),
                2
            )
            
            # Draw trajectory
            if len(tracked_obj.trajectory) > 1:
                points = np.array(tracked_obj.trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 255, 0), 2)
            
            # Find speed for this object
            speed_result = next((s for s in speeds if s.object_id == tracked_obj.object_id), None)
            
            if speed_result:
                # Display speed information
                speed_text = f"ID:{speed_result.object_id} Speed:{speed_result.instantaneous_speed:.1f} {speed_result.unit}"
                avg_text = f"Avg:{speed_result.average_speed:.1f} {speed_result.unit}"
                
                cv2.putText(
                    frame,
                    speed_text,
                    (bbox.x, bbox.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
                
                cv2.putText(
                    frame,
                    avg_text,
                    (bbox.x, bbox.y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )
        
        # Add frame info
        info_text = f"Frame: {i+1}/{num_frames} | Objects: {len(tracking.tracked_objects)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Print speed info every 10 frames
        if (i + 1) % 10 == 0 and speeds:
            print(f"\nFrame {i+1}:")
            for speed_result in speeds:
                print(f"  Object {speed_result.object_id}:")
                print(f"    Instantaneous: {speed_result.instantaneous_speed:.2f} {speed_result.unit}")
                print(f"    Average: {speed_result.average_speed:.2f} {speed_result.unit}")
                print(f"    Displacement: {speed_result.displacement_vector}")
                print(f"    Calibrated: {speed_result.calibrated}")
        
        frame_count += 1
    
    out.release()
    print(f"\n✓ Processed {frame_count} frames")
    print(f"✓ Output saved to: output_speed_demo.mp4")
    
    # Example: Using speed estimator with custom calibration
    print("\n" + "="*60)
    print("Example: Speed estimation with custom calibration")
    print("="*60)
    
    # Create custom calibration with known scale
    custom_calibration = CalibrationParameters(
        camera_matrix=np.eye(3),
        distortion_coefficients=np.zeros(5),
        pixels_per_meter=50.0  # 50 pixels = 1 meter
    )
    
    # Simulate a simple tracking scenario
    from src.models import TrackingResult, TrackedObject, BoundingBox
    
    # Object moving 15 pixels per frame
    trajectory = [
        (100.0, 100.0),
        (115.0, 100.0),
        (130.0, 100.0),
        (145.0, 100.0),
        (160.0, 100.0),
    ]
    
    bbox = BoundingBox(x=160, y=100, width=30, height=30, area=900, centroid=(175.0, 115.0))
    tracked_obj = TrackedObject(
        object_id=1,
        position=(160.0, 100.0),
        bounding_box=bbox,
        trajectory=trajectory,
        age=5,
        disappeared_count=0
    )
    
    tracking_result = TrackingResult(
        frame_number=5,
        tracked_objects=[tracked_obj]
    )
    
    # Estimate speed
    speeds = speed_estimator.estimate_speeds(tracking_result, custom_calibration, fps=30.0)
    
    print(f"\nObject moving 15 pixels/frame at 30 fps:")
    print(f"  Calibration: 50 pixels/meter")
    print(f"  Expected speed: 15 px/frame * 30 fps / 50 px/m = 9 m/s = 32.4 km/h")
    print(f"\nCalculated:")
    for speed in speeds:
        print(f"  Instantaneous: {speed.instantaneous_speed:.2f} {speed.unit}")
        print(f"  Average: {speed.average_speed:.2f} {speed.unit}")
        print(f"  Calibrated: {speed.calibrated}")
    
    # Example: Different output units
    print("\n" + "="*60)
    print("Example: Different output units")
    print("="*60)
    
    for unit in ["m/s", "km/h", "mph"]:
        config = SpeedEstimatorConfig(
            averaging_window_frames=10,
            min_trajectory_length=2,
            output_unit=unit
        )
        estimator = SpeedEstimator(config)
        speeds = estimator.estimate_speeds(tracking_result, custom_calibration, fps=30.0)
        
        print(f"\n{unit}:")
        print(f"  Speed: {speeds[0].instantaneous_speed:.2f} {speeds[0].unit}")


if __name__ == "__main__":
    main()
