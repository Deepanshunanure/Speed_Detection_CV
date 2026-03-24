"""Example demonstrating Detector usage"""
import numpy as np
import cv2
from src.components.preprocessor import Preprocessor
from src.components.detector import Detector
from src.config.models import PreprocessorConfig, DetectorConfig


def main():
    """Demonstrate object detection with background subtraction"""
    
    # Configure preprocessor
    prep_config = PreprocessorConfig(
        target_resolution=(640, 480),
        noise_reduction_method="gaussian",
        noise_reduction_kernel_size=5,
        normalize_intensity=False
    )
    preprocessor = Preprocessor(prep_config)
    
    # Configure detector
    det_config = DetectorConfig(
        background_subtraction_method="MOG2",
        background_learning_rate=0.01,
        edge_detection_enabled=False,
        canny_threshold1=50,
        canny_threshold2=150,
        min_contour_area=500.0,
        max_contour_area=50000.0
    )
    detector = Detector(det_config)
    
    print("Object Detection Example")
    print("=" * 50)
    print(f"Preprocessor config: {prep_config}")
    print(f"Detector config: {det_config}")
    print()
    
    # Simulate video frames with moving object
    num_frames = 20
    
    for frame_num in range(num_frames):
        # Create synthetic frame
        frame = np.random.randint(40, 60, (720, 1280, 3), dtype=np.uint8)
        
        # Add moving object after frame 5
        if frame_num >= 5:
            x_pos = 200 + (frame_num - 5) * 30
            y_pos = 200
            # Draw a white rectangle as the moving object
            cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 100, y_pos + 100),
                         (255, 255, 255), -1)
        
        # Process through pipeline
        preprocessed = preprocessor.process(frame)
        result = detector.detect(preprocessed)
        
        # Display results
        print(f"Frame {frame_num}:")
        print(f"  Detected objects: {len(result.bounding_boxes)}")
        
        for i, bbox in enumerate(result.bounding_boxes):
            print(f"  Object {i}:")
            print(f"    Position: ({bbox.x}, {bbox.y})")
            print(f"    Size: {bbox.width}x{bbox.height}")
            print(f"    Area: {bbox.area:.2f}")
            print(f"    Centroid: ({bbox.centroid[0]:.2f}, {bbox.centroid[1]:.2f})")
        
        if len(result.bounding_boxes) == 0:
            print("  (No objects detected)")
        
        print()
    
    print("=" * 50)
    print("Detection complete!")


if __name__ == "__main__":
    main()
