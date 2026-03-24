"""Example usage of the Preprocessor component"""
import numpy as np
import cv2
from src.components.preprocessor import Preprocessor
from src.config.models import PreprocessorConfig


def main():
    """Demonstrate Preprocessor usage"""
    
    # Create a sample video frame (simulating a BGR image)
    print("Creating sample video frame (1920x1080)...")
    frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
    
    # Add some structure to the frame
    cv2.rectangle(frame, (400, 300), (800, 700), (255, 0, 0), -1)
    cv2.circle(frame, (1200, 500), 150, (0, 255, 0), -1)
    
    print(f"Original frame shape: {frame.shape}, dtype: {frame.dtype}")
    
    # Example 1: Basic preprocessing with Gaussian filter
    print("\n--- Example 1: Gaussian filter with normalization ---")
    config1 = PreprocessorConfig(
        target_resolution=(640, 480),
        noise_reduction_method="gaussian",
        noise_reduction_kernel_size=5,
        normalize_intensity=True
    )
    preprocessor1 = Preprocessor(config1)
    result1 = preprocessor1.process(frame)
    print(f"Processed frame shape: {result1.shape}, dtype: {result1.dtype}")
    print(f"Value range: [{result1.min():.3f}, {result1.max():.3f}]")
    
    # Example 2: Bilateral filter without normalization
    print("\n--- Example 2: Bilateral filter without normalization ---")
    config2 = PreprocessorConfig(
        target_resolution=(1280, 720),
        noise_reduction_method="bilateral",
        noise_reduction_kernel_size=5,
        normalize_intensity=False
    )
    preprocessor2 = Preprocessor(config2)
    result2 = preprocessor2.process(frame)
    print(f"Processed frame shape: {result2.shape}, dtype: {result2.dtype}")
    print(f"Value range: [{result2.min()}, {result2.max()}]")
    
    # Example 3: Median filter for noise reduction
    print("\n--- Example 3: Median filter ---")
    config3 = PreprocessorConfig(
        target_resolution=(800, 600),
        noise_reduction_method="median",
        noise_reduction_kernel_size=7,
        normalize_intensity=True
    )
    preprocessor3 = Preprocessor(config3)
    result3 = preprocessor3.process(frame)
    print(f"Processed frame shape: {result3.shape}, dtype: {result3.dtype}")
    print(f"Value range: [{result3.min():.3f}, {result3.max():.3f}]")
    
    print("\n✓ All preprocessing examples completed successfully!")


if __name__ == "__main__":
    main()
