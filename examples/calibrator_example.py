"""Example usage of the Calibrator component"""
import numpy as np
import cv2
from src.components.calibrator import Calibrator
from src.config.models import CalibratorConfig


def create_sample_chessboard(chessboard_size=(9, 6), image_size=(640, 480)):
    """Create a sample chessboard image for demonstration"""
    square_size = 50
    border = square_size
    board_width = (chessboard_size[0] + 1) * square_size
    board_height = (chessboard_size[1] + 1) * square_size
    
    board = np.ones((board_height + 2 * border, board_width + 2 * border), dtype=np.uint8) * 255
    
    for i in range(chessboard_size[1] + 1):
        for j in range(chessboard_size[0] + 1):
            if (i + j) % 2 == 1:
                y1 = border + i * square_size
                y2 = border + (i + 1) * square_size
                x1 = border + j * square_size
                x2 = border + (j + 1) * square_size
                board[y1:y2, x1:x2] = 0
    
    board = cv2.resize(board, image_size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)


def main():
    """Demonstrate Calibrator usage"""
    print("=== Camera Calibrator Example ===\n")
    
    # 1. Initialize calibrator with configuration
    print("1. Initializing calibrator...")
    config = CalibratorConfig(
        calibration_file_path="calibration.json",
        chessboard_size=(9, 6),
        square_size_mm=25.0,
        perspective_transform_enabled=True
    )
    calibrator = Calibrator(config)
    print(f"   Chessboard size: {config.chessboard_size}")
    print(f"   Square size: {config.square_size_mm}mm")
    print(f"   Perspective transform: {config.perspective_transform_enabled}\n")
    
    # 2. Create sample calibration images
    print("2. Creating sample calibration images...")
    images = []
    for i, angle in enumerate(range(-10, 11, 2)):
        img = create_sample_chessboard((9, 6))
        # Apply slight rotation for variation
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))
        images.append(img)
    print(f"   Created {len(images)} calibration images\n")
    
    # 3. Perform calibration
    print("3. Performing camera calibration...")
    try:
        params = calibrator.calibrate(images, chessboard_size=(9, 6))
        print("   ✓ Calibration successful!")
        print(f"   Camera matrix:\n{params.camera_matrix}")
        print(f"   Distortion coefficients: {params.distortion_coefficients}")
        print(f"   Calibration error: {params.calibration_error:.4f}")
        print(f"   Pixels per meter: {params.pixels_per_meter:.2f}")
        if params.homography_matrix is not None:
            print(f"   Homography matrix computed: {params.homography_matrix.shape}\n")
    except ValueError as e:
        print(f"   ✗ Calibration failed: {e}\n")
        return
    
    # 4. Save calibration parameters
    print("4. Saving calibration parameters...")
    calibrator.save_calibration(params, "calibration.json")
    print("   ✓ Saved to calibration.json\n")
    
    # 5. Load calibration parameters
    print("5. Loading calibration parameters...")
    loaded_params = calibrator.load_calibration("calibration.json")
    print("   ✓ Loaded from calibration.json\n")
    
    # 6. Undistort a test frame
    print("6. Undistorting a test frame...")
    test_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    undistorted = calibrator.undistort(test_frame, loaded_params)
    print(f"   Original frame shape: {test_frame.shape}")
    print(f"   Undistorted frame shape: {undistorted.shape}")
    print("   ✓ Undistortion complete\n")
    
    # 7. Apply perspective transform
    if loaded_params.homography_matrix is not None:
        print("7. Applying perspective transform...")
        transformed = calibrator.apply_perspective_transform(test_frame, loaded_params)
        print(f"   Transformed frame shape: {transformed.shape}")
        print("   ✓ Perspective transform complete\n")
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    main()
