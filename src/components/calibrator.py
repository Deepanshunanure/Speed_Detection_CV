"""Camera calibration component"""
import json
import logging
import time
from typing import List, Tuple, Optional
from datetime import datetime
import numpy as np
import cv2

from src.config.models import CalibratorConfig
from src.models import CalibrationParameters


logger = logging.getLogger(__name__)


class Calibrator:
    """Camera calibration for distortion correction and spatial measurements"""
    
    def __init__(self, config: CalibratorConfig):
        """
        Initialize calibrator with configuration
        
        Args:
            config: Calibrator configuration parameters
        """
        self.config = config
        logger.info(json.dumps({
            "component_name": "Calibrator",
            "event": "initialized",
            "config": {
                "chessboard_size": config.chessboard_size,
                "square_size_mm": config.square_size_mm,
                "perspective_transform_enabled": config.perspective_transform_enabled
            }
        }))
    
    def calibrate(
        self, 
        images: List[np.ndarray], 
        chessboard_size: Tuple[int, int]
    ) -> CalibrationParameters:
        """
        Calibrate camera from chessboard images
        
        Args:
            images: List of calibration images containing chessboard pattern
            chessboard_size: Inner corners (cols, rows) of chessboard
            
        Returns:
            Calibration parameters including camera matrix and distortion coefficients
            
        Raises:
            ValueError: If insufficient valid images or calibration fails
        """
        start_time = time.time()
        
        try:
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_started",
                "num_images": len(images),
                "chessboard_size": chessboard_size
            }))
            
            # Prepare object points (3D points in real world space)
            objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
            objp *= self.config.square_size_mm
            
            # Arrays to store object points and image points
            objpoints = []  # 3D points in real world space
            imgpoints = []  # 2D points in image plane
            
            # Find chessboard corners in each image
            valid_images = 0
            for i, img in enumerate(images):
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                    
                    # Find chessboard corners
                    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                    
                    if ret:
                        objpoints.append(objp)
                        
                        # Refine corner locations
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                        imgpoints.append(corners_refined)
                        
                        valid_images += 1
                        logger.debug(json.dumps({
                            "component_name": "Calibrator",
                            "event": "chessboard_detected",
                            "image_index": i
                        }))
                    else:
                        logger.warning(json.dumps({
                            "component_name": "Calibrator",
                            "event": "chessboard_detection_failed",
                            "image_index": i
                        }))
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "Calibrator",
                        "event": "image_processing_failed",
                        "image_index": i,
                        "error": str(e)
                    }))
                    continue
            
            if valid_images < 10:
                error_msg = f"Insufficient valid calibration images: {valid_images} found, 10 required"
                logger.error(json.dumps({
                    "component_name": "Calibrator",
                    "event": "calibration_failed",
                    "error": error_msg,
                    "valid_images": valid_images
                }))
                raise ValueError(error_msg)
            
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "images_processed",
                "valid_images": valid_images,
                "total_images": len(images)
            }))
            
            # Calibrate camera
            try:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                if not ret:
                    raise ValueError("Camera calibration failed")
                
                # Flatten distortion coefficients to 1D array
                dist_coeffs = dist_coeffs.flatten()
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Calibrator",
                    "event": "calibration_computation_failed",
                    "error": str(e)
                }))
                raise
            
            # Calculate reprojection error
            try:
                total_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(
                        objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                
                mean_error = total_error / len(objpoints)
            except Exception as e:
                logger.warning(json.dumps({
                    "component_name": "Calibrator",
                    "event": "reprojection_error_calculation_failed",
                    "error": str(e)
                }))
                mean_error = 0.0
            
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_complete",
                "mean_reprojection_error": round(mean_error, 4)
            }))
            
            # Compute homography if perspective transform is enabled
            homography_matrix = None
            if self.config.perspective_transform_enabled and len(imgpoints) > 0:
                try:
                    # Use first image for homography computation
                    src_points = imgpoints[0].reshape(-1, 2)
                    # Create destination points for top-down view
                    dst_points = objp[:, :2].astype(np.float32)
                    homography_matrix, _ = cv2.findHomography(src_points, dst_points)
                    logger.info(json.dumps({
                        "component_name": "Calibrator",
                        "event": "homography_computed"
                    }))
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "Calibrator",
                        "event": "homography_computation_failed",
                        "error": str(e)
                    }))
            
            # Calculate pixels per meter
            pixels_per_meter = None
            if len(imgpoints) > 0:
                try:
                    # Estimate from average distance between corners
                    corners = imgpoints[0].reshape(-1, 2)
                    if len(corners) > 1:
                        # Calculate average pixel distance between adjacent corners
                        pixel_dist = np.mean([
                            np.linalg.norm(corners[i] - corners[i+1]) 
                            for i in range(min(10, len(corners)-1))
                        ])
                        # Convert to pixels per meter
                        pixels_per_meter = pixel_dist / (self.config.square_size_mm / 1000.0)
                        logger.info(json.dumps({
                            "component_name": "Calibrator",
                            "event": "pixels_per_meter_calculated",
                            "pixels_per_meter": round(pixels_per_meter, 2)
                        }))
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "Calibrator",
                        "event": "pixels_per_meter_calculation_failed",
                        "error": str(e)
                    }))
            
            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_finished",
                "processing_time_ms": round(processing_time_ms, 2)
            }))
            
            return CalibrationParameters(
                camera_matrix=camera_matrix,
                distortion_coefficients=dist_coeffs,
                homography_matrix=homography_matrix,
                pixels_per_meter=pixels_per_meter,
                calibration_error=mean_error,
                calibration_date=datetime.now().isoformat()
            )
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_failed",
                "error": str(e),
                "processing_time_ms": round(processing_time_ms, 2)
            }))
            raise
    
    def save_calibration(self, params: CalibrationParameters, filepath: str) -> None:
        """
        Save calibration parameters to JSON file
        
        Args:
            params: Calibration parameters to save
            filepath: Path to output JSON file
        """
        try:
            data = {
                "camera_matrix": params.camera_matrix.tolist(),
                "distortion_coefficients": params.distortion_coefficients.tolist(),
                "homography_matrix": params.homography_matrix.tolist() if params.homography_matrix is not None else None,
                "pixels_per_meter": float(params.pixels_per_meter) if params.pixels_per_meter is not None else None,
                "calibration_error": float(params.calibration_error),
                "calibration_date": params.calibration_date
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_saved",
                "filepath": filepath
            }))
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_save_failed",
                "filepath": filepath,
                "error": str(e)
            }))
            raise
    
    def load_calibration(self, filepath: str) -> CalibrationParameters:
        """
        Load calibration parameters from JSON file
        
        Args:
            filepath: Path to JSON file containing calibration data
            
        Returns:
            Loaded calibration parameters
            
        Raises:
            FileNotFoundError: If calibration file doesn't exist
            ValueError: If calibration file is invalid
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            homography_matrix = None
            if data.get("homography_matrix") is not None:
                homography_matrix = np.array(data["homography_matrix"], dtype=np.float32)
            
            params = CalibrationParameters(
                camera_matrix=np.array(data["camera_matrix"], dtype=np.float32),
                distortion_coefficients=np.array(data["distortion_coefficients"], dtype=np.float32),
                homography_matrix=homography_matrix,
                pixels_per_meter=data.get("pixels_per_meter"),
                calibration_error=data.get("calibration_error", 0.0),
                calibration_date=data.get("calibration_date", "")
            )
            
            logger.info(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_loaded",
                "filepath": filepath
            }))
            return params
            
        except FileNotFoundError:
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_file_not_found",
                "filepath": filepath
            }))
            raise
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "calibration_load_failed",
                "filepath": filepath,
                "error": str(e)
            }))
            raise ValueError(f"Invalid calibration file: {e}")
    
    def undistort(self, frame: np.ndarray, params: CalibrationParameters) -> np.ndarray:
        """
        Apply distortion correction to frame
        
        Args:
            frame: Input frame to undistort
            params: Calibration parameters with camera matrix and distortion coefficients
            
        Returns:
            Undistorted frame with same dimensions as input
        """
        try:
            undistorted = cv2.undistort(
                frame,
                params.camera_matrix,
                params.distortion_coefficients
            )
            logger.debug(json.dumps({
                "component_name": "Calibrator",
                "event": "frame_undistorted"
            }))
            return undistorted
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "undistortion_failed",
                "error": str(e)
            }))
            # Return original frame as fallback
            return frame
    
    def apply_perspective_transform(
        self, 
        frame: np.ndarray, 
        params: CalibrationParameters
    ) -> np.ndarray:
        """
        Apply homography for top-down perspective view
        
        Args:
            frame: Input frame
            params: Calibration parameters with homography matrix
            
        Returns:
            Transformed frame
            
        Raises:
            ValueError: If homography matrix is not available
        """
        if params.homography_matrix is None:
            error_msg = "Homography matrix not available in calibration parameters"
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "perspective_transform_failed",
                "error": error_msg
            }))
            raise ValueError(error_msg)
        
        try:
            h, w = frame.shape[:2]
            transformed = cv2.warpPerspective(frame, params.homography_matrix, (w, h))
            logger.debug(json.dumps({
                "component_name": "Calibrator",
                "event": "perspective_transform_applied"
            }))
            return transformed
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "Calibrator",
                "event": "perspective_transform_failed",
                "error": str(e)
            }))
            # Return original frame as fallback
            return frame
