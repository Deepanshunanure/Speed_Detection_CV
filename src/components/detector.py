"""Object detection module using classical computer vision techniques"""
import cv2
import numpy as np
import logging
import json
import time
from typing import Optional
from src.config.models import DetectorConfig
from src.models import DetectionResult, BoundingBox


logger = logging.getLogger(__name__)


class Detector:
    """Detects objects in video frames using background subtraction and edge detection"""
    
    def __init__(self, config: DetectorConfig):
        """
        Initialize detector with background subtractor and parameters
        
        Args:
            config: DetectorConfig with detection parameters
        """
        self.config = config
        self._validate_config()
        
        try:
            self._background_subtractor = self._create_background_subtractor()
            logger.info(json.dumps({
                "component_name": "Detector",
                "event": "initialized",
                "config": {
                    "background_subtraction_method": self.config.background_subtraction_method,
                    "background_learning_rate": self.config.background_learning_rate,
                    "edge_detection_enabled": self.config.edge_detection_enabled,
                    "min_contour_area": self.config.min_contour_area,
                    "max_contour_area": self.config.max_contour_area
                }
            }))
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "Detector",
                "event": "initialization_failed",
                "error": str(e)
            }))
            raise
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.background_subtraction_method not in ["MOG2", "KNN", "GMG"]:
            raise ValueError(
                f"Invalid background_subtraction_method: {self.config.background_subtraction_method}. "
                "Must be 'MOG2', 'KNN', or 'GMG'"
            )
        
        if not 0.0 <= self.config.background_learning_rate <= 1.0:
            raise ValueError(
                f"Invalid background_learning_rate: {self.config.background_learning_rate}. "
                "Must be between 0.0 and 1.0"
            )
        
        if self.config.canny_threshold1 < 0 or self.config.canny_threshold2 < 0:
            raise ValueError(
                "Canny thresholds must be non-negative"
            )
        
        if self.config.canny_threshold1 >= self.config.canny_threshold2:
            raise ValueError(
                f"canny_threshold1 ({self.config.canny_threshold1}) must be less than "
                f"canny_threshold2 ({self.config.canny_threshold2})"
            )
        
        if self.config.min_contour_area < 0:
            raise ValueError(
                f"Invalid min_contour_area: {self.config.min_contour_area}. "
                "Must be non-negative"
            )
        
        if self.config.max_contour_area < self.config.min_contour_area:
            raise ValueError(
                f"max_contour_area ({self.config.max_contour_area}) must be greater than "
                f"min_contour_area ({self.config.min_contour_area})"
            )
    
    def _create_background_subtractor(self):
        """
        Create background subtractor based on configuration
        
        Returns:
            OpenCV background subtractor object
        """
        method = self.config.background_subtraction_method
        
        if method == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                detectShadows=True
            )
        elif method == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                detectShadows=True
            )
        elif method == "GMG":
            # GMG is in contrib module, may not be available
            # Fall back to MOG2 if not available
            try:
                return cv2.bgsegm.createBackgroundSubtractorGMG()
            except AttributeError:
                # GMG not available, use MOG2 as fallback
                return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        # Should never reach here due to validation
        return cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect objects in preprocessed frame
        
        Args:
            frame: Preprocessed grayscale frame
            
        Returns:
            Detection results with bounding boxes and contours
        """
        start_time = time.time()
        
        try:
            # Validate input
            if frame is None or frame.size == 0:
                logger.error(json.dumps({
                    "component_name": "Detector",
                    "event": "invalid_input",
                    "error": "frame is None or empty"
                }))
                raise ValueError("Invalid frame: frame is None or empty")
            
            # Ensure frame is in correct format for background subtraction
            # Convert normalized float frames back to uint8 if needed
            try:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                else:
                    frame_uint8 = frame
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Detector",
                    "event": "frame_conversion_failed",
                    "error": str(e),
                    "frame_dtype": str(frame.dtype)
                }))
                raise
            
            # Step 1: Apply background subtraction to identify foreground mask
            try:
                foreground_mask = self._background_subtractor.apply(
                    frame_uint8,
                    learningRate=self.config.background_learning_rate
                )
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Detector",
                    "event": "background_subtraction_failed",
                    "error": str(e)
                }))
                # Fallback: create empty mask
                foreground_mask = np.zeros(frame_uint8.shape[:2], dtype=np.uint8)
            
            # Step 2: Optionally apply Canny edge detection
            if self.config.edge_detection_enabled:
                try:
                    edges = cv2.Canny(
                        frame_uint8,
                        self.config.canny_threshold1,
                        self.config.canny_threshold2
                    )
                    # Combine foreground mask with edges using bitwise OR
                    foreground_mask = cv2.bitwise_or(foreground_mask, edges)
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "Detector",
                        "event": "edge_detection_failed",
                        "error": str(e)
                    }))
                    # Continue with foreground mask only
            
            # Step 3: Find contours in the foreground mask
            try:
                contours, _ = cv2.findContours(
                    foreground_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Detector",
                    "event": "contour_detection_failed",
                    "error": str(e)
                }))
                contours = []
            
            # Step 4: Filter contours by area thresholds
            valid_contours = []
            bounding_boxes = []
            
            for contour in contours:
                try:
                    area = cv2.contourArea(contour)
                    
                    # Filter by area thresholds
                    if self.config.min_contour_area <= area <= self.config.max_contour_area:
                        valid_contours.append(contour)
                        
                        # Step 5: Compute bounding box and centroid
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Compute centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                        else:
                            # Fallback to bounding box center if moment calculation fails
                            cx = x + w / 2.0
                            cy = y + h / 2.0
                        
                        bbox = BoundingBox(
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            area=area,
                            centroid=(cx, cy)
                        )
                        bounding_boxes.append(bbox)
                except Exception as e:
                    logger.warning(json.dumps({
                        "component_name": "Detector",
                        "event": "contour_processing_failed",
                        "error": str(e)
                    }))
                    # Skip this contour and continue
                    continue
            
            # Create detection result
            # Note: frame_number will be set by the caller (pipeline orchestrator)
            result = DetectionResult(
                frame_number=0,  # Placeholder, will be set by orchestrator
                bounding_boxes=bounding_boxes,
                contours=valid_contours,
                foreground_mask=foreground_mask
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            logger.debug(json.dumps({
                "component_name": "Detector",
                "event": "detection_complete",
                "num_detections": len(bounding_boxes),
                "processing_time_ms": round(processing_time_ms, 2)
            }))
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(json.dumps({
                "component_name": "Detector",
                "event": "detection_failed",
                "error": str(e),
                "processing_time_ms": round(processing_time_ms, 2)
            }))
            raise
