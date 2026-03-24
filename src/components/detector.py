"""Object detection module using classical computer vision techniques"""
import cv2
import numpy as np
from typing import Optional
from src.config.models import DetectorConfig
from src.models import DetectionResult, BoundingBox


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
        self._background_subtractor = self._create_background_subtractor()
    
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
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        # Ensure frame is in correct format for background subtraction
        # Convert normalized float frames back to uint8 if needed
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame
        
        # Step 1: Apply background subtraction to identify foreground mask
        foreground_mask = self._background_subtractor.apply(
            frame_uint8,
            learningRate=self.config.background_learning_rate
        )
        
        # Step 2: Optionally apply Canny edge detection
        if self.config.edge_detection_enabled:
            edges = cv2.Canny(
                frame_uint8,
                self.config.canny_threshold1,
                self.config.canny_threshold2
            )
            # Combine foreground mask with edges using bitwise OR
            foreground_mask = cv2.bitwise_or(foreground_mask, edges)
        
        # Step 3: Find contours in the foreground mask
        contours, _ = cv2.findContours(
            foreground_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Step 4: Filter contours by area thresholds
        valid_contours = []
        bounding_boxes = []
        
        for contour in contours:
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
        
        # Create detection result
        # Note: frame_number will be set by the caller (pipeline orchestrator)
        result = DetectionResult(
            frame_number=0,  # Placeholder, will be set by orchestrator
            bounding_boxes=bounding_boxes,
            contours=valid_contours,
            foreground_mask=foreground_mask
        )
        
        return result
