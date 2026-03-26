"""Video frame preprocessor module"""
import cv2
import numpy as np
import logging
import json
import time
from src.config.models import PreprocessorConfig


logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses video frames for optimal processing by downstream components"""
    
    def __init__(self, config: PreprocessorConfig):
        """
        Initialize preprocessor with configuration parameters
        
        Args:
            config: PreprocessorConfig with preprocessing parameters
        """
        self.config = config
        self._validate_config()
        logger.info(json.dumps({
            "component_name": "Preprocessor",
            "event": "initialized",
            "config": {
                "target_resolution": self.config.target_resolution,
                "noise_reduction_method": self.config.noise_reduction_method,
                "noise_reduction_kernel_size": self.config.noise_reduction_kernel_size,
                "normalize_intensity": self.config.normalize_intensity
            }
        }))
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config.noise_reduction_method not in ["gaussian", "bilateral", "median"]:
            raise ValueError(
                f"Invalid noise_reduction_method: {self.config.noise_reduction_method}. "
                "Must be 'gaussian', 'bilateral', or 'median'"
            )
        
        if self.config.noise_reduction_kernel_size <= 0:
            raise ValueError(
                f"Invalid noise_reduction_kernel_size: {self.config.noise_reduction_kernel_size}. "
                "Must be positive"
            )
        
        # Kernel size must be odd for OpenCV filters
        if self.config.noise_reduction_kernel_size % 2 == 0:
            raise ValueError(
                f"Invalid noise_reduction_kernel_size: {self.config.noise_reduction_kernel_size}. "
                "Must be odd"
            )
        
        if self.config.target_resolution[0] <= 0 or self.config.target_resolution[1] <= 0:
            raise ValueError(
                f"Invalid target_resolution: {self.config.target_resolution}. "
                "Width and height must be positive"
            )
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Processed frame (grayscale, filtered, resized, normalized)
        """
        start_time = time.time()
        
        try:
            # Validate input
            if frame is None or frame.size == 0:
                logger.error(json.dumps({
                    "component_name": "Preprocessor",
                    "event": "invalid_input",
                    "error": "frame is None or empty"
                }))
                raise ValueError("Invalid frame: frame is None or empty")
            
            # Step 1: Convert BGR to grayscale
            try:
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    # Already grayscale
                    gray = frame
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Preprocessor",
                    "event": "grayscale_conversion_failed",
                    "error": str(e),
                    "frame_shape": frame.shape
                }))
                raise
            
            # Step 2: Apply noise reduction filter
            try:
                filtered = self._apply_noise_reduction(gray)
            except Exception as e:
                logger.warning(json.dumps({
                    "component_name": "Preprocessor",
                    "event": "noise_reduction_failed",
                    "error": str(e)
                }))
                # Fallback: use unfiltered frame
                filtered = gray
            
            # Step 3: Resize to target resolution using INTER_AREA interpolation
            try:
                resized = cv2.resize(
                    filtered,
                    self.config.target_resolution,
                    interpolation=cv2.INTER_AREA
                )
            except Exception as e:
                logger.error(json.dumps({
                    "component_name": "Preprocessor",
                    "event": "resize_failed",
                    "error": str(e),
                    "target_resolution": self.config.target_resolution
                }))
                raise
            
            # Step 4: Normalize pixel values to [0, 1] range if enabled
            try:
                if self.config.normalize_intensity:
                    normalized = resized.astype(np.float32) / 255.0
                    result = normalized
                else:
                    result = resized
            except Exception as e:
                logger.warning(json.dumps({
                    "component_name": "Preprocessor",
                    "event": "normalization_failed",
                    "error": str(e)
                }))
                # Fallback: return unnormalized
                result = resized
            
            processing_time_ms = (time.time() - start_time) * 1000
            logger.debug(json.dumps({
                "component_name": "Preprocessor",
                "event": "frame_processed",
                "processing_time_ms": round(processing_time_ms, 2),
                "output_shape": result.shape
            }))
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(json.dumps({
                "component_name": "Preprocessor",
                "event": "processing_failed",
                "error": str(e),
                "processing_time_ms": round(processing_time_ms, 2)
            }))
            raise
    
    def _apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction filter based on configuration
        
        Args:
            frame: Grayscale frame
            
        Returns:
            Filtered frame
        """
        kernel_size = self.config.noise_reduction_kernel_size
        
        if self.config.noise_reduction_method == "gaussian":
            return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        elif self.config.noise_reduction_method == "bilateral":
            # Bilateral filter uses diameter instead of kernel size
            # sigmaColor and sigmaSpace are set to kernel_size for consistency
            return cv2.bilateralFilter(frame, kernel_size, kernel_size * 2, kernel_size * 2)
        
        elif self.config.noise_reduction_method == "median":
            return cv2.medianBlur(frame, kernel_size)
        
        # Should never reach here due to validation
        return frame
