"""Video frame preprocessor module"""
import cv2
import numpy as np
from src.config.models import PreprocessorConfig


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
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: frame is None or empty")
        
        # Step 1: Convert BGR to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Already grayscale
            gray = frame
        
        # Step 2: Apply noise reduction filter
        filtered = self._apply_noise_reduction(gray)
        
        # Step 3: Resize to target resolution using INTER_AREA interpolation
        resized = cv2.resize(
            filtered,
            self.config.target_resolution,
            interpolation=cv2.INTER_AREA
        )
        
        # Step 4: Normalize pixel values to [0, 1] range if enabled
        if self.config.normalize_intensity:
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        
        return resized
    
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
