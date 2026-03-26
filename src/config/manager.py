"""Configuration management for the video processing pipeline"""
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, List
from src.config.models import (
    PipelineConfig,
    PreprocessorConfig,
    DetectorConfig,
    TrackerConfig,
    CalibratorConfig,
    SpeedEstimatorConfig,
    LoggingConfig
)


logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Manages pipeline configuration loading, validation, and access"""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager and load configuration file
        
        Args:
            config_path: Path to YAML or JSON configuration file
        """
        self.config_path = Path(config_path)
        self._config_data = {}
        
        try:
            self._load_configuration()
            logger.info(json.dumps({
                "component_name": "ConfigurationManager",
                "event": "initialized",
                "config_path": str(self.config_path)
            }))
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "ConfigurationManager",
                "event": "initialization_failed",
                "config_path": str(self.config_path),
                "error": str(e)
            }))
            raise
    
    def _load_configuration(self) -> None:
        """Load configuration from file or use defaults if file doesn't exist"""
        if not self.config_path.exists():
            # Use default configuration when file is missing
            logger.warning(json.dumps({
                "component_name": "ConfigurationManager",
                "event": "config_file_not_found",
                "config_path": str(self.config_path),
                "action": "using_defaults"
            }))
            self._config_data = self._get_default_config()
            return
        
        # Load from file based on extension
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    self._config_data = yaml.safe_load(f)
                elif self.config_path.suffix == '.json':
                    self._config_data = json.load(f)
                else:
                    error_msg = f"Unsupported configuration file format: {self.config_path.suffix}"
                    logger.error(json.dumps({
                        "component_name": "ConfigurationManager",
                        "event": "unsupported_format",
                        "config_path": str(self.config_path),
                        "suffix": self.config_path.suffix
                    }))
                    raise ValueError(error_msg)
            
            logger.info(json.dumps({
                "component_name": "ConfigurationManager",
                "event": "config_loaded",
                "config_path": str(self.config_path)
            }))
        except Exception as e:
            logger.error(json.dumps({
                "component_name": "ConfigurationManager",
                "event": "config_load_failed",
                "config_path": str(self.config_path),
                "error": str(e)
            }))
            raise RuntimeError(f"Failed to load configuration from {self.config_path}: {e}")
    
    def _get_default_config(self) -> dict:
        """Return default configuration dictionary"""
        return {
            'pipeline': {
                'enabled_components': ['preprocessor', 'detector', 'tracker', 'speed_estimator']
            },
            'preprocessor': {},
            'detector': {},
            'tracker': {},
            'calibrator': {},
            'speed_estimator': {},
            'logging': {}
        }
    
    def get_preprocessor_config(self) -> PreprocessorConfig:
        """Get preprocessor configuration with defaults"""
        config_dict = self._config_data.get('preprocessor', {})
        return PreprocessorConfig(
            target_resolution=tuple(config_dict.get('target_resolution', (1280, 720))),
            noise_reduction_method=config_dict.get('noise_reduction_method', 'gaussian'),
            noise_reduction_kernel_size=config_dict.get('noise_reduction_kernel_size', 5),
            normalize_intensity=config_dict.get('normalize_intensity', True)
        )
    
    def get_detector_config(self) -> DetectorConfig:
        """Get detector configuration with defaults"""
        config_dict = self._config_data.get('detector', {})
        return DetectorConfig(
            background_subtraction_method=config_dict.get('background_subtraction_method', 'MOG2'),
            background_learning_rate=config_dict.get('background_learning_rate', 0.01),
            edge_detection_enabled=config_dict.get('edge_detection_enabled', False),
            canny_threshold1=config_dict.get('canny_threshold1', 50),
            canny_threshold2=config_dict.get('canny_threshold2', 150),
            min_contour_area=config_dict.get('min_contour_area', 500.0),
            max_contour_area=config_dict.get('max_contour_area', 50000.0)
        )
    
    def get_tracker_config(self) -> TrackerConfig:
        """Get tracker configuration with defaults"""
        config_dict = self._config_data.get('tracker', {})
        return TrackerConfig(
            max_tracking_distance=config_dict.get('max_tracking_distance', 50.0),
            max_disappeared_frames=config_dict.get('max_disappeared_frames', 30),
            trajectory_history_length=config_dict.get('trajectory_history_length', 100)
        )
    
    def get_calibrator_config(self) -> CalibratorConfig:
        """Get calibrator configuration with defaults"""
        config_dict = self._config_data.get('calibrator', {})
        return CalibratorConfig(
            calibration_file_path=config_dict.get('calibration_file_path', 'calibration.json'),
            chessboard_size=tuple(config_dict.get('chessboard_size', (9, 6))),
            square_size_mm=config_dict.get('square_size_mm', 25.0),
            perspective_transform_enabled=config_dict.get('perspective_transform_enabled', False)
        )
    
    def get_speed_estimator_config(self) -> SpeedEstimatorConfig:
        """Get speed estimator configuration with defaults"""
        config_dict = self._config_data.get('speed_estimator', {})
        return SpeedEstimatorConfig(
            averaging_window_frames=config_dict.get('averaging_window_frames', 10),
            min_trajectory_length=config_dict.get('min_trajectory_length', 2),
            output_unit=config_dict.get('output_unit', 'm/s')
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration with defaults"""
        config_dict = self._config_data.get('logging', {})
        return LoggingConfig(
            level=config_dict.get('level', 'INFO'),
            file_path=config_dict.get('file_path'),
            log_to_file=config_dict.get('log_to_file', False)
        )
    
    def get_pipeline_config(self) -> PipelineConfig:
        """Get complete pipeline configuration"""
        pipeline_dict = self._config_data.get('pipeline', {})
        enabled_components = pipeline_dict.get('enabled_components', 
                                               ['preprocessor', 'detector', 'tracker', 'speed_estimator'])
        
        return PipelineConfig(
            enabled_components=enabled_components,
            preprocessor=self.get_preprocessor_config(),
            detector=self.get_detector_config(),
            tracker=self.get_tracker_config(),
            calibrator=self.get_calibrator_config(),
            speed_estimator=self.get_speed_estimator_config(),
            logging=self.get_logging_config()
        )
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters and return list of warnings
        
        Returns:
            List of warning messages for invalid or suboptimal parameters
        """
        warnings = []
        
        # Validate preprocessor config
        preprocessor = self.get_preprocessor_config()
        if preprocessor.target_resolution[0] < 64 or preprocessor.target_resolution[1] < 64:
            warnings.append("Preprocessor: target_resolution is very small (< 64 pixels)")
        if preprocessor.target_resolution[0] > 3840 or preprocessor.target_resolution[1] > 2160:
            warnings.append("Preprocessor: target_resolution is very large (> 4K)")
        if preprocessor.noise_reduction_method not in ['gaussian', 'bilateral', 'median']:
            warnings.append(f"Preprocessor: unknown noise_reduction_method '{preprocessor.noise_reduction_method}'")
        if preprocessor.noise_reduction_kernel_size % 2 == 0:
            warnings.append("Preprocessor: noise_reduction_kernel_size should be odd")
        if preprocessor.noise_reduction_kernel_size < 3 or preprocessor.noise_reduction_kernel_size > 31:
            warnings.append("Preprocessor: noise_reduction_kernel_size should be between 3 and 31")
        
        # Validate detector config
        detector = self.get_detector_config()
        if detector.background_subtraction_method not in ['MOG2', 'KNN', 'GMG']:
            warnings.append(f"Detector: unknown background_subtraction_method '{detector.background_subtraction_method}'")
        if detector.background_learning_rate < 0.0 or detector.background_learning_rate > 1.0:
            warnings.append("Detector: background_learning_rate should be between 0.0 and 1.0")
        if detector.canny_threshold1 >= detector.canny_threshold2:
            warnings.append("Detector: canny_threshold1 should be less than canny_threshold2")
        if detector.min_contour_area >= detector.max_contour_area:
            warnings.append("Detector: min_contour_area should be less than max_contour_area")
        if detector.min_contour_area < 0:
            warnings.append("Detector: min_contour_area should be non-negative")
        
        # Validate tracker config
        tracker = self.get_tracker_config()
        if tracker.max_tracking_distance <= 0:
            warnings.append("Tracker: max_tracking_distance should be positive")
        if tracker.max_disappeared_frames <= 0:
            warnings.append("Tracker: max_disappeared_frames should be positive")
        if tracker.trajectory_history_length <= 0:
            warnings.append("Tracker: trajectory_history_length should be positive")
        if tracker.trajectory_history_length > 1000:
            warnings.append("Tracker: trajectory_history_length is very large (> 1000), may use excessive memory")
        
        # Validate calibrator config
        calibrator = self.get_calibrator_config()
        if calibrator.chessboard_size[0] < 3 or calibrator.chessboard_size[1] < 3:
            warnings.append("Calibrator: chessboard_size should have at least 3 corners in each dimension")
        if calibrator.square_size_mm <= 0:
            warnings.append("Calibrator: square_size_mm should be positive")
        
        # Validate speed estimator config
        speed_estimator = self.get_speed_estimator_config()
        if speed_estimator.averaging_window_frames <= 0:
            warnings.append("SpeedEstimator: averaging_window_frames should be positive")
        if speed_estimator.min_trajectory_length < 2:
            warnings.append("SpeedEstimator: min_trajectory_length should be at least 2")
        if speed_estimator.output_unit not in ['m/s', 'km/h', 'mph', 'px/s']:
            warnings.append(f"SpeedEstimator: unknown output_unit '{speed_estimator.output_unit}'")
        
        # Validate logging config
        logging = self.get_logging_config()
        if logging.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            warnings.append(f"Logging: unknown level '{logging.level}'")
        
        # Validate pipeline config
        pipeline = self.get_pipeline_config()
        valid_components = ['preprocessor', 'detector', 'tracker', 'calibrator', 'speed_estimator']
        for component in pipeline.enabled_components:
            if component not in valid_components:
                warnings.append(f"Pipeline: unknown component '{component}' in enabled_components")
        
        return warnings
