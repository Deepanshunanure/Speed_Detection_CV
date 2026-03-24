"""Example demonstrating ConfigurationManager usage"""
from src.config.manager import ConfigurationManager


def main():
    # Load configuration from file
    config_manager = ConfigurationManager('config/default.yaml')
    
    # Get individual component configurations
    preprocessor_config = config_manager.get_preprocessor_config()
    print(f"Preprocessor target resolution: {preprocessor_config.target_resolution}")
    print(f"Noise reduction method: {preprocessor_config.noise_reduction_method}")
    
    detector_config = config_manager.get_detector_config()
    print(f"\nDetector background method: {detector_config.background_subtraction_method}")
    print(f"Min contour area: {detector_config.min_contour_area}")
    
    tracker_config = config_manager.get_tracker_config()
    print(f"\nTracker max distance: {tracker_config.max_tracking_distance}")
    print(f"Trajectory history length: {tracker_config.trajectory_history_length}")
    
    # Get complete pipeline configuration
    pipeline_config = config_manager.get_pipeline_config()
    print(f"\nEnabled components: {pipeline_config.enabled_components}")
    
    # Validate configuration
    warnings = config_manager.validate()
    if warnings:
        print(f"\nConfiguration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguration is valid (no warnings)")


if __name__ == "__main__":
    main()
