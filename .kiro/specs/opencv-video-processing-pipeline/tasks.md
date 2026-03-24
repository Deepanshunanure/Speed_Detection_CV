# Implementation Plan: OpenCV Video Processing Pipeline

## Overview

This implementation plan breaks down the OpenCV Video Processing Pipeline into discrete coding tasks. The system will be built incrementally, starting with core data structures and configuration, then implementing each processing component (Preprocessor, Detector, Tracker, Calibrator, Speed Estimator), followed by the pipeline orchestrator and FastAPI backend. Each task builds on previous work, with checkpoints to validate functionality before proceeding.

## Tasks

- [x] 1. Set up project structure and core data models
  - Create directory structure: `src/`, `src/components/`, `src/api/`, `src/config/`, `tests/`
  - Define core data classes: `VideoFrame`, `DetectionResult`, `TrackingResult`, `CalibrationParameters`, `SpeedResult`, `PipelineResult`
  - Create configuration data models: `PipelineConfig`, `PreprocessorConfig`, `DetectorConfig`, `TrackerConfig`, `CalibratorConfig`, `SpeedEstimatorConfig`, `LoggingConfig`
  - Set up dependencies: `requirements.txt` with opencv-python, numpy, fastapi, uvicorn, pydantic, pyyaml
  - _Requirements: 6.1, 6.5, 9.1_

- [ ]* 1.1 Write unit tests for data models
  - Test data class instantiation and field validation
  - Test configuration model validation with valid and invalid inputs
  - _Requirements: 6.1, 9.7_

- [ ] 2. Implement Configuration Manager
  - [x] 2.1 Create ConfigurationManager class with YAML/JSON loading
    - Implement `__init__(config_path: str)` to load configuration files
    - Implement getter methods for each component configuration
    - Implement `validate()` method to check parameter validity
    - Support default values when configuration file is missing
    - _Requirements: 6.5, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [ ]* 2.2 Write property test for configuration round trip
    - **Property 20: Configuration Round Trip**
    - **Validates: Requirements 6.5**

  - [ ]* 2.3 Write unit tests for Configuration Manager
    - Test loading valid YAML and JSON files
    - Test default value fallback when file is missing
    - Test validation warnings for invalid parameters
    - _Requirements: 9.6, 9.7_

- [ ] 3. Implement Preprocessor Module
  - [x] 3.1 Create Preprocessor class with frame processing
    - Implement `__init__(config: PreprocessorConfig)` to initialize parameters
    - Implement `process(frame: np.ndarray) -> np.ndarray` method
    - Add grayscale conversion using `cv2.cvtColor`
    - Add noise reduction with Gaussian, bilateral, or median filters
    - Add resizing to target resolution using `cv2.resize` with INTER_AREA
    - Add optional intensity normalization to [0, 1] range
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 3.2 Write property test for grayscale conversion
    - **Property 1: Grayscale Conversion**
    - **Validates: Requirements 1.1**

  - [ ]* 3.3 Write property test for resolution transformation
    - **Property 2: Resolution Transformation**
    - **Validates: Requirements 1.3**

  - [ ]* 3.4 Write property test for intensity normalization
    - **Property 3: Intensity Normalization**
    - **Validates: Requirements 1.4**

  - [ ]* 3.5 Write unit tests for Preprocessor
    - Test each noise reduction method with synthetic noisy images
    - Test edge cases: empty frames, single-pixel frames, maximum resolution
    - Test performance target: < 50ms for 1920x1080 input
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 4. Implement Detector Module
  - [x] 4.1 Create Detector class with object detection
    - Implement `__init__(config: DetectorConfig)` to initialize background subtractor
    - Implement `detect(frame: np.ndarray) -> DetectionResult` method
    - Add background subtraction using MOG2, KNN, or GMG
    - Add optional Canny edge detection
    - Add contour detection using `cv2.findContours`
    - Add contour filtering by area thresholds
    - Compute bounding boxes and centroids for valid contours
    - Update background model with each frame
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 4.2 Write property test for contour area filtering
    - **Property 4: Contour Area Filtering**
    - **Validates: Requirements 2.4**

  - [ ]* 4.3 Write property test for detection result completeness
    - **Property 5: Detection Result Completeness**
    - **Validates: Requirements 2.5**

  - [ ]* 4.4 Write property test for background model updates
    - **Property 6: Background Model Updates**
    - **Validates: Requirements 2.6**

  - [ ]* 4.5 Write unit tests for Detector
    - Test background subtraction with static and moving objects
    - Test contour filtering with known shapes
    - Test edge cases: no objects, overlapping objects, frame boundaries
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 5. Checkpoint - Ensure preprocessing and detection tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement Tracker Module
  - [x] 6.1 Create Tracker class with centroid tracking
    - Implement `__init__(config: TrackerConfig)` to initialize tracking state
    - Implement `update(detections: DetectionResult) -> TrackingResult` method
    - Add centroid computation from detection bounding boxes
    - Add centroid matching using Euclidean distance
    - Add new ID assignment for unmatched detections
    - Add trajectory history management with configurable length
    - Add disappeared counter and track removal after max_disappeared_frames
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 6.2 Write property test for unique object identifiers
    - **Property 7: Unique Object Identifiers**
    - **Validates: Requirements 3.1**

  - [ ]* 6.3 Write property test for ID persistence across frames
    - **Property 8: ID Persistence Across Frames**
    - **Validates: Requirements 3.2**

  - [ ]* 6.4 Write property test for trajectory history length
    - **Property 9: Trajectory History Length**
    - **Validates: Requirements 3.4**

  - [ ]* 6.5 Write property test for tracking result completeness
    - **Property 10: Tracking Result Completeness**
    - **Validates: Requirements 3.5**

  - [ ]* 6.6 Write property test for separate identity maintenance
    - **Property 11: Separate Identity Maintenance**
    - **Validates: Requirements 3.6**

  - [ ]* 6.7 Write unit tests for Tracker
    - Test ID assignment for new objects
    - Test ID persistence for stationary objects
    - Test object removal after 30 frames disappeared
    - Test trajectory history management
    - Test edge cases: objects entering/leaving frame, rapid movement
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 7. Implement Calibrator Module
  - [x] 7.1 Create Calibrator class with camera calibration
    - Implement `__init__(config: CalibratorConfig)` to initialize parameters
    - Implement `calibrate(images: List[np.ndarray], chessboard_size: Tuple[int, int]) -> CalibrationParameters` method
    - Add chessboard corner detection using `cv2.findChessboardCorners`
    - Add corner refinement using `cv2.cornerSubPix`
    - Add camera calibration using `cv2.calibrateCamera`
    - Add optional homography computation for perspective transform
    - Implement `save_calibration(params: CalibrationParameters, filepath: str)` to save to JSON
    - Implement `load_calibration(filepath: str) -> CalibrationParameters` to load from JSON
    - Implement `undistort(frame: np.ndarray, params: CalibrationParameters) -> np.ndarray` for distortion correction
    - Implement `apply_perspective_transform(frame: np.ndarray, params: CalibrationParameters) -> np.ndarray` for homography
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

  - [ ]* 7.2 Write property test for chessboard corner detection
    - **Property 12: Chessboard Corner Detection**
    - **Validates: Requirements 4.1**

  - [ ]* 7.3 Write property test for calibration parameter computation
    - **Property 13: Calibration Parameter Computation**
    - **Validates: Requirements 4.2, 4.3**

  - [ ]* 7.4 Write property test for calibration persistence round trip
    - **Property 14: Calibration Persistence Round Trip**
    - **Validates: Requirements 4.4, 4.5**

  - [ ]* 7.5 Write property test for frame undistortion application
    - **Property 15: Frame Undistortion Application**
    - **Validates: Requirements 4.6**

  - [ ]* 7.6 Write unit tests for Calibrator
    - Test calibration with standard chessboard images
    - Test calibration file save/load with known parameters
    - Test undistortion with known distortion patterns
    - Test edge cases: insufficient images, invalid chessboard size, corrupted file
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 8. Implement Speed Estimator Module
  - [-] 8.1 Create SpeedEstimator class with velocity calculation
    - Implement `__init__(config: SpeedEstimatorConfig)` to initialize parameters
    - Implement `estimate_speeds(tracking: TrackingResult, calibration: Optional[CalibrationParameters], fps: float) -> List[SpeedResult]` method
    - Add pixel displacement calculation from trajectory points
    - Add temporal scaling using fps to convert to velocity
    - Add spatial scaling using pixels_per_meter when calibration available
    - Add instantaneous speed calculation from last two trajectory points
    - Add average speed calculation over averaging window
    - Add unit conversion support (m/s, km/h, mph)
    - Add fallback to px/s when calibration unavailable
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [ ]* 8.2 Write property test for velocity calculation from trajectory
    - **Property 16: Velocity Calculation from Trajectory**
    - **Validates: Requirements 5.1, 5.3**

  - [ ]* 8.3 Write property test for calibrated speed scaling
    - **Property 17: Calibrated Speed Scaling**
    - **Validates: Requirements 5.2, 5.5**

  - [ ]* 8.4 Write property test for average speed window
    - **Property 18: Average Speed Window**
    - **Validates: Requirements 5.4**

  - [ ]* 8.5 Write property test for uncalibrated speed fallback
    - **Property 19: Uncalibrated Speed Fallback**
    - **Validates: Requirements 5.6**

  - [ ]* 8.6 Write unit tests for Speed Estimator
    - Test speed calculation with known trajectories
    - Test unit conversion with known calibration parameters
    - Test averaging over specific window sizes
    - Test edge cases: single-point trajectory, zero displacement, missing calibration
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [~] 9. Checkpoint - Ensure all component tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement Pipeline Orchestrator
  - [~] 10.1 Create PipelineOrchestrator class with component coordination
    - Implement `__init__(config: PipelineConfig)` to initialize all enabled components
    - Implement `process_frame(frame: np.ndarray, frame_number: int) -> PipelineResult` method
    - Add sequential component execution: preprocessor → detector → tracker → speed estimator
    - Add frame annotation with bounding boxes, IDs, and speeds
    - Add per-component timing measurement
    - Add error handling with graceful degradation (log and continue)
    - Implement `process_video(video_path: str, output_path: Optional[str]) -> PipelineSummary` method
    - Add video file reading using `cv2.VideoCapture`
    - Add optional output video writing using `cv2.VideoWriter`
    - Add summary statistics generation
    - _Requirements: 6.3, 6.4, 6.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

  - [ ]* 10.2 Write property test for component skipping
    - **Property 21: Component Skipping**
    - **Validates: Requirements 6.6**

  - [ ]* 10.3 Write property test for sequential frame reading
    - **Property 23: Sequential Frame Reading**
    - **Validates: Requirements 8.1**

  - [ ]* 10.4 Write property test for annotated frame content
    - **Property 24: Annotated Frame Content**
    - **Validates: Requirements 8.4**

  - [ ]* 10.5 Write property test for summary report completeness
    - **Property 25: Summary Report Completeness**
    - **Validates: Requirements 8.5**

  - [ ]* 10.6 Write integration tests for Pipeline Orchestrator
    - Test full pipeline with sample video
    - Test component chaining and data flow
    - Test error propagation and graceful degradation
    - Test configuration-based component enabling/disabling
    - Test performance target: 15+ FPS for 1920x1080 video
    - _Requirements: 6.3, 6.4, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_

- [ ] 11. Implement Error Handling and Logging
  - [~] 11.1 Add comprehensive error handling to all components
    - Add try-catch blocks around critical operations in each component
    - Add component-specific error handling (invalid input, algorithm failures)
    - Add resource management (memory monitoring, file handle cleanup)
    - Implement structured logging with JSON format
    - Add log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    - Add context to logs: frame_number, object_id, component_name, processing_time_ms
    - _Requirements: 6.4, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [ ]* 11.2 Write unit tests for error handling
    - Test invalid video file handling with descriptive error messages
    - Test component failure logging with stack traces
    - Test memory warning at 80% usage
    - Test log level filtering
    - Test file logging when enabled
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

- [ ] 12. Implement FastAPI Backend
  - [~] 12.1 Create FastAPI application with core endpoints
    - Set up FastAPI app with CORS middleware
    - Implement `POST /api/v1/process/frame` endpoint for single frame processing
    - Implement `POST /api/v1/process/video` endpoint for async video processing
    - Implement `GET /api/v1/process/video/{task_id}` endpoint for status polling
    - Add request validation using Pydantic models
    - Add base64 image decoding and encoding
    - Add multipart file upload handling
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [~] 12.2 Create calibration endpoints
    - Implement `POST /api/v1/calibration/calibrate` endpoint
    - Implement `GET /api/v1/calibration/status` endpoint
    - Implement `POST /api/v1/calibration/load` endpoint
    - _Requirements: 7.6_

  - [~] 12.3 Create configuration endpoints
    - Implement `GET /api/v1/config` endpoint to retrieve current configuration
    - Implement `PUT /api/v1/config` endpoint to update configuration
    - _Requirements: 7.5_

  - [~] 12.4 Add async task management for long-running video processing
    - Implement background worker for video processing tasks
    - Add task ID generation and storage
    - Add task status tracking (pending, processing, completed, failed)
    - Add temporary result storage with task_id key
    - _Requirements: 7.7_

  - [ ]* 12.5 Write property test for API JSON response format
    - **Property 22: API JSON Response Format**
    - **Validates: Requirements 7.3**

  - [ ]* 12.6 Write API integration tests
    - Test all endpoints with valid requests
    - Test input format handling (base64, multipart)
    - Test async task creation and retrieval
    - Test error responses for invalid inputs (400, 404, 422, 500, 503)
    - Test request timeout handling
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7_

- [~] 13. Create custom Hypothesis strategies for property-based testing
  - Create `video_frames()` strategy for generating random video frames
  - Create `detection_results()` strategy for generating random detections
  - Create `trajectories()` strategy for generating random object trajectories
  - Create `chessboard_images()` strategy for generating synthetic calibration images
  - Create `pipeline_configs()` strategy for generating valid configurations
  - Configure all property tests to run minimum 100 iterations
  - Add feature tags to all property tests: `# Feature: opencv-video-processing-pipeline, Property N: ...`
  - _Requirements: All requirements (testing infrastructure)_

- [ ] 14. Final integration and documentation
  - [~] 14.1 Create example configuration files
    - Create `config/default.yaml` with sensible defaults
    - Create `config/high_performance.yaml` for speed-optimized settings
    - Create `config/high_quality.yaml` for quality-optimized settings
    - _Requirements: 9.1, 9.6_

  - [~] 14.2 Create example usage scripts
    - Create `examples/process_video.py` demonstrating video processing
    - Create `examples/calibrate_camera.py` demonstrating calibration workflow
    - Create `examples/api_client.py` demonstrating API usage
    - _Requirements: 8.1, 8.2, 8.3_

  - [~] 14.3 Create README with setup and usage instructions
    - Document installation steps
    - Document configuration options
    - Document API endpoints
    - Document example usage
    - _Requirements: All requirements (documentation)_

- [~] 15. Final checkpoint - Run full test suite and verify all requirements
  - Run all unit tests and property-based tests
  - Verify test coverage meets goals (85% line coverage, 80% branch coverage)
  - Run performance benchmarks to verify FPS targets
  - Test with sample videos to verify end-to-end functionality
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation uses Python with OpenCV, NumPy, FastAPI, and Hypothesis for property-based testing
- All property tests should include feature tags for traceability to the design document
