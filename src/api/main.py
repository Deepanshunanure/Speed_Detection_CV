"""FastAPI application for video processing pipeline"""
import base64
import logging
import uuid
import asyncio
from typing import Optional, Dict, List, Any
from datetime import datetime
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.api.orchestrator import PipelineOrchestrator
from src.config.manager import ConfigurationManager
from src.config.models import PipelineConfig
from src.models import PipelineResult, PipelineSummary


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="OpenCV Video Processing Pipeline API",
    description="REST API for video processing with classical computer vision techniques",
    version="1.0.0"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory storage for async video processing tasks
video_tasks: Dict[str, Dict[str, Any]] = {}


# Pydantic models for request/response validation

class FrameProcessRequest(BaseModel):
    """Request model for single frame processing"""
    image: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    config: Optional[Dict] = Field(None, description="Optional pipeline configuration overrides")
    include_annotated_frame: bool = Field(True, description="Include annotated frame in response")
    
    @field_validator('image')
    @classmethod
    def validate_base64(cls, v):
        """Validate base64 encoding"""
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("Invalid base64 encoding")
        return v


class FrameProcessResponse(BaseModel):
    """Response model for single frame processing"""
    frame_number: int
    timestamp: float
    annotated_frame: Optional[str] = Field(None, description="Base64-encoded annotated frame")
    detections: Optional[Dict] = None
    tracking: Optional[Dict] = None
    speeds: Optional[List[Dict]] = None
    processing_time_ms: float
    component_times: Dict[str, float]


class VideoProcessResponse(BaseModel):
    """Response model for async video processing"""
    task_id: str = Field(..., description="Unique task identifier for status polling")
    status: str = Field(..., description="Task status: queued, processing, completed, failed")
    message: str


class VideoStatusResponse(BaseModel):
    """Response model for video processing status"""
    task_id: str
    status: str = Field(..., description="Task status: queued, processing, completed, failed")
    progress: Optional[float] = Field(None, description="Processing progress (0.0 to 1.0)")
    summary: Optional[Dict] = Field(None, description="Processing summary (available when completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error_code: str
    message: str
    details: Optional[Dict] = None
    timestamp: str


class CalibrationRequest(BaseModel):
    """Request model for camera calibration"""
    images: List[str] = Field(..., description="List of base64-encoded calibration images")
    chessboard_size: List[int] = Field(..., description="Chessboard inner corners [cols, rows]")
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v):
        """Validate images list"""
        if len(v) < 10:
            raise ValueError("At least 10 calibration images required")
        for img in v:
            try:
                base64.b64decode(img)
            except Exception:
                raise ValueError("Invalid base64 encoding in images")
        return v
    
    @field_validator('chessboard_size')
    @classmethod
    def validate_chessboard_size(cls, v):
        """Validate chessboard size"""
        if len(v) != 2:
            raise ValueError("Chessboard size must be [cols, rows]")
        if v[0] < 3 or v[1] < 3:
            raise ValueError("Chessboard size must be at least 3x3")
        return v


class CalibrationResponse(BaseModel):
    """Response model for calibration"""
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    homography_matrix: Optional[List[List[float]]] = None
    pixels_per_meter: Optional[float] = None
    calibration_error: float
    calibration_date: str
    message: str


class CalibrationStatusResponse(BaseModel):
    """Response model for calibration status"""
    calibrated: bool
    calibration_date: Optional[str] = None
    calibration_error: Optional[float] = None
    pixels_per_meter: Optional[float] = None
    message: str


class CalibrationLoadRequest(BaseModel):
    """Request model for loading calibration"""
    filepath: str = Field(..., description="Path to calibration JSON file")


class CalibrationLoadResponse(BaseModel):
    """Response model for loading calibration"""
    camera_matrix: List[List[float]]
    distortion_coefficients: List[float]
    homography_matrix: Optional[List[List[float]]] = None
    pixels_per_meter: Optional[float] = None
    calibration_error: float
    calibration_date: str
    message: str


class ConfigResponse(BaseModel):
    """Response model for configuration retrieval"""
    pipeline: Dict = Field(..., description="Pipeline configuration")
    preprocessor: Dict = Field(..., description="Preprocessor configuration")
    detector: Dict = Field(..., description="Detector configuration")
    tracker: Dict = Field(..., description="Tracker configuration")
    calibrator: Dict = Field(..., description="Calibrator configuration")
    speed_estimator: Dict = Field(..., description="Speed estimator configuration")
    logging: Dict = Field(..., description="Logging configuration")


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration update"""
    pipeline: Optional[Dict] = Field(None, description="Pipeline configuration")
    preprocessor: Optional[Dict] = Field(None, description="Preprocessor configuration")
    detector: Optional[Dict] = Field(None, description="Detector configuration")
    tracker: Optional[Dict] = Field(None, description="Tracker configuration")
    calibrator: Optional[Dict] = Field(None, description="Calibrator configuration")
    speed_estimator: Optional[Dict] = Field(None, description="Speed estimator configuration")
    logging: Optional[Dict] = Field(None, description="Logging configuration")


class ConfigUpdateResponse(BaseModel):
    """Response model for configuration update"""
    message: str
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    updated_config: Dict = Field(..., description="Updated configuration")


# Helper functions

def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image
    
    Args:
        base64_str: Base64-encoded image string
        
    Returns:
        OpenCV image as numpy array (BGR format)
        
    Raises:
        ValueError: If decoding fails
    """
    try:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(base64_str)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise ValueError(f"Invalid image data: {e}")


def encode_image_to_base64(img: np.ndarray, format: str = '.jpg') -> str:
    """
    Encode OpenCV image to base64 string
    
    Args:
        img: OpenCV image as numpy array
        format: Image format ('.jpg' or '.png')
        
    Returns:
        Base64-encoded image string
    """
    try:
        # Encode image to bytes
        success, buffer = cv2.imencode(format, img)
        if not success:
            raise ValueError("Failed to encode image")
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    except Exception as e:
        logger.error(f"Failed to encode image to base64: {e}")
        raise ValueError(f"Image encoding failed: {e}")


def pipeline_result_to_dict(result: PipelineResult, include_frame: bool = True) -> Dict:
    """
    Convert PipelineResult to JSON-serializable dictionary
    
    Args:
        result: Pipeline processing result
        include_frame: Whether to include annotated frame
        
    Returns:
        Dictionary representation
    """
    response_dict = {
        "frame_number": result.frame_number,
        "timestamp": result.timestamp,
        "processing_time_ms": result.processing_time_ms,
        "component_times": result.component_times
    }
    
    # Add annotated frame if requested
    if include_frame and result.annotated_frame is not None:
        response_dict["annotated_frame"] = encode_image_to_base64(result.annotated_frame)
    
    # Add detections
    if result.detections is not None:
        response_dict["detections"] = {
            "frame_number": result.detections.frame_number,
            "num_objects": len(result.detections.bounding_boxes),
            "bounding_boxes": [
                {
                    "x": bbox.x,
                    "y": bbox.y,
                    "width": bbox.width,
                    "height": bbox.height,
                    "area": bbox.area,
                    "centroid": bbox.centroid
                }
                for bbox in result.detections.bounding_boxes
            ]
        }
    
    # Add tracking
    if result.tracking is not None:
        response_dict["tracking"] = {
            "frame_number": result.tracking.frame_number,
            "num_tracked_objects": len(result.tracking.tracked_objects),
            "tracked_objects": [
                {
                    "object_id": obj.object_id,
                    "position": obj.position,
                    "bounding_box": {
                        "x": obj.bounding_box.x,
                        "y": obj.bounding_box.y,
                        "width": obj.bounding_box.width,
                        "height": obj.bounding_box.height
                    },
                    "age": obj.age,
                    "trajectory_length": len(obj.trajectory)
                }
                for obj in result.tracking.tracked_objects
            ]
        }
    
    # Add speeds
    if result.speeds is not None:
        response_dict["speeds"] = [
            {
                "object_id": speed.object_id,
                "instantaneous_speed": speed.instantaneous_speed,
                "average_speed": speed.average_speed,
                "unit": speed.unit,
                "calibrated": speed.calibrated,
                "confidence": speed.confidence
            }
            for speed in result.speeds
        ]
    
    return response_dict


def pipeline_summary_to_dict(summary: PipelineSummary) -> Dict:
    """
    Convert PipelineSummary to JSON-serializable dictionary
    
    Args:
        summary: Pipeline summary statistics
        
    Returns:
        Dictionary representation
    """
    return {
        "total_frames": summary.total_frames,
        "processed_frames": summary.processed_frames,
        "average_fps": summary.average_fps,
        "total_objects_detected": summary.total_objects_detected,
        "unique_objects_tracked": summary.unique_objects_tracked,
        "average_speed": summary.average_speed,
        "max_speed": summary.max_speed,
        "processing_errors": summary.processing_errors,
        "component_statistics": {
            name: {
                "component_name": stats.component_name,
                "average_time_ms": stats.average_time_ms,
                "max_time_ms": stats.max_time_ms,
                "error_count": stats.error_count
            }
            for name, stats in summary.component_statistics.items()
        }
    }


def config_to_dict(config_manager: ConfigurationManager) -> Dict:
    """
    Convert configuration to JSON-serializable dictionary
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary representation of all configurations
    """
    pipeline_config = config_manager.get_pipeline_config()
    
    return {
        "pipeline": {
            "enabled_components": pipeline_config.enabled_components
        },
        "preprocessor": {
            "target_resolution": list(pipeline_config.preprocessor.target_resolution),
            "noise_reduction_method": pipeline_config.preprocessor.noise_reduction_method,
            "noise_reduction_kernel_size": pipeline_config.preprocessor.noise_reduction_kernel_size,
            "normalize_intensity": pipeline_config.preprocessor.normalize_intensity
        },
        "detector": {
            "background_subtraction_method": pipeline_config.detector.background_subtraction_method,
            "background_learning_rate": pipeline_config.detector.background_learning_rate,
            "edge_detection_enabled": pipeline_config.detector.edge_detection_enabled,
            "canny_threshold1": pipeline_config.detector.canny_threshold1,
            "canny_threshold2": pipeline_config.detector.canny_threshold2,
            "min_contour_area": pipeline_config.detector.min_contour_area,
            "max_contour_area": pipeline_config.detector.max_contour_area
        },
        "tracker": {
            "max_tracking_distance": pipeline_config.tracker.max_tracking_distance,
            "max_disappeared_frames": pipeline_config.tracker.max_disappeared_frames,
            "trajectory_history_length": pipeline_config.tracker.trajectory_history_length
        },
        "calibrator": {
            "calibration_file_path": pipeline_config.calibrator.calibration_file_path,
            "chessboard_size": list(pipeline_config.calibrator.chessboard_size),
            "square_size_mm": pipeline_config.calibrator.square_size_mm,
            "perspective_transform_enabled": pipeline_config.calibrator.perspective_transform_enabled
        },
        "speed_estimator": {
            "averaging_window_frames": pipeline_config.speed_estimator.averaging_window_frames,
            "min_trajectory_length": pipeline_config.speed_estimator.min_trajectory_length,
            "output_unit": pipeline_config.speed_estimator.output_unit
        },
        "logging": {
            "level": pipeline_config.logging.level,
            "file_path": pipeline_config.logging.file_path,
            "log_to_file": pipeline_config.logging.log_to_file
        }
    }


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "OpenCV Video Processing Pipeline API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "frame_processing": "/api/v1/process/frame",
            "video_processing": "/api/v1/process/video",
            "video_status": "/api/v1/process/video/{task_id}",
            "calibration_calibrate": "/api/v1/calibration/calibrate",
            "calibration_status": "/api/v1/calibration/status",
            "calibration_load": "/api/v1/calibration/load",
            "config_get": "/api/v1/config",
            "config_update": "/api/v1/config"
        }
    }


@app.post("/api/v1/process/frame", response_model=FrameProcessResponse)
async def process_frame(request: FrameProcessRequest):
    """
    Process a single frame through the pipeline
    
    Args:
        request: Frame processing request with base64-encoded image
        
    Returns:
        Processing results with detections, tracking, and speeds
        
    Raises:
        HTTPException: If processing fails
    """
    try:
        logger.info("Received frame processing request")
        
        # Decode image
        frame = decode_base64_image(request.image)
        logger.info(f"Decoded image with shape: {frame.shape}")
        
        # Load configuration (use default or override)
        if request.config:
            # TODO: Implement config override logic
            config_manager = ConfigurationManager("config/default.yaml")
        else:
            config_manager = ConfigurationManager("config/default.yaml")
        
        pipeline_config = config_manager.get_pipeline_config()
        
        # Initialize pipeline
        orchestrator = PipelineOrchestrator(pipeline_config)
        
        # Process frame
        result = orchestrator.process_frame(frame, frame_number=0)
        
        # Convert to response format
        response_dict = pipeline_result_to_dict(result, include_frame=request.include_annotated_frame)
        
        logger.info(f"Frame processed successfully in {result.processing_time_ms:.2f}ms")
        
        return FrameProcessResponse(**response_dict)
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Frame processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/v1/process/video", response_model=VideoProcessResponse)
async def process_video(
    video_file: UploadFile = File(..., description="Video file to process"),
    config: Optional[str] = Form(None, description="Optional JSON configuration")
):
    """
    Process a video file asynchronously
    
    Args:
        video_file: Uploaded video file
        config: Optional JSON configuration string
        
    Returns:
        Task ID for status polling
        
    Raises:
        HTTPException: If video upload fails
    """
    try:
        logger.info(f"Received video processing request: {video_file.filename}")
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Save uploaded video to temporary file
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        input_path = os.path.join(temp_dir, f"{task_id}_input.mp4")
        output_path = os.path.join(temp_dir, f"{task_id}_output.mp4")
        
        # Write uploaded file
        with open(input_path, "wb") as f:
            content = await video_file.read()
            f.write(content)
        
        logger.info(f"Video saved to {input_path}, size: {len(content)} bytes")
        
        # Initialize task status
        video_tasks[task_id] = {
            "status": "queued",
            "input_path": input_path,
            "output_path": output_path,
            "progress": 0.0,
            "summary": None,
            "error": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Start async processing
        asyncio.create_task(process_video_async(task_id, input_path, output_path, config))
        
        return VideoProcessResponse(
            task_id=task_id,
            status="queued",
            message="Video processing task created successfully"
        )
        
    except Exception as e:
        logger.error(f"Video upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video upload failed: {str(e)}")


@app.get("/api/v1/process/video/{task_id}", response_model=VideoStatusResponse)
async def get_video_status(task_id: str):
    """
    Get status of video processing task
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        Task status and results (if completed)
        
    Raises:
        HTTPException: If task not found
    """
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = video_tasks[task_id]
    
    return VideoStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        summary=task.get("summary"),
        error=task.get("error")
    )


# Async video processing function

async def process_video_async(task_id: str, input_path: str, output_path: str, config: Optional[str]):
    """
    Process video asynchronously in background
    
    Args:
        task_id: Unique task identifier
        input_path: Path to input video file
        output_path: Path to save output video
        config: Optional JSON configuration
    """
    try:
        logger.info(f"Starting async video processing for task {task_id}")
        
        # Update status to processing
        video_tasks[task_id]["status"] = "processing"
        
        # Load configuration
        config_manager = ConfigurationManager("config/default.yaml")
        pipeline_config = config_manager.get_pipeline_config()
        
        # Initialize pipeline
        orchestrator = PipelineOrchestrator(pipeline_config)
        
        # Process video (run in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None,
            orchestrator.process_video,
            input_path,
            output_path
        )
        
        # Convert summary to dict
        summary_dict = pipeline_summary_to_dict(summary)
        
        # Update task status
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["progress"] = 1.0
        video_tasks[task_id]["summary"] = summary_dict
        video_tasks[task_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Video processing completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed for task {task_id}: {e}", exc_info=True)
        video_tasks[task_id]["status"] = "failed"
        video_tasks[task_id]["error"] = str(e)


# Calibration endpoints

@app.post("/api/v1/calibration/calibrate", response_model=CalibrationResponse)
async def calibrate_camera(request: CalibrationRequest):
    """
    Calibrate camera from chessboard images
    
    Args:
        request: Calibration request with base64-encoded images and chessboard size
        
    Returns:
        Calibration parameters including camera matrix and distortion coefficients
        
    Raises:
        HTTPException: If calibration fails
    """
    try:
        logger.info(f"Received calibration request with {len(request.images)} images")
        
        # Decode images
        images = []
        for i, base64_img in enumerate(request.images):
            try:
                img = decode_base64_image(base64_img)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to decode image {i}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to decode image {i}: {str(e)}"
                )
        
        logger.info(f"Decoded {len(images)} images successfully")
        
        # Load configuration
        config_manager = ConfigurationManager("config/default.yaml")
        calibrator_config = config_manager.get_calibrator_config()
        
        # Initialize calibrator
        from src.components.calibrator import Calibrator
        calibrator = Calibrator(calibrator_config)
        
        # Perform calibration
        chessboard_size = tuple(request.chessboard_size)
        params = calibrator.calibrate(images, chessboard_size)
        
        # Save calibration to default location
        calibrator.save_calibration(params, "calibration.json")
        
        logger.info("Calibration completed successfully")
        
        # Convert to response format
        response = CalibrationResponse(
            camera_matrix=params.camera_matrix.tolist(),
            distortion_coefficients=params.distortion_coefficients.tolist(),
            homography_matrix=params.homography_matrix.tolist() if params.homography_matrix is not None else None,
            pixels_per_meter=float(params.pixels_per_meter) if params.pixels_per_meter is not None else None,
            calibration_error=float(params.calibration_error),
            calibration_date=params.calibration_date,
            message="Camera calibration completed successfully"
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Calibration validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")


@app.get("/api/v1/calibration/status", response_model=CalibrationStatusResponse)
async def get_calibration_status():
    """
    Get current calibration status
    
    Returns:
        Calibration status including whether camera is calibrated and parameters
        
    Raises:
        HTTPException: If status check fails
    """
    try:
        import os
        
        # Check if calibration file exists
        calibration_file = "calibration.json"
        
        if not os.path.exists(calibration_file):
            return CalibrationStatusResponse(
                calibrated=False,
                message="No calibration data available"
            )
        
        # Load calibration parameters
        config_manager = ConfigurationManager("config/default.yaml")
        calibrator_config = config_manager.get_calibrator_config()
        
        from src.components.calibrator import Calibrator
        calibrator = Calibrator(calibrator_config)
        
        try:
            params = calibrator.load_calibration(calibration_file)
            
            return CalibrationStatusResponse(
                calibrated=True,
                calibration_date=params.calibration_date,
                calibration_error=float(params.calibration_error),
                pixels_per_meter=float(params.pixels_per_meter) if params.pixels_per_meter is not None else None,
                message="Camera is calibrated"
            )
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return CalibrationStatusResponse(
                calibrated=False,
                message=f"Calibration file exists but failed to load: {str(e)}"
            )
        
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/api/v1/calibration/load", response_model=CalibrationLoadResponse)
async def load_calibration(request: CalibrationLoadRequest):
    """
    Load calibration parameters from file
    
    Args:
        request: Load request with filepath
        
    Returns:
        Loaded calibration parameters
        
    Raises:
        HTTPException: If loading fails
    """
    try:
        logger.info(f"Loading calibration from {request.filepath}")
        
        # Load configuration
        config_manager = ConfigurationManager("config/default.yaml")
        calibrator_config = config_manager.get_calibrator_config()
        
        # Initialize calibrator
        from src.components.calibrator import Calibrator
        calibrator = Calibrator(calibrator_config)
        
        # Load calibration
        params = calibrator.load_calibration(request.filepath)
        
        logger.info("Calibration loaded successfully")
        
        # Convert to response format
        response = CalibrationLoadResponse(
            camera_matrix=params.camera_matrix.tolist(),
            distortion_coefficients=params.distortion_coefficients.tolist(),
            homography_matrix=params.homography_matrix.tolist() if params.homography_matrix is not None else None,
            pixels_per_meter=float(params.pixels_per_meter) if params.pixels_per_meter is not None else None,
            calibration_error=float(params.calibration_error),
            calibration_date=params.calibration_date,
            message="Calibration loaded successfully"
        )
        
        return response
        
    except FileNotFoundError:
        logger.error(f"Calibration file not found: {request.filepath}")
        raise HTTPException(status_code=404, detail=f"Calibration file not found: {request.filepath}")
    except ValueError as e:
        logger.error(f"Invalid calibration file: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load calibration: {str(e)}")


# Configuration endpoints

@app.get("/api/v1/config", response_model=ConfigResponse)
async def get_config():
    """
    Get current pipeline configuration
    
    Returns:
        Current configuration for all pipeline components
        
    Raises:
        HTTPException: If configuration retrieval fails
    """
    try:
        logger.info("Retrieving current configuration")
        
        # Load configuration from default location
        config_manager = ConfigurationManager("config/default.yaml")
        
        # Convert to dictionary
        config_dict = config_to_dict(config_manager)
        
        logger.info("Configuration retrieved successfully")
        
        return ConfigResponse(**config_dict)
        
    except Exception as e:
        logger.error(f"Failed to retrieve configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")


@app.put("/api/v1/config", response_model=ConfigUpdateResponse)
async def update_config(request: ConfigUpdateRequest):
    """
    Update pipeline configuration
    
    Args:
        request: Configuration update request with new parameters
        
    Returns:
        Validation results and updated configuration
        
    Raises:
        HTTPException: If configuration update fails
    """
    try:
        logger.info("Updating pipeline configuration")
        
        # Load current configuration
        config_manager = ConfigurationManager("config/default.yaml")
        current_config = config_manager._config_data
        
        # Update configuration with provided values
        if request.pipeline is not None:
            current_config['pipeline'] = {**current_config.get('pipeline', {}), **request.pipeline}
        
        if request.preprocessor is not None:
            current_config['preprocessor'] = {**current_config.get('preprocessor', {}), **request.preprocessor}
        
        if request.detector is not None:
            current_config['detector'] = {**current_config.get('detector', {}), **request.detector}
        
        if request.tracker is not None:
            current_config['tracker'] = {**current_config.get('tracker', {}), **request.tracker}
        
        if request.calibrator is not None:
            current_config['calibrator'] = {**current_config.get('calibrator', {}), **request.calibrator}
        
        if request.speed_estimator is not None:
            current_config['speed_estimator'] = {**current_config.get('speed_estimator', {}), **request.speed_estimator}
        
        if request.logging is not None:
            current_config['logging'] = {**current_config.get('logging', {}), **request.logging}
        
        # Save updated configuration to file
        import yaml
        with open("config/default.yaml", 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        
        logger.info("Configuration file updated")
        
        # Reload configuration manager to validate
        updated_manager = ConfigurationManager("config/default.yaml")
        
        # Validate configuration
        warnings = updated_manager.validate()
        
        if warnings:
            logger.warning(f"Configuration validation warnings: {warnings}")
        
        # Get updated configuration
        updated_config_dict = config_to_dict(updated_manager)
        
        logger.info("Configuration updated successfully")
        
        return ConfigUpdateResponse(
            message="Configuration updated successfully",
            warnings=warnings,
            updated_config=updated_config_dict
        )
        
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


# Health check endpoint

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in video_tasks.values() if t["status"] == "processing"])
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
