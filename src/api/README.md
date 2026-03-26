# FastAPI Video Processing Pipeline

REST API for the OpenCV video processing pipeline with classical computer vision techniques.

## Features

- **Single Frame Processing**: Process individual frames with base64 encoding
- **Async Video Processing**: Upload and process complete videos asynchronously
- **Status Polling**: Track video processing progress with task IDs
- **CORS Support**: Cross-origin requests enabled for web integration
- **Request Validation**: Pydantic models for robust input validation
- **Error Handling**: Comprehensive error responses with detailed messages

## API Endpoints

### Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### Health Check
```
GET /health
```
Returns API health status and active task count.

### Process Single Frame
```
POST /api/v1/process/frame
```

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "config": {},  // Optional configuration overrides
  "include_annotated_frame": true
}
```

**Response:**
```json
{
  "frame_number": 0,
  "timestamp": 0.0,
  "annotated_frame": "base64_encoded_result",
  "detections": {
    "num_objects": 5,
    "bounding_boxes": [...]
  },
  "tracking": {
    "num_tracked_objects": 3,
    "tracked_objects": [...]
  },
  "speeds": [...],
  "processing_time_ms": 25.5,
  "component_times": {
    "preprocessor": 5.0,
    "detector": 10.0,
    "tracker": 8.0,
    "speed_estimator": 2.5
  }
}
```

### Process Video (Async)
```
POST /api/v1/process/video
```

**Request:** Multipart form data
- `video_file`: Video file (MP4, AVI, etc.)
- `config`: Optional JSON configuration string

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "Video processing task created successfully"
}
```

### Get Video Processing Status
```
GET /api/v1/process/video/{task_id}
```

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "completed",
  "progress": 1.0,
  "summary": {
    "total_frames": 300,
    "processed_frames": 300,
    "average_fps": 28.5,
    "total_objects_detected": 150,
    "unique_objects_tracked": 25,
    "average_speed": 5.5,
    "max_speed": 15.0,
    "processing_errors": [],
    "component_statistics": {...}
  }
}
```

**Status Values:**
- `queued`: Task created, waiting to start
- `processing`: Video is being processed
- `completed`: Processing finished successfully
- `failed`: Processing encountered an error

### Calibrate Camera
```
POST /api/v1/calibration/calibrate
```

**Request Body:**
```json
{
  "images": ["base64_image_1", "base64_image_2", ...],
  "chessboard_size": [9, 6]
}
```

**Requirements:**
- At least 10 calibration images with chessboard pattern
- Chessboard size: inner corners [columns, rows]
- Minimum chessboard size: 3x3

**Response:**
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "homography_matrix": [[...], [...], [...]],
  "pixels_per_meter": 100.5,
  "calibration_error": 0.45,
  "calibration_date": "2024-01-01T12:00:00",
  "message": "Camera calibration completed successfully"
}
```

### Get Calibration Status
```
GET /api/v1/calibration/status
```

**Response:**
```json
{
  "calibrated": true,
  "calibration_date": "2024-01-01T12:00:00",
  "calibration_error": 0.45,
  "pixels_per_meter": 100.5,
  "message": "Camera is calibrated"
}
```

### Load Calibration from File
```
POST /api/v1/calibration/load
```

**Request Body:**
```json
{
  "filepath": "calibration.json"
}
```

**Response:**
```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distortion_coefficients": [k1, k2, p1, p2, k3],
  "homography_matrix": [[...], [...], [...]],
  "pixels_per_meter": 100.5,
  "calibration_error": 0.45,
  "calibration_date": "2024-01-01T12:00:00",
  "message": "Calibration loaded successfully"
}
```

## Running the API Server

### Development Mode
```bash
# Using uvicorn with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Or run directly
python src/api/main.py
```

### Production Mode
```bash
# Using uvicorn with multiple workers
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Usage Examples

### Python with requests

#### Process a frame
```python
import requests
import base64

# Process a frame
with open("frame.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/v1/process/frame",
    json={"image": img_base64}
)

result = response.json()
print(f"Processing time: {result['processing_time_ms']}ms")
```

#### Calibrate camera
```python
import requests
import base64
import glob

# Load calibration images
images = []
for img_file in glob.glob("calibration_images/*.jpg")[:15]:
    with open(img_file, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()
        images.append(img_base64)

# Calibrate
response = requests.post(
    "http://localhost:8000/api/v1/calibration/calibrate",
    json={
        "images": images,
        "chessboard_size": [9, 6]
    }
)

result = response.json()
print(f"Calibration error: {result['calibration_error']:.4f}")
print(f"Pixels per meter: {result['pixels_per_meter']:.2f}")
```

#### Check calibration status
```python
response = requests.get("http://localhost:8000/api/v1/calibration/status")
status = response.json()

if status['calibrated']:
    print(f"Camera calibrated on {status['calibration_date']}")
else:
    print("Camera not calibrated")
```

### cURL
```bash
# Health check
curl http://localhost:8000/health

# Process video
curl -X POST http://localhost:8000/api/v1/process/video \
  -F "video_file=@demo_input.mp4"

# Check status
curl http://localhost:8000/api/v1/process/video/{task_id}
```

### JavaScript/Fetch
```javascript
// Process frame
const response = await fetch('http://localhost:8000/api/v1/process/frame', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: base64Image,
    include_annotated_frame: true
  })
});

const result = await response.json();
console.log('Detections:', result.detections.num_objects);
```

## Configuration

The API uses the configuration file at `config/default.yaml`. You can modify pipeline parameters by editing this file or providing configuration overrides in API requests.

## Error Handling

The API returns standard HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Invalid input (bad base64, invalid parameters)
- `404 Not Found`: Resource not found (task ID doesn't exist)
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Processing failure

Error responses include detailed information:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Performance Considerations

- **Frame Processing**: Typically 20-50ms per frame (1280x720)
- **Video Processing**: Runs asynchronously to avoid blocking
- **Memory Usage**: Monitored automatically, warnings at 80%
- **Concurrent Requests**: Supports multiple simultaneous frame processing requests
- **Task Storage**: In-memory storage (consider Redis for production)

## Testing

Run the test suite:
```bash
# Run all API tests
pytest tests/test_api_main.py -v

# Run with coverage
pytest tests/test_api_main.py --cov=src.api.main
```

## Interactive API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to test endpoints directly from your browser.

## Security Considerations

For production deployment:

1. **CORS**: Configure `allow_origins` to specific domains instead of `["*"]`
2. **Authentication**: Add API key or OAuth2 authentication
3. **Rate Limiting**: Implement rate limiting for resource-intensive endpoints
4. **File Upload Limits**: Configure maximum file size limits
5. **HTTPS**: Use HTTPS in production with proper SSL certificates
6. **Input Validation**: Already implemented with Pydantic models

## Monitoring and Logging

The API logs all requests and processing events:

```python
# Configure logging level
import logging
logging.basicConfig(level=logging.INFO)
```

Logs include:
- Request timestamps
- Processing times
- Error details with stack traces
- Component-level performance metrics

## Troubleshooting

### API won't start
- Check if port 8000 is already in use
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check configuration file exists: `config/default.yaml`

### Video processing fails
- Verify video file format is supported (MP4, AVI, MOV)
- Check available memory (processing requires significant RAM)
- Review logs for specific error messages

### Slow processing
- Reduce target resolution in configuration
- Disable unnecessary components
- Use smaller video files for testing
- Consider using GPU-accelerated OpenCV build

## License

Part of the OpenCV Video Processing Pipeline project.
