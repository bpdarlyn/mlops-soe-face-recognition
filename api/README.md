# Face Analytics API

FastAPI service for real-time face detection, recognition, age and gender estimation.

## Features

- **Face Detection**: Detect faces in uploaded images with bounding box coordinates
- **Age Estimation**: Predict age for detected faces
- **Gender Classification**: Classify gender (male/female) with confidence scores
- **Face Recognition**: Identify if faces are known or unknown
- **Database Storage**: Store unknown faces for future recognition
- **Face Registration**: Register known faces with names

## API Endpoints

### Core Endpoints

#### `POST /analyze`
Analyze an uploaded image for faces.

**Request:**
- `file`: Image file (JPG, PNG, etc.)

**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "bbox": {"x": 100, "y": 50, "width": 150, "height": 200},
      "confidence": 0.95,
      "age": 25.5,
      "gender": "female",
      "gender_confidence": 0.85,
      "identity": "unknown",
      "person_id": "unknown_1703123456_7890"
    }
  ],
  "message": "Successfully analyzed 1 face(s)"
}
```

#### `POST /register`
Register a known face with a name.

**Request:**
- `person_name`: Name to associate with the face
- `file`: Image file containing the face

**Response:**
```json
{
  "success": true,
  "message": "Successfully registered John Doe",
  "person_id": "known_john_doe_1703123456"
}
```

#### `GET /stats`
Get statistics about stored faces.

**Response:**
```json
{
  "known_faces": 10,
  "unknown_faces": 25,
  "total_detections": 150,
  "recent_detections_24h": 12,
  "top_unknown_faces": [
    {
      "person_id": "unknown_1703123456_7890",
      "detection_count": 5,
      "last_seen": "2023-12-21T10:30:00"
    }
  ],
  "last_updated": "2023-12-21T12:00:00"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "face_detector": true,
    "face_analytics": true
  }
}
```

## Usage Examples

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Analyze a face
curl -X POST -F "file=@photo.jpg" http://localhost:8000/analyze

# Register a known face
curl -X POST -F "file=@john.jpg" -F "person_name=John Doe" http://localhost:8000/register

# Get statistics
curl http://localhost:8000/stats
```

### Using Python requests

```python
import requests

# Analyze a face
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    result = response.json()
    print(f"Found {len(result['faces'])} faces")

# Register a known face
with open('john.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/register',
        files={'file': f},
        data={'person_name': 'John Doe'}
    )
    print(response.json())
```

## Architecture

The API consists of several key components:

- **Face Detector** (`face_detector.py`): Uses OpenCV and MediaPipe for face detection
- **Face Analytics Service** (`face_analytics.py`): Handles age/gender prediction and face recognition using trained models
- **Database Manager** (`database.py`): Manages MySQL database for storing face data
- **Main API** (`main.py`): FastAPI application with all endpoints

## Database Schema

The API uses three main tables:

### `known_faces`
- `person_id`: Unique identifier
- `person_name`: Human-readable name
- `embedding`: Face embedding vector (BLOB)
- `image_data`: Original image data (BLOB)
- `age`: Estimated age
- `gender`: Estimated gender

### `unknown_faces`  
- `person_id`: Unique identifier
- `embedding`: Face embedding vector (BLOB)
- `detection_count`: Number of times detected
- `first_seen`: First detection timestamp
- `last_seen`: Last detection timestamp

### `face_detections`
- `person_id`: Reference to person
- `face_type`: "known" or "unknown"
- `confidence`: Detection confidence
- `bbox_*`: Bounding box coordinates
- `detected_at`: Detection timestamp

## Model Integration

The API integrates with trained models from MLflow:

1. **Age/Gender Model**: Predicts age (regression) and gender (binary classification)
2. **Face Recognition Model**: Extracts face embeddings for similarity comparison
3. **Combined Model**: Multi-task model handling all tasks simultaneously

Models are automatically loaded from MLflow registry using the production alias.

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format or missing faces
- **500 Internal Server Error**: Model loading or processing errors
- **503 Service Unavailable**: Database connection issues

## Performance Considerations

- **Model Loading**: Models are loaded once at startup
- **Database Pooling**: Connection pooling for efficient database access
- **Image Processing**: Optimized OpenCV operations
- **Memory Management**: Proper cleanup of image data

## Development

### Running Locally

```bash
# Start the API service
docker compose up -d face-api

# View logs
docker compose logs -f face-api

# Test the API
python test_api.py path/to/test/image.jpg
```

### Development Mode

```bash
# Run in development mode with auto-reload
docker compose exec face-api uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Configuration

Environment variables (set in docker-compose.yml):

- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MYSQL_HOST`, `MYSQL_PORT`: Database connection
- `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`: Database credentials

## Security Notes

- File uploads are validated for image types
- Face embeddings are stored securely in the database
- No sensitive data is logged
- Database connections use proper authentication