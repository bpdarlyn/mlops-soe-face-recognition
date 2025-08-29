# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that uses TensorFlow for training models and MLflow for experiment tracking. The project is containerized with Docker and uses MySQL as the backend database for MLflow.

## Architecture

The project consists of four main Docker services:
- **db**: MySQL database for MLflow experiment tracking and face data storage
- **mlflow**: MLflow server for experiment management and model registry
- **trainer**: Python environment for model training and testing
- **face-api**: FastAPI service for face detection, recognition, age/gender analysis

## Common Commands

### Environment Setup
```bash
# Build all Docker images
docker compose build

# Start database and MLflow server in background
docker compose up -d db mlflow

# Start the Face Analytics API service
docker compose up -d face-api

# Check MLflow server status
docker compose logs -f mlflow
# Wait for "Listening at: http://0.0.0.0:5000", then access http://localhost:5000

# Check Face API status
docker compose logs -f face-api
# API will be available at http://localhost:8000
```

### Face Analytics API
```bash
# Start the Face Analytics API
docker compose up -d face-api

# Test the API health
curl http://localhost:8000/health

# Test face analysis (upload an image)
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/analyze

# Get face statistics
curl http://localhost:8000/stats

# Register a known face
curl -X POST -F "file=@path/to/face.jpg" -F "person_name=John Doe" http://localhost:8000/register

# Run API tests with Python script
python test_api.py path/to/test_image.jpg
```

### Testing
```bash
# Run all tests with pytest
docker compose run --rm trainer pytest -q

# Run tests with coverage
docker compose run --rm trainer pytest --cov

# Test the Face Analytics API
python test_api.py
```

### Training
```bash
# Run the smoke model training pipeline
docker compose run --rm trainer python training/train_smoke_tf.py

# Train face recognition model
docker compose run --rm trainer python training/train_face_recognition_tf.py

# Train age/gender model
docker compose run --rm trainer python training/train_age_gender_tf.py
```

### Model Management
```bash
# Register a trained model
docker compose run --rm trainer python scripts/register_model.py

# Deploy the best model automatically
docker compose run --rm trainer python scripts/deploy_model.py

# Check model status
docker compose run --rm trainer python scripts/check_models.py
```

### Interactive Development
```bash
# Start an interactive shell in the trainer container
docker compose run --rm trainer bash

# Start an interactive shell in the API container
docker compose exec face-api bash
```

## Code Structure

### Training Components
- `training/smoke_model.py`: Core model definition and training utilities
  - `build_model()`: Creates a simple dense neural network for binary classification
  - `make_data()`: Generates synthetic training data
  - `train_smoke()`: Full training pipeline with validation
- `training/train_smoke_tf.py`: MLflow experiment script that logs metrics, artifacts, and exports ONNX models
- `training/train_face_recognition_tf.py`: Face recognition model training with QMUL-SurvFace dataset
- `training/train_age_gender_tf.py`: Age and gender estimation model training with UTKFace dataset
- `training/combined_face_analytics.py`: Multi-task face analytics pipeline (recognition + age/gender)
- `training/datasets/survface.py`: QMUL-SurvFace dataset processing utilities
- `training/datasets/utkface.py`: UTKFace dataset processing utilities

### API Components
- `api/main.py`: FastAPI application with face analytics endpoints
  - `/analyze`: Analyze uploaded image for faces, age, gender, and identity
  - `/register`: Register known faces with names
  - `/stats`: Get face detection statistics
  - `/health`: Health check endpoint
- `api/face_detector.py`: Face detection using OpenCV and MediaPipe
- `api/face_analytics.py`: Age/gender prediction and face recognition service
- `api/database.py`: MySQL database management for storing face data

### Scripts and Management
- `scripts/register_model.py`: Register trained models in MLflow
- `scripts/deploy_model.py`: Deploy best model to production
- `scripts/check_models.py`: Check model status and performance
- `scripts/model_utils.py`: Model management utilities

### Testing
- `tests/test_smoke.py`: Unit tests for smoke model
- `tests/test_face_recognition.py`: Unit tests for face recognition training
- `tests/test_survface_dataset.py`: Unit tests for QMUL-SurvFace dataset processing
- `tests/test_combined_face_analytics.py`: Unit tests for combined face analytics
- `test_api.py`: Integration tests for Face Analytics API

## Key Dependencies

- TensorFlow 2.15.0
- MLflow 2.14.3 for experiment tracking
- tf2onnx for ONNX model export
- pytest for testing

## Environment Variables

The project uses environment variables for database configuration (defined in docker-compose.yml):
- `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `MLFLOW_TRACKING_URI`: Set to `http://mlflow:5000` in the trainer container