# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that uses TensorFlow for training models and MLflow for experiment tracking. The project is containerized with Docker and uses MySQL as the backend database for MLflow.

## Architecture

The project consists of three main Docker services:
- **db**: MySQL database for MLflow experiment tracking
- **mlflow**: MLflow server for experiment management and model registry
- **trainer**: Python environment for model training and testing

## Common Commands

### Environment Setup
```bash
# Build all Docker images
docker compose build

# Start database and MLflow server in background
docker compose up -d db mlflow

# Check MLflow server status
docker compose logs -f mlflow
# Wait for "Listening at: http://0.0.0.0:5000", then access http://localhost:5000
```

### Testing
```bash
# Run all tests with pytest
docker compose run --rm trainer pytest -q

# Run tests with coverage
docker compose run --rm trainer pytest --cov
```

### Training
```bash
# Run the smoke model training pipeline
docker compose run --rm trainer python training/train_smoke_tf.py
```

### Interactive Development
```bash
# Start an interactive shell in the trainer container
docker compose run --rm trainer bash
```

## Code Structure

- `training/smoke_model.py`: Core model definition and training utilities
  - `build_model()`: Creates a simple dense neural network for binary classification
  - `make_data()`: Generates synthetic training data
  - `train_smoke()`: Full training pipeline with validation
- `training/train_smoke_tf.py`: MLflow experiment script that logs metrics, artifacts, and exports ONNX models
- `tests/test_smoke.py`: Unit tests for model building, training, and prediction functionality

## Key Dependencies

- TensorFlow 2.15.0
- MLflow 2.14.3 for experiment tracking
- tf2onnx for ONNX model export
- pytest for testing

## Environment Variables

The project uses environment variables for database configuration (defined in docker-compose.yml):
- `MYSQL_ROOT_PASSWORD`, `MYSQL_DATABASE`, `MYSQL_USER`, `MYSQL_PASSWORD`
- `MLFLOW_TRACKING_URI`: Set to `http://mlflow:5000` in the trainer container