import os
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .database import DatabaseManager, get_db_manager
from .face_detector import FaceDetector
from .face_analytics import FaceAnalyticsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Face Analytics API",
    description="API for face detection, recognition, age and gender estimation",
    version="1.0.0"
)

class FaceAnalyticsResponse(BaseModel):
    success: bool
    faces: List[Dict[str, Any]]
    message: Optional[str] = None

class FaceData(BaseModel):
    bbox: Dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    confidence: float
    age: Optional[float] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None
    identity: Optional[str] = None  # "known" or "unknown"
    person_id: Optional[str] = None

class AgeGenderResponse(BaseModel):
    success: bool
    faces: List[Dict[str, Any]]
    message: Optional[str] = None
    
class AgeGenderFaceData(BaseModel):
    bbox: Dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    confidence: float
    age: float
    gender: str  # "male" or "female"
    gender_confidence: float

# Global services
face_detector = None
face_analytics_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global face_detector, face_analytics_service
    
    try:
        # Initialize face detector
        face_detector = FaceDetector()
        logger.info("Face detector initialized")
        
        # Initialize face analytics service
        face_analytics_service = FaceAnalyticsService()
        logger.info("Face analytics service initialized")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Face Analytics API is running"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "services": {
            "face_detector": face_detector is not None,
            "face_analytics": face_analytics_service is not None,
        }
    }
    
    if not all(health_status["services"].values()):
        health_status["status"] = "unhealthy"
    
    return health_status

@app.post("/analyze", response_model=FaceAnalyticsResponse)
async def analyze_face(
    file: UploadFile = File(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Analyze an uploaded image for faces, age, gender, and identity.
    
    Args:
        file: Image file to analyze (JPG, PNG, etc.)
        
    Returns:
        JSON with face detection results including:
        - bbox: bounding box coordinates
        - age: estimated age
        - gender: estimated gender (male/female)
        - identity: known/unknown status
        - person_id: if known, the person's ID
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_detector.detect_faces(image_cv)
        
        if not faces:
            return FaceAnalyticsResponse(
                success=True,
                faces=[],
                message="No faces detected in the image"
            )
        
        # Analyze each detected face
        analyzed_faces = []
        for face in faces:
            try:
                # Extract face region
                x, y, w, h = face["bbox"]
                face_roi = image_cv[y:y+h, x:x+w]
                
                # Get age and gender predictions
                age, gender, gender_conf = face_analytics_service.predict_age_gender(face_roi)
                
                # Get face embedding for identity recognition
                embedding = face_analytics_service.extract_embedding(face_roi)
                
                # Check if face is known
                identity, person_id = await face_analytics_service.identify_face(
                    embedding, db_manager
                )
                
                # If unknown, store in database for future recognition
                if identity == "unknown":
                    person_id = await db_manager.store_unknown_face(
                        embedding, image_bytes, age, gender
                    )
                
                analyzed_face = FaceData(
                    bbox={
                        "x": int(x),
                        "y": int(y), 
                        "width": int(w),
                        "height": int(h)
                    },
                    confidence=float(face["confidence"]),
                    age=float(age) if age is not None else None,
                    gender=gender,
                    gender_confidence=float(gender_conf) if gender_conf is not None else None,
                    identity=identity,
                    person_id=person_id
                )
                
                analyzed_faces.append(analyzed_face.dict())
                
            except Exception as e:
                logger.error(f"Error analyzing face: {e}")
                # Continue with other faces even if one fails
                continue
        
        return FaceAnalyticsResponse(
            success=True,
            faces=analyzed_faces,
            message=f"Successfully analyzed {len(analyzed_faces)} face(s)"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.post("/infer_age_genre", response_model=AgeGenderResponse)
async def infer_age_genre(file: UploadFile = File(...)):
    """
    Infer age and gender from an uploaded image using the dedicated age/gender model.
    
    This endpoint uses only the age/gender model trained with UTKFace dataset,
    without face recognition or database storage.
    
    Args:
        file: Image file to analyze (JPG, PNG, etc.)
        
    Returns:
        JSON with age and gender predictions for detected faces:
        - bbox: bounding box coordinates
        - confidence: face detection confidence
        - age: estimated age (years)
        - gender: estimated gender (male/female)
        - gender_confidence: gender prediction confidence (0-1)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = face_detector.detect_faces(image_cv)
        
        if not faces:
            return AgeGenderResponse(
                success=True,
                faces=[],
                message="No faces detected in the image"
            )
        
        # Analyze each detected face for age and gender only
        analyzed_faces = []
        for face in faces:
            try:
                # Extract face region
                x, y, w, h = face["bbox"]
                face_roi = image_cv[y:y+h, x:x+w]
                
                # Get age and gender predictions using dedicated model
                age, gender, gender_conf = face_analytics_service.predict_age_gender(face_roi)
                
                if age is not None and gender is not None and gender_conf is not None:
                    analyzed_face = AgeGenderFaceData(
                        bbox={
                            "x": int(x),
                            "y": int(y), 
                            "width": int(w),
                            "height": int(h)
                        },
                        confidence=float(face["confidence"]),
                        age=float(age),
                        gender=gender,
                        gender_confidence=float(gender_conf)
                    )
                    
                    analyzed_faces.append(analyzed_face.dict())
                else:
                    logger.warning(f"Could not predict age/gender for face at {face['bbox']}")
                
            except Exception as e:
                logger.error(f"Error analyzing face for age/gender: {e}")
                # Continue with other faces even if one fails
                continue
        
        return AgeGenderResponse(
            success=True,
            faces=analyzed_faces,
            message=f"Successfully analyzed {len(analyzed_faces)} face(s) for age and gender"
        )
        
    except Exception as e:
        logger.error(f"Error processing image for age/gender: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/stats")
async def get_stats(db_manager: DatabaseManager = Depends(get_db_manager)):
    """Get statistics about stored faces"""
    try:
        stats = await db_manager.get_face_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting statistics: {str(e)}"
        )

@app.post("/register")
async def register_known_face(
    person_name: str,
    file: UploadFile = File(...),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    Register a known face with a name for future recognition.
    
    Args:
        person_name: Name to associate with this face
        file: Image file containing the face
        
    Returns:
        Success message with person ID
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face
        faces = face_detector.detect_faces(image_cv)
        
        if not faces:
            raise HTTPException(
                status_code=400,
                detail="No faces detected in the image"
            )
        
        if len(faces) > 1:
            raise HTTPException(
                status_code=400,
                detail="Multiple faces detected. Please provide an image with only one face"
            )
        
        # Extract face and get embedding
        face = faces[0]
        x, y, w, h = face["bbox"]
        face_roi = image_cv[y:y+h, x:x+w]
        
        # Get predictions
        age, gender, _ = face_analytics_service.predict_age_gender(face_roi)
        embedding = face_analytics_service.extract_embedding(face_roi)
        
        # Register known face
        person_id = await db_manager.register_known_face(
            person_name, embedding, image_bytes, age, gender
        )
        
        return {
            "success": True,
            "message": f"Successfully registered {person_name}",
            "person_id": person_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error registering face: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)