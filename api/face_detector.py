import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detection using OpenCV's DNN module with pre-trained models.
    Supports multiple detection methods for robust face detection.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.4):
        """
        Initialize face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection
            nms_threshold: Non-maximum suppression threshold
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Initialize detection models
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize face detection models"""
        try:
            # Method 1: OpenCV DNN Face Detector (primary)
            self.face_net = None
            try:
                # Try to load DNN face detector if available
                # This would require downloading the model files
                # For now, we'll use Haar cascades as fallback
                pass
            except Exception as e:
                logger.warning(f"DNN face detector not available: {e}")
            
            # Method 2: Haar Cascade Classifier (fallback)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Method 3: MediaPipe Face Detection (if available)
            self.mp_detector = None
            try:
                import mediapipe as mp
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_detector = self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=self.confidence_threshold
                )
                logger.info("MediaPipe face detector initialized")
            except ImportError:
                logger.warning("MediaPipe not available, using OpenCV only")
            
            logger.info("Face detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing face detectors: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using multiple methods.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of face detections with bounding boxes and confidence scores
        """
        if image is None or image.size == 0:
            return []
        
        faces = []
        
        # Try MediaPipe first (most accurate)
        if self.mp_detector is not None:
            mp_faces = self._detect_with_mediapipe(image)
            if mp_faces:
                faces.extend(mp_faces)
        
        # If no faces found with MediaPipe, try Haar cascades
        if not faces:
            haar_faces = self._detect_with_haar(image)
            faces.extend(haar_faces)
        
        # Remove duplicate detections and apply NMS
        faces = self._apply_nms(faces)
        
        return faces
    
    def _detect_with_mediapipe(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using MediaPipe"""
        if self.mp_detector is None:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Process image
            results = self.mp_detector.process(rgb_image)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Extract bounding box
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Get confidence
                    confidence = detection.score[0] if detection.score else 0.0
                    
                    if confidence >= self.confidence_threshold:
                        faces.append({
                            "bbox": [x, y, width, height],
                            "confidence": confidence,
                            "method": "mediapipe"
                        })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in MediaPipe detection: {e}")
            return []
    
    def _detect_with_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar cascades"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces = []
            for (x, y, w, h) in faces_rect:
                faces.append({
                    "bbox": [x, y, w, h],
                    "confidence": 0.8,  # Haar cascades don't provide confidence
                    "method": "haar"
                })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in Haar detection: {e}")
            return []
    
    def _apply_nms(self, faces: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(faces) <= 1:
            return faces
        
        try:
            # Extract bounding boxes and scores
            boxes = []
            scores = []
            
            for face in faces:
                x, y, w, h = face["bbox"]
                boxes.append([x, y, x + w, y + h])
                scores.append(face["confidence"])
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores, dtype=np.float32)
            
            # Apply NMS
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                self.confidence_threshold,
                self.nms_threshold
            )
            
            # Filter faces based on NMS results
            if len(indices) > 0:
                indices = indices.flatten()
                return [faces[i] for i in indices]
            else:
                return faces[:1]  # Return at least one detection if available
                
        except Exception as e:
            logger.error(f"Error applying NMS: {e}")
            return faces  # Return original faces if NMS fails
    
    def draw_faces(self, image: np.ndarray, faces: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes around detected faces.
        
        Args:
            image: Input image
            faces: List of face detections
            
        Returns:
            Image with drawn bounding boxes
        """
        result_image = image.copy()
        
        for face in faces:
            x, y, w, h = face["bbox"]
            confidence = face["confidence"]
            
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face: {confidence:.2f}"
            cv2.putText(result_image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], 
                       target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Extract and preprocess face region for model inference.
        
        Args:
            image: Full image
            bbox: Bounding box [x, y, width, height]
            target_size: Target size for resizing
            
        Returns:
            Preprocessed face image
        """
        try:
            x, y, w, h = bbox
            
            # Extract face region
            face = image[y:y+h, x:x+w]
            
            # Resize to target size
            face_resized = cv2.resize(face, target_size)
            
            # Convert to RGB if needed
            if len(face_resized.shape) == 3 and face_resized.shape[2] == 3:
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            return face_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            # Return a blank image if preprocessing fails
            return np.zeros((*target_size, 3), dtype=np.float32)