import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import logging
from typing import Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class FaceAnalyticsService:
    """
    Service for face analytics including age/gender prediction and face recognition.
    """
    
    def __init__(self, 
                 model_name: str = "face-analytics-model",
                 model_alias: str = "prod",
                 similarity_threshold: float = 0.8):
        """
        Initialize face analytics service.
        
        Args:
            model_name: Name of the registered model in MLflow
            model_alias: Model alias to load (e.g., "prod", "latest")
            similarity_threshold: Threshold for face similarity matching
        """
        self.model_name = model_name
        self.model_alias = model_alias
        self.similarity_threshold = similarity_threshold
        
        # Initialize model attributes
        self.full_model = None
        self.embedding_model = None
        self.age_gender_model = None
        
        # Initialize MLflow
        self.mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models from MLflow"""
        try:
            logger.info(f"Loading models from MLflow: {self.mlflow_uri}")
            
            # Try to load the full model first
            model_uri = f"models:/{self.model_name}/{self.model_alias}"
            
            try:
                self.full_model = mlflow.tensorflow.load_model(model_uri)
                logger.info(f"Loaded full model: {model_uri}")
                
                # Extract individual models from the full model if possible
                self.embedding_model = None
                self.age_gender_model = None
                
                # Try to create embedding model (extract up to embedding layer)
                try:
                    # Find the embedding layer
                    embedding_layer_name = "face_embeddings_normalized"
                    if any(layer.name == embedding_layer_name for layer in self.full_model.layers):
                        embedding_output = self.full_model.get_layer(embedding_layer_name).output
                        self.embedding_model = tf.keras.Model(
                            inputs=self.full_model.input,
                            outputs=embedding_output,
                            name="embedding_extractor"
                        )
                        logger.info("Created embedding model from full model")
                except Exception as e:
                    logger.warning(f"Could not create embedding model: {e}")
                
                # Try to create age/gender model
                try:
                    age_output = self.full_model.get_layer("age").output
                    gender_output = self.full_model.get_layer("gender").output
                    self.age_gender_model = tf.keras.Model(
                        inputs=self.full_model.input,
                        outputs=[age_output, gender_output],
                        name="age_gender_extractor"
                    )
                    logger.info("Created age/gender model from full model")
                except Exception as e:
                    logger.warning(f"Could not create age/gender model: {e}")
                    
            except Exception as e:
                logger.error(f"Could not load full model: {e}")
                # Try to load individual models
                self._load_fallback_models()
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Load fallback models or create dummy ones
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models or create dummy ones"""
        logger.warning("Loading fallback models")
        
        try:
            # Try to load age/gender model from previous training
            age_gender_path = "/app/artifacts/age_gender_savedmodel"
            if os.path.exists(age_gender_path):
                self.age_gender_model = tf.keras.models.load_model(age_gender_path)
                logger.info(f"Loaded age/gender model from {age_gender_path}")
            else:
                self._create_dummy_age_gender_model()
        except Exception as e:
            logger.error(f"Error loading fallback age/gender model: {e}")
            self._create_dummy_age_gender_model()
        
        # For embedding, create a dummy model if needed
        if self.embedding_model is None:
            self._create_dummy_embedding_model()
    
    def _create_dummy_age_gender_model(self):
        """Create a dummy age/gender model for testing"""
        logger.warning("Creating dummy age/gender model")
        
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        age_output = tf.keras.layers.Dense(1, name="age")(x)
        gender_output = tf.keras.layers.Dense(1, activation="sigmoid", name="gender")(x)
        
        self.age_gender_model = tf.keras.Model(
            inputs=inputs,
            outputs=[age_output, gender_output],
            name="dummy_age_gender"
        )
    
    def _create_dummy_embedding_model(self):
        """Create a dummy embedding model for testing"""
        logger.warning("Creating dummy embedding model")
        
        inputs = tf.keras.Input(shape=(160, 160, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        embedding_output = tf.keras.layers.Dense(512, name="embeddings")(x)
        embedding_normalized = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name="embeddings_normalized"
        )(embedding_output)
        
        self.embedding_model = tf.keras.Model(
            inputs=inputs,
            outputs=embedding_normalized,
            name="dummy_embedding"
        )
    
    def predict_age_gender(self, face_image: np.ndarray) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """
        Predict age and gender for a face image.
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Tuple of (age, gender, gender_confidence)
        """
        try:
            # Preprocess image
            face_processed = self._preprocess_face(face_image)
            face_batch = np.expand_dims(face_processed, axis=0)
            
            if self.age_gender_model is not None:
                # Predict using age/gender model
                predictions = self.age_gender_model.predict(face_batch, verbose=0)
                age_pred, gender_pred = predictions
                
                age = float(age_pred[0][0])
                gender_conf = float(gender_pred[0][0])
                gender = "female" if gender_conf > 0.5 else "male"
                
                # Ensure reasonable age range
                age = max(0, min(100, age))
                
                return age, gender, gender_conf
            
            elif self.full_model is not None:
                # Use full model
                predictions = self.full_model.predict(face_batch, verbose=0)
                
                if isinstance(predictions, dict):
                    age = float(predictions["age"][0][0])
                    gender_conf = float(predictions["gender"][0][0])
                elif isinstance(predictions, list) and len(predictions) >= 2:
                    age = float(predictions[1][0][0])  # Assuming age is second output
                    gender_conf = float(predictions[2][0][0])  # Assuming gender is third output
                else:
                    return None, None, None
                
                gender = "female" if gender_conf > 0.5 else "male"
                age = max(0, min(100, age))
                
                return age, gender, gender_conf
            
            else:
                logger.warning("No age/gender model available")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Error predicting age/gender: {e}")
            return None, None, None
    
    def extract_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding for recognition.
        
        Args:
            face_image: Face image in BGR format
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            # Preprocess image
            face_processed = self._preprocess_face(face_image)
            face_batch = np.expand_dims(face_processed, axis=0)
            
            if self.embedding_model is not None:
                # Extract embedding using embedding model
                embedding = self.embedding_model.predict(face_batch, verbose=0)
                return embedding[0]  # Return first (and only) embedding
            
            elif self.full_model is not None:
                # Try to extract from full model
                # This might need adjustment based on the actual model structure
                try:
                    # If the model has embedding output
                    predictions = self.full_model.predict(face_batch, verbose=0)
                    
                    # Handle different output formats
                    if isinstance(predictions, dict):
                        # Look for embedding output
                        if "embeddings" in predictions:
                            return predictions["embeddings"][0]
                        elif "face_embeddings_normalized" in predictions:
                            return predictions["face_embeddings_normalized"][0]
                    
                    # If no embedding found, create a simple feature vector
                    logger.warning("No embedding output found, creating simple feature")
                    return self._create_simple_feature(face_processed)
                    
                except Exception as e:
                    logger.error(f"Error extracting embedding from full model: {e}")
                    return self._create_simple_feature(face_processed)
            
            else:
                logger.warning("No embedding model available, creating simple feature")
                return self._create_simple_feature(face_processed)
                
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def _create_simple_feature(self, face_image: np.ndarray) -> np.ndarray:
        """Create a simple feature vector from face image"""
        try:
            # Convert to grayscale and resize
            if len(face_image.shape) == 3:
                gray = np.mean(face_image, axis=2)
            else:
                gray = face_image
            
            # Create histogram of gradients as simple features
            from skimage.feature import hog
            features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            
            # Normalize
            features = features / np.linalg.norm(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating simple feature: {e}")
            # Return random normalized vector as absolute fallback
            feature = np.random.randn(512)
            return feature / np.linalg.norm(feature)
    
    async def identify_face(self, embedding: np.ndarray, db_manager) -> Tuple[str, Optional[str]]:
        """
        Identify if a face is known or unknown.
        
        Args:
            embedding: Face embedding vector
            db_manager: Database manager instance
            
        Returns:
            Tuple of (identity, person_id) where identity is "known" or "unknown"
        """
        try:
            if embedding is None:
                return "unknown", None
            
            # Get known faces from database
            known_faces = await db_manager.get_known_faces()
            
            if not known_faces:
                return "unknown", None
            
            # Calculate similarities
            best_match_id = None
            best_similarity = 0.0
            
            for face_record in known_faces:
                stored_embedding = np.frombuffer(face_record["embedding"], dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    embedding.reshape(1, -1),
                    stored_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = face_record["person_id"]
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.similarity_threshold:
                return "known", best_match_id
            else:
                return "unknown", None
                
        except Exception as e:
            logger.error(f"Error identifying face: {e}")
            return "unknown", None
    
    def _preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Preprocess face image for model inference.
        
        Args:
            face_image: Face image in BGR format
            target_size: Target size for resizing
            
        Returns:
            Preprocessed face image
        """
        try:
            import cv2
            
            # Convert BGR to RGB
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                face_rgb = face_image
            
            # Resize
            face_resized = cv2.resize(face_rgb, target_size)
            
            # Normalize to [0, 1]
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            return face_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return np.zeros((*target_size, 3), dtype=np.float32)