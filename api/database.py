import os
import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import aiomysql
from contextlib import asynccontextmanager
import json
import base64

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Async database manager for storing and retrieving face data.
    """
    
    def __init__(self):
        self.host = os.getenv("MYSQL_HOST", "db")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.user = os.getenv("MYSQL_USER", "mlflow")
        self.password = os.getenv("MYSQL_PASSWORD", "mlflow")
        self.database = os.getenv("MYSQL_DATABASE", "mlflow")
        
        self.pool = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection pool and create tables"""
        if self._initialized:
            return
            
        try:
            # Create connection pool
            self.pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=1,
                maxsize=10,
                autocommit=True
            )
            
            # Create tables
            await self._create_tables()
            self._initialized = True
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            self._initialized = False
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                yield cursor
    
    async def _create_tables(self):
        """Create necessary tables for face storage"""
        async with self.get_connection() as cursor:
            
            # Table for storing known faces (registered by users)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS known_faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_id VARCHAR(255) UNIQUE NOT NULL,
                    person_name VARCHAR(255) NOT NULL,
                    embedding LONGBLOB NOT NULL,
                    image_data LONGBLOB,
                    age FLOAT,
                    gender VARCHAR(10),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_person_id (person_id),
                    INDEX idx_person_name (person_name)
                )
            """)
            
            # Table for storing unknown faces (detected but not identified)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS unknown_faces (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_id VARCHAR(255) UNIQUE NOT NULL,
                    embedding LONGBLOB NOT NULL,
                    image_data LONGBLOB,
                    age FLOAT,
                    gender VARCHAR(10),
                    detection_count INT DEFAULT 1,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_person_id (person_id),
                    INDEX idx_last_seen (last_seen)
                )
            """)
            
            # Table for storing face detection history
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS face_detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    person_id VARCHAR(255),
                    face_type ENUM('known', 'unknown') NOT NULL,
                    confidence FLOAT,
                    bbox_x INT,
                    bbox_y INT,
                    bbox_width INT,
                    bbox_height INT,
                    age FLOAT,
                    gender VARCHAR(10),
                    gender_confidence FLOAT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_person_id (person_id),
                    INDEX idx_detected_at (detected_at),
                    INDEX idx_face_type (face_type)
                )
            """)
            
            logger.info("Database tables created successfully")
    
    async def register_known_face(self, 
                                 person_name: str, 
                                 embedding: np.ndarray,
                                 image_data: bytes = None,
                                 age: float = None,
                                 gender: str = None) -> str:
        """
        Register a known face in the database.
        
        Args:
            person_name: Name of the person
            embedding: Face embedding vector
            image_data: Original image data
            age: Estimated age
            gender: Estimated gender
            
        Returns:
            Generated person_id
        """
        try:
            # Generate unique person ID
            person_id = f"known_{person_name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
            
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    INSERT INTO known_faces 
                    (person_id, person_name, embedding, image_data, age, gender)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    person_name = VALUES(person_name),
                    embedding = VALUES(embedding),
                    image_data = VALUES(image_data),
                    age = VALUES(age),
                    gender = VALUES(gender),
                    updated_at = CURRENT_TIMESTAMP
                """, (person_id, person_name, embedding_bytes, image_data, age, gender))
                
                logger.info(f"Registered known face: {person_name} -> {person_id}")
                return person_id
                
        except Exception as e:
            logger.error(f"Error registering known face: {e}")
            raise
    
    async def store_unknown_face(self,
                                embedding: np.ndarray,
                                image_data: bytes = None,
                                age: float = None,
                                gender: str = None) -> str:
        """
        Store an unknown face in the database.
        
        Args:
            embedding: Face embedding vector
            image_data: Original image data
            age: Estimated age  
            gender: Estimated gender
            
        Returns:
            Generated person_id
        """
        try:
            # Generate unique person ID for unknown face
            person_id = f"unknown_{int(datetime.now().timestamp())}_{np.random.randint(1000, 9999)}"
            
            # Convert embedding to bytes
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    INSERT INTO unknown_faces 
                    (person_id, embedding, image_data, age, gender)
                    VALUES (%s, %s, %s, %s, %s)
                """, (person_id, embedding_bytes, image_data, age, gender))
                
                logger.info(f"Stored unknown face: {person_id}")
                return person_id
                
        except Exception as e:
            logger.error(f"Error storing unknown face: {e}")
            raise
    
    async def get_known_faces(self) -> List[Dict[str, Any]]:
        """
        Get all known faces from database.
        
        Returns:
            List of known face records
        """
        try:
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    SELECT person_id, person_name, embedding, age, gender, created_at
                    FROM known_faces
                    ORDER BY created_at DESC
                """)
                
                results = await cursor.fetchall()
                return results
                
        except Exception as e:
            logger.error(f"Error getting known faces: {e}")
            return []
    
    async def get_unknown_faces(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get unknown faces from database.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of unknown face records
        """
        try:
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    SELECT person_id, embedding, age, gender, detection_count, 
                           first_seen, last_seen
                    FROM unknown_faces
                    ORDER BY last_seen DESC
                    LIMIT %s
                """, (limit,))
                
                results = await cursor.fetchall()
                return results
                
        except Exception as e:
            logger.error(f"Error getting unknown faces: {e}")
            return []
    
    async def update_unknown_face_count(self, person_id: str):
        """
        Update detection count for an unknown face.
        
        Args:
            person_id: ID of the unknown face
        """
        try:
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    UPDATE unknown_faces 
                    SET detection_count = detection_count + 1,
                        last_seen = CURRENT_TIMESTAMP
                    WHERE person_id = %s
                """, (person_id,))
                
        except Exception as e:
            logger.error(f"Error updating unknown face count: {e}")
    
    async def log_face_detection(self,
                                person_id: str,
                                face_type: str,
                                confidence: float,
                                bbox: List[int],
                                age: float = None,
                                gender: str = None,
                                gender_confidence: float = None):
        """
        Log a face detection event.
        
        Args:
            person_id: ID of the detected person
            face_type: "known" or "unknown"
            confidence: Detection confidence
            bbox: Bounding box [x, y, width, height]
            age: Estimated age
            gender: Estimated gender
            gender_confidence: Gender prediction confidence
        """
        try:
            x, y, w, h = bbox
            
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    INSERT INTO face_detections 
                    (person_id, face_type, confidence, bbox_x, bbox_y, 
                     bbox_width, bbox_height, age, gender, gender_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (person_id, face_type, confidence, x, y, w, h, 
                     age, gender, gender_confidence))
                
        except Exception as e:
            logger.error(f"Error logging face detection: {e}")
    
    async def get_face_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored faces.
        
        Returns:
            Dictionary with face statistics
        """
        try:
            async with self.get_connection() as cursor:
                # Count known faces
                await cursor.execute("SELECT COUNT(*) as count FROM known_faces")
                known_count = (await cursor.fetchone())['count']
                
                # Count unknown faces
                await cursor.execute("SELECT COUNT(*) as count FROM unknown_faces")
                unknown_count = (await cursor.fetchone())['count']
                
                # Count total detections
                await cursor.execute("SELECT COUNT(*) as count FROM face_detections")
                total_detections = (await cursor.fetchone())['count']
                
                # Get recent detections (last 24 hours)
                await cursor.execute("""
                    SELECT COUNT(*) as count FROM face_detections 
                    WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """)
                recent_detections = (await cursor.fetchone())['count']
                
                # Get most detected unknown faces
                await cursor.execute("""
                    SELECT person_id, detection_count, last_seen
                    FROM unknown_faces
                    ORDER BY detection_count DESC
                    LIMIT 5
                """)
                top_unknown = await cursor.fetchall()
                
                return {
                    "known_faces": known_count,
                    "unknown_faces": unknown_count,
                    "total_detections": total_detections,
                    "recent_detections_24h": recent_detections,
                    "top_unknown_faces": top_unknown,
                    "last_updated": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting face statistics: {e}")
            return {
                "error": str(e),
                "known_faces": 0,
                "unknown_faces": 0,
                "total_detections": 0
            }
    
    async def search_faces_by_name(self, name_query: str) -> List[Dict[str, Any]]:
        """
        Search known faces by name.
        
        Args:
            name_query: Name search query
            
        Returns:
            List of matching face records
        """
        try:
            async with self.get_connection() as cursor:
                await cursor.execute("""
                    SELECT person_id, person_name, age, gender, created_at
                    FROM known_faces
                    WHERE person_name LIKE %s
                    ORDER BY person_name
                """, (f"%{name_query}%",))
                
                results = await cursor.fetchall()
                return results
                
        except Exception as e:
            logger.error(f"Error searching faces by name: {e}")
            return []

# Global database manager instance
_db_manager = None

async def get_db_manager() -> DatabaseManager:
    """Dependency injection for database manager"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def close_db_manager():
    """Close global database manager"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None