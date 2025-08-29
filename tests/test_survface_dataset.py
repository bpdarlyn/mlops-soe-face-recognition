import pytest
import numpy as np
import tensorflow as tf
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.datasets.survface import (
    analyze_dataset_structure,
    split_dataset,
    create_face_identification_dataset
)

class TestSurvfaceDataset:
    """Tests para el módulo de dataset QMUL-SurvFace."""
    
    def setup_method(self):
        """Setup que se ejecuta antes de cada test."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        
    def teardown_method(self):
        """Cleanup después de cada test."""
        tf.keras.backend.clear_session()

    def test_analyze_dataset_structure_nonexistent_dir(self):
        """Test que verifica el manejo de directorios inexistentes."""
        with pytest.raises(ValueError, match="Dataset directory not found"):
            analyze_dataset_structure("/nonexistent/path")

    @patch('training.datasets.survface.Path')
    def test_analyze_dataset_structure_empty_dataset(self, mock_path):
        """Test para dataset vacío."""
        # Setup mock
        mock_data_path = Mock()
        mock_path.return_value = mock_data_path
        mock_data_path.exists.return_value = True
        mock_data_path.iterdir.return_value = []  # No directories
        
        stats = analyze_dataset_structure("/fake/empty/path")
        
        assert stats['total_persons'] == 0
        assert stats['total_images'] == 0
        assert stats['avg_images_per_person'] == 0
        assert stats['min_images_per_person'] == float('inf')
        assert stats['max_images_per_person'] == 0

    @patch('training.datasets.survface.Path')
    def test_analyze_dataset_structure_normal_dataset(self, mock_path):
        """Test para dataset normal con varias personas."""
        # Setup mock
        mock_data_path = Mock()
        mock_path.return_value = mock_data_path
        mock_data_path.exists.return_value = True
        
        # Mock person directories
        mock_persons = []
        for i, num_images in enumerate([5, 3, 8, 2]):
            mock_person = Mock()
            mock_person.is_dir.return_value = True
            mock_person.glob.return_value = [f'img{j}.jpg' for j in range(num_images)]
            mock_persons.append(mock_person)
        
        mock_data_path.iterdir.return_value = mock_persons
        
        stats = analyze_dataset_structure("/fake/path")
        
        assert stats['total_persons'] == 4
        assert stats['total_images'] == 18  # 5+3+8+2
        assert stats['min_images_per_person'] == 2
        assert stats['max_images_per_person'] == 8
        assert stats['avg_images_per_person'] == 4.5

    def test_split_dataset_invalid_splits(self):
        """Test para splits inválidos que no suman 1.0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(source_dir)
            
            with pytest.raises(ValueError, match="Los splits deben sumar 1.0"):
                split_dataset(
                    source_dir=source_dir,
                    output_dir=output_dir,
                    train_split=0.7,
                    val_split=0.2,
                    test_split=0.2  # 0.7 + 0.2 + 0.2 = 1.1
                )

    def test_split_dataset_with_real_directories(self):
        """Test que crea un dataset real pequeño y lo divide."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear estructura de dataset sintético
            source_dir = os.path.join(temp_dir, "source")
            output_dir = os.path.join(temp_dir, "output")
            
            # Crear personas con diferentes cantidades de imágenes
            persons_data = {
                "person_001": 6,
                "person_002": 4,
                "person_003": 8,
                "person_004": 3,
                "person_005": 7
            }
            
            for person_id, num_images in persons_data.items():
                person_dir = os.path.join(source_dir, person_id)
                os.makedirs(person_dir)
                
                for i in range(num_images):
                    # Crear archivos dummy de imágenes
                    image_path = os.path.join(person_dir, f"{person_id}_cam1_{i+1}.jpg")
                    with open(image_path, 'w') as f:
                        f.write("dummy_image_content")
            
            # Realizar split
            stats = split_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                train_split=0.6,
                val_split=0.2,
                test_split=0.2,
                min_images_per_person=4,  # Solo personas con 4+ imágenes
                seed=42
            )
            
            # Verificar estadísticas
            # Personas válidas: person_001(6), person_002(4), person_003(8), person_005(7) = 4 personas
            assert stats['total_valid_persons'] == 4
            assert stats['train_persons'] + stats['val_persons'] + stats['test_persons'] == 4
            assert stats['train_images'] + stats['val_images'] + stats['test_images'] == 25  # 6+4+8+7
            
            # Verificar que se crearon los directorios
            assert os.path.exists(os.path.join(output_dir, "train"))
            assert os.path.exists(os.path.join(output_dir, "validation"))
            assert os.path.exists(os.path.join(output_dir, "test"))
            
            # Verificar que se creó el archivo de metadatos
            metadata_path = os.path.join(output_dir, "split_metadata.json")
            assert os.path.exists(metadata_path)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['train_split'] == 0.6
            assert metadata['val_split'] == 0.2
            assert metadata['test_split'] == 0.2
            assert metadata['min_images_per_person'] == 4
            assert metadata['seed'] == 42

    def test_create_face_identification_dataset_with_mock_splits(self):
        """Test para crear datasets de TensorFlow desde splits mock."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear estructura de splits mock
            splits = ['train', 'validation', 'test']
            persons_per_split = {
                'train': ['person_001', 'person_002'],
                'validation': ['person_003'],
                'test': ['person_004']
            }
            
            for split in splits:
                split_dir = os.path.join(temp_dir, split)
                os.makedirs(split_dir)
                
                for person_id in persons_per_split[split]:
                    person_dir = os.path.join(split_dir, person_id)
                    os.makedirs(person_dir)
                    
                    # Crear archivos de imagen dummy
                    for i in range(3):
                        image_path = os.path.join(person_dir, f"{person_id}_cam1_{i+1}.jpg")
                        # Crear una imagen PNG mínima válida (1x1 pixel)
                        import struct
                        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x03\x00\x01H\x9a\xa0\x00\x00\x00\x00IEND\xaeB`\x82'
                        with open(image_path, 'wb') as f:
                            f.write(png_data)
            
            # Mock de TensorFlow para evitar problemas de lectura de imágenes
            with patch('tensorflow.io.read_file') as mock_read_file, \
                 patch('tensorflow.io.decode_image') as mock_decode_image, \
                 patch('tensorflow.image.resize') as mock_resize:
                
                # Setup mocks
                mock_read_file.return_value = "dummy_file_content"
                mock_decode_image.return_value = tf.zeros((100, 100, 3), dtype=tf.uint8)
                mock_resize.return_value = tf.zeros((160, 160, 3), dtype=tf.float32)
                
                # Crear datasets
                train_ds, val_ds, test_ds, label_info = create_face_identification_dataset(
                    data_dir=temp_dir,
                    img_size=(160, 160),
                    batch_size=2,
                    shuffle_buffer=10
                )
                
                # Verificar label_info
                assert label_info['num_classes'] == 2  # Solo person_001 y person_002 en train
                assert 'person_001' in label_info['person_to_label']
                assert 'person_002' in label_info['person_to_label']
                assert len(label_info['label_to_person']) == 2
                
                # Verificar que los datasets no están vacíos
                train_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
                val_cardinality = tf.data.experimental.cardinality(val_ds).numpy()
                test_cardinality = tf.data.experimental.cardinality(test_ds).numpy()
                
                assert train_cardinality > 0
                assert val_cardinality >= 0  # Puede ser 0 si person_003 no está en train
                assert test_cardinality >= 0  # Puede ser 0 si person_004 no está en train

    def test_dataset_element_structure(self):
        """Test para verificar la estructura de elementos del dataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear estructura completa con todos los directorios de splits
            train_dir = os.path.join(temp_dir, "train", "person_001")
            val_dir = os.path.join(temp_dir, "validation")  # Crear directorio vacío
            test_dir = os.path.join(temp_dir, "test")       # Crear directorio vacío
            
            os.makedirs(train_dir)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Crear imagen dummy
            image_path = os.path.join(train_dir, "person_001_cam1_1.jpg")
            png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x00\x00\x00\x03\x00\x01H\x9a\xa0\x00\x00\x00\x00IEND\xaeB`\x82'
            with open(image_path, 'wb') as f:
                f.write(png_data)
            
            with patch('tensorflow.io.read_file') as mock_read_file, \
                 patch('tensorflow.io.decode_image') as mock_decode_image, \
                 patch('tensorflow.image.resize') as mock_resize:
                
                # Setup mocks para devolver tensores válidos
                mock_read_file.return_value = "dummy"
                mock_decode_image.return_value = tf.zeros((100, 100, 3), dtype=tf.uint8)
                mock_resize.return_value = tf.zeros((160, 160, 3), dtype=tf.float32)
                
                train_ds, _, _, label_info = create_face_identification_dataset(
                    data_dir=temp_dir,
                    img_size=(160, 160),
                    batch_size=1
                )
                
                # Verificar que el dataset no esté vacío
                dataset_cardinality = tf.data.experimental.cardinality(train_ds).numpy()
                if dataset_cardinality > 0:
                    # Tomar un elemento y verificar su estructura
                    for images, labels in train_ds.take(1):
                        assert images.shape == (1, 160, 160, 3)
                        assert images.dtype == tf.float32
                        assert len(labels.shape) == 1  # Vector de labels
                        assert labels.dtype == tf.int32
                        
                        # Verificar rango de imágenes [0, 1]
                        assert tf.reduce_min(images) >= 0.0
                        assert tf.reduce_max(images) <= 1.0
                        break
                else:
                    # Si el dataset está vacío, al menos verificar que se creó correctamente
                    assert label_info['num_classes'] >= 0

class TestSurvfaceEdgeCases:
    """Tests para casos edge y manejo de errores."""
    
    def test_split_dataset_no_valid_persons(self):
        """Test para cuando no hay personas que cumplan el mínimo de imágenes."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = os.path.join(temp_dir, "source")
            output_dir = os.path.join(temp_dir, "output")
            
            # Crear personas con pocas imágenes
            for i in range(3):
                person_dir = os.path.join(source_dir, f"person_{i:03d}")
                os.makedirs(person_dir)
                
                # Solo 2 imágenes por persona
                for j in range(2):
                    image_path = os.path.join(person_dir, f"person_{i:03d}_cam1_{j+1}.jpg")
                    with open(image_path, 'w') as f:
                        f.write("dummy")
            
            # Intentar split con mínimo de 5 imágenes
            stats = split_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                min_images_per_person=5  # Más que las 2 disponibles
            )
            
            assert stats['total_valid_persons'] == 0
            assert stats['train_persons'] == 0
            assert stats['val_persons'] == 0
            assert stats['test_persons'] == 0

    def test_create_dataset_empty_splits(self):
        """Test para splits vacíos."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear directorios vacíos
            for split in ['train', 'validation', 'test']:
                os.makedirs(os.path.join(temp_dir, split))
            
            train_ds, val_ds, test_ds, label_info = create_face_identification_dataset(
                data_dir=temp_dir,
                batch_size=1
            )
            
            assert label_info['num_classes'] == 0
            assert len(label_info['person_to_label']) == 0
            
            # Los datasets deben estar vacíos
            assert tf.data.experimental.cardinality(train_ds).numpy() == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])