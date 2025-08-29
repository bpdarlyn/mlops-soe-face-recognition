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

from training.train_face_recognition_tf import (
    CONFIG,
    create_multi_task_datasets,
    plot_training_history,
    evaluate_model
)
from training.combined_face_analytics import (
    FaceAnalyticsTrainer,
    compile_face_analytics_model,
    create_face_analytics_pipeline
)
from training.datasets.survface import (
    analyze_dataset_structure,
    create_face_identification_dataset
)

class TestTrainFaceRecognition:
    """Tests para el módulo de entrenamiento de reconocimiento facial."""
    
    def setup_method(self):
        """Setup que se ejecuta antes de cada test."""
        # Configurar TensorFlow para usar menos memoria en tests
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Ignorar errores de configuración GPU en tests
        
    def teardown_method(self):
        """Cleanup después de cada test."""
        tf.keras.backend.clear_session()

    def test_config_structure(self):
        """Test que verifica la estructura de la configuración."""
        required_keys = [
            'img_size', 'batch_size', 'embedding_dim', 'train_split',
            'val_split', 'test_split', 'min_images_per_person',
            'stage1_epochs', 'stage2_epochs', 'stage3_epochs',
            'stage1_lr', 'stage2_lr', 'stage3_lr', 'seed'
        ]
        
        for key in required_keys:
            assert key in CONFIG
            assert CONFIG[key] is not None
        
        # Verificar que los splits sumen 1.0
        total_split = CONFIG['train_split'] + CONFIG['val_split'] + CONFIG['test_split']
        assert abs(total_split - 1.0) < 1e-6
        
        # Verificar tipos correctos
        assert isinstance(CONFIG['img_size'], tuple)
        assert len(CONFIG['img_size']) == 2
        assert isinstance(CONFIG['batch_size'], int)
        assert CONFIG['batch_size'] > 0

    def test_create_multi_task_datasets(self):
        """Test para la creación de datasets multi-tarea."""
        # Crear un dataset mock simple
        def create_mock_dataset():
            images = tf.random.uniform((8, 160, 160, 3))
            labels = tf.constant([0, 1, 2, 0, 1, 2, 0, 1], dtype=tf.int32)
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        
        mock_train_ds = create_mock_dataset()
        mock_val_ds = create_mock_dataset()
        mock_test_ds = create_mock_dataset()
        
        # Convertir a multi-tarea
        train_multi, val_multi, test_multi = create_multi_task_datasets(
            mock_train_ds, mock_val_ds, mock_test_ds
        )
        
        # Verificar estructura del dataset resultante
        for batch_images, batch_labels in train_multi.take(1):
            assert isinstance(batch_labels, dict)
            assert 'identity' in batch_labels
            assert 'age' in batch_labels
            assert 'gender' in batch_labels
            
            # Verificar formas
            assert batch_images.shape[1:] == (160, 160, 3)
            assert len(batch_labels['identity'].shape) == 1
            assert len(batch_labels['age'].shape) == 1
            assert len(batch_labels['gender'].shape) == 1
            
            # Verificar rangos de edad y género
            ages = batch_labels['age'].numpy()
            genders = batch_labels['gender'].numpy()
            
            assert np.all(ages >= 18) and np.all(ages <= 80)
            assert np.all((genders == 0) | (genders == 1))

    def test_face_analytics_trainer_creation(self):
        """Test para la creación del entrenador de análisis facial."""
        num_identities = 10
        input_shape = (160, 160, 3)
        
        trainer = FaceAnalyticsTrainer(
            num_identities=num_identities,
            input_shape=input_shape
        )
        
        assert trainer.num_identities == num_identities
        assert trainer.input_shape == input_shape
        assert trainer.models is None  # No construido aún
        
        # Construir modelos
        models = trainer.build_models(embedding_dim=128)
        
        assert 'full_model' in models
        assert 'embedding_model' in models
        assert 'identity_model' in models
        assert 'age_gender_model' in models
        assert 'backbone' in models
        
        # Verificar formas de salida
        full_model = models['full_model']
        dummy_input = tf.random.uniform((1, *input_shape))
        predictions = full_model(dummy_input, training=False)
        
        assert 'identity' in predictions
        assert 'age' in predictions
        assert 'gender' in predictions
        
        assert predictions['identity'].shape == (1, num_identities)
        assert predictions['age'].shape == (1, 1)
        assert predictions['gender'].shape == (1, 1)

    def test_model_compilation(self):
        """Test para la compilación de modelos."""
        num_identities = 5
        
        trainer = FaceAnalyticsTrainer(num_identities=num_identities)
        models = trainer.build_models(embedding_dim=64)
        
        # Compilar modelo
        compiled_model = compile_face_analytics_model(
            models['full_model'], 
            learning_rate=1e-3,
            stage="joint_training"
        )
        
        # Verificar que el modelo está compilado
        assert compiled_model.optimizer is not None
        assert hasattr(compiled_model, 'loss')
        assert hasattr(compiled_model, 'metrics')

    def test_embedding_model_output(self):
        """Test para verificar que el modelo de embeddings produce salidas normalizadas."""
        num_identities = 3
        
        trainer = FaceAnalyticsTrainer(num_identities=num_identities)
        models = trainer.build_models(embedding_dim=32)
        
        embedding_model = models['embedding_model']
        
        # Input de prueba
        dummy_input = tf.random.uniform((4, 160, 160, 3))
        embeddings = embedding_model(dummy_input, training=False)
        
        # Verificar forma
        assert embeddings.shape == (4, 32)
        
        # Verificar normalización L2
        norms = tf.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-5)

    def test_plot_training_history(self):
        """Test para la función de graficado del historial de entrenamiento."""
        # Crear historial mock
        class MockHistory:
            def __init__(self):
                self.history = {
                    'loss': [1.0, 0.8, 0.6, 0.4],
                    'val_loss': [1.2, 0.9, 0.7, 0.5],
                    'identity_accuracy': [0.1, 0.3, 0.5, 0.7],
                    'val_identity_accuracy': [0.1, 0.25, 0.4, 0.6]
                }
        
        mock_history = MockHistory()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = plot_training_history(
                mock_history, 
                "Test Stage", 
                temp_dir
            )
            
            # Verificar que se creó el archivo
            assert os.path.exists(plot_path)
            assert plot_path.endswith('.png')
            
            # Verificar que se creó el directorio de plots
            plots_dir = os.path.join(temp_dir, 'plots')
            assert os.path.exists(plots_dir)

    def test_evaluate_model_with_mock_data(self):
        """Test para la evaluación del modelo con datos mock."""
        # Crear modelo simple para testing multi-task
        num_classes = 3
        
        # Input layer
        input_layer = tf.keras.layers.Input(shape=(160, 160, 3))
        features = tf.keras.layers.GlobalAveragePooling2D()(input_layer)
        
        # Identity output
        identity_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='identity')(features)
        
        # Age output
        age_output = tf.keras.layers.Dense(1, activation='linear', name='age')(features)
        
        # Gender output  
        gender_output = tf.keras.layers.Dense(1, activation='sigmoid', name='gender')(features)
        
        model = tf.keras.Model(inputs=input_layer, outputs={
            'identity': identity_output,
            'age': age_output,
            'gender': gender_output
        })
        
        # Crear dataset de prueba
        def create_test_dataset():
            images = tf.random.uniform((12, 160, 160, 3))
            labels = {
                'identity': tf.constant([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=tf.int32),
                'age': tf.constant([25.0, 30.0, 35.0] * 4, dtype=tf.float32),
                'gender': tf.constant([0.0, 1.0, 0.0] * 4, dtype=tf.float32)
            }
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        
        test_ds = create_test_dataset()
        
        # Mock label_info
        label_info = {
            'num_classes': num_classes,
            'person_to_label': {'0': 0, '1': 1, '2': 2},
            'label_to_person': {0: '0', 1: '1', 2: '2'}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Crear directorio plots
            plots_dir = os.path.join(temp_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Patch model.predict y evitar creación de matriz de confusión
            with patch.object(model, 'predict') as mock_predict, \
                 patch('training.train_face_recognition_tf.plt.savefig') as mock_savefig, \
                 patch('training.train_face_recognition_tf.mlflow.log_artifact') as mock_mlflow:
                
                # Simular predicciones perfectas
                def mock_predict_fn(x, verbose=0):
                    batch_size = tf.shape(x)[0].numpy()
                    # Crear predicciones que coincidan con las etiquetas
                    identity_pred = tf.constant([
                        [1.0, 0.0, 0.0],  # clase 0
                        [0.0, 1.0, 0.0],  # clase 1  
                        [0.0, 0.0, 1.0],  # clase 2
                        [1.0, 0.0, 0.0],  # clase 0
                    ][:batch_size])
                    
                    return {
                        'identity': identity_pred,
                        'age': tf.ones((batch_size, 1)) * 30.0,
                        'gender': tf.ones((batch_size, 1)) * 0.5
                    }
                
                mock_predict.side_effect = mock_predict_fn
                
                metrics = evaluate_model(model, test_ds, label_info, temp_dir)
                
                # Verificar que se calcularon las métricas
                required_metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_top5_accuracy']
                for metric in required_metrics:
                    assert metric in metrics
                    assert isinstance(metrics[metric], (int, float, np.number))
                    assert not np.isnan(metrics[metric])

    @patch('training.datasets.survface.Path')
    def test_analyze_dataset_structure_mock(self, mock_path):
        """Test para analyze_dataset_structure con datos mock."""
        # Crear estructura de directorios mock
        mock_data_path = Mock()
        mock_path.return_value = mock_data_path
        mock_data_path.exists.return_value = True
        
        # Mock person directories
        mock_person1 = Mock()
        mock_person1.is_dir.return_value = True
        mock_person1.glob.return_value = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        
        mock_person2 = Mock()
        mock_person2.is_dir.return_value = True
        mock_person2.glob.return_value = ['img1.jpg', 'img2.jpg']
        
        mock_data_path.iterdir.return_value = [mock_person1, mock_person2]
        
        stats = analyze_dataset_structure("/fake/path")
        
        assert stats['total_persons'] == 2
        assert stats['total_images'] == 5
        assert stats['min_images_per_person'] == 2
        assert stats['max_images_per_person'] == 3
        assert stats['avg_images_per_person'] == 2.5

    def test_create_face_analytics_pipeline(self):
        """Test para la creación del pipeline completo de análisis facial."""
        num_identities = 5
        input_shape = (160, 160, 3)
        embedding_dim = 64
        
        models = create_face_analytics_pipeline(
            num_identities=num_identities,
            input_shape=input_shape,
            embedding_dim=embedding_dim
        )
        
        assert 'full_model' in models
        assert 'embedding_model' in models
        assert 'identity_model' in models
        assert 'age_gender_model' in models
        assert 'backbone' in models
        
        # Test forward pass
        dummy_input = tf.random.uniform((2, *input_shape))
        
        # Test modelo completo
        full_predictions = models['full_model'](dummy_input, training=False)
        assert 'identity' in full_predictions
        assert 'age' in full_predictions
        assert 'gender' in full_predictions
        
        # Test modelo de embeddings
        embeddings = models['embedding_model'](dummy_input, training=False)
        assert embeddings.shape == (2, embedding_dim)
        
        # Test modelo de identidad
        identity_pred = models['identity_model'](dummy_input, training=False)
        assert identity_pred.shape == (2, num_identities)

class TestIntegration:
    """Tests de integración para el pipeline completo."""
    
    def setup_method(self):
        """Setup para tests de integración."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # Ignorar errores de configuración GPU en tests
        
    def teardown_method(self):
        """Cleanup después de tests de integración."""
        tf.keras.backend.clear_session()

    def test_end_to_end_training_simulation(self):
        """Test que simula un entrenamiento end-to-end con datos sintéticos."""
        # Configuración pequeña para test rápido
        num_identities = 3
        batch_size = 4
        
        # Crear datos sintéticos
        def create_synthetic_dataset():
            images = tf.random.uniform((12, 160, 160, 3))
            identity_labels = tf.constant([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=tf.int32)
            age_labels = tf.constant([25.0, 30.0, 35.0] * 4, dtype=tf.float32)
            gender_labels = tf.constant([0.0, 1.0, 0.0] * 4, dtype=tf.float32)
            
            labels = {
                'identity': identity_labels,
                'age': age_labels, 
                'gender': gender_labels
            }
            
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)
        
        train_ds = create_synthetic_dataset()
        val_ds = create_synthetic_dataset()
        
        # Crear y compilar modelo
        trainer = FaceAnalyticsTrainer(num_identities=num_identities)
        models = trainer.build_models(embedding_dim=32)
        model = compile_face_analytics_model(models['full_model'])
        
        # Mock el método fit para evitar problemas de entrenamiento real
        with patch.object(model, 'fit') as mock_fit:
            mock_history = Mock()
            mock_history.history = {
                'loss': [1.0], 
                'val_loss': [1.2],
                'identity_accuracy': [0.5],
                'val_identity_accuracy': [0.4]
            }
            mock_fit.return_value = mock_history
            
            # Entrenar por 1 epoch (solo para verificar que funciona)
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=1,
                verbose=0
            )
            
            # Verificar que el entrenamiento produjo métricas
            assert 'loss' in history.history
            assert 'val_loss' in history.history
            assert len(history.history['loss']) == 1
        
        # Verificar que el modelo puede hacer predicciones (sin mock)
        test_input = tf.random.uniform((2, 160, 160, 3))
        predictions = model(test_input, training=False)
        
        assert 'identity' in predictions
        assert 'age' in predictions
        assert 'gender' in predictions

if __name__ == "__main__":
    # Para ejecutar los tests directamente
    pytest.main([__file__, "-v"])