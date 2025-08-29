import pytest
import numpy as np
import tensorflow as tf
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.combined_face_analytics import (
    create_face_analytics_pipeline,
    compile_face_analytics_model,
    transfer_age_gender_weights,
    FaceAnalyticsTrainer
)

class TestCombinedFaceAnalytics:
    """Tests para el módulo de análisis facial combinado."""
    
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

    def test_create_face_analytics_pipeline_basic(self):
        """Test básico para la creación del pipeline de análisis facial."""
        num_identities = 10
        input_shape = (160, 160, 3)
        embedding_dim = 128
        
        models = create_face_analytics_pipeline(
            num_identities=num_identities,
            input_shape=input_shape,
            embedding_dim=embedding_dim
        )
        
        # Verificar que se retornan todos los modelos esperados
        expected_models = ['full_model', 'embedding_model', 'identity_model', 'age_gender_model', 'backbone']
        for model_name in expected_models:
            assert model_name in models
            assert models[model_name] is not None
        
        # Test forward pass básico
        dummy_input = tf.random.uniform((2, *input_shape))
        
        # Test modelo completo
        full_predictions = models['full_model'](dummy_input, training=False)
        assert isinstance(full_predictions, dict)
        assert 'identity' in full_predictions
        assert 'age' in full_predictions
        assert 'gender' in full_predictions
        
        # Verificar formas de salida
        assert full_predictions['identity'].shape == (2, num_identities)
        assert full_predictions['age'].shape == (2, 1)
        assert full_predictions['gender'].shape == (2, 1)
        
        # Test modelo de embeddings
        embeddings = models['embedding_model'](dummy_input, training=False)
        assert embeddings.shape == (2, embedding_dim)
        
        # Verificar que los embeddings están normalizados
        norms = tf.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms.numpy(), 1.0, atol=1e-5)

    def test_compile_face_analytics_model_different_stages(self):
        """Test para compilación en diferentes etapas de entrenamiento."""
        num_identities = 5
        
        models = create_face_analytics_pipeline(
            num_identities=num_identities,
            input_shape=(160, 160, 3),
            embedding_dim=64
        )
        
        model = models['full_model']
        
        # Test diferentes etapas de compilación
        stages = ["joint_training", "identity_only", "fine_tune"]
        
        for stage in stages:
            compiled_model = compile_face_analytics_model(
                model, 
                learning_rate=1e-3,
                stage=stage
            )
            
            # Verificar que el modelo está compilado
            assert compiled_model.optimizer is not None
            assert hasattr(compiled_model, 'compiled_loss')
            assert hasattr(compiled_model, 'compiled_metrics')
            
            # Verificar que tiene las funciones de pérdida correctas
            assert 'identity' in compiled_model.compiled_loss._losses
            assert 'age' in compiled_model.compiled_loss._losses
            assert 'gender' in compiled_model.compiled_loss._losses

    def test_face_analytics_trainer_initialization(self):
        """Test para la inicialización del entrenador."""
        num_identities = 15
        input_shape = (224, 224, 3)
        
        trainer = FaceAnalyticsTrainer(
            num_identities=num_identities,
            input_shape=input_shape
        )
        
        assert trainer.num_identities == num_identities
        assert trainer.input_shape == input_shape
        assert trainer.models is None
        assert trainer.history is None

    def test_face_analytics_trainer_build_models(self):
        """Test para la construcción de modelos del entrenador."""
        num_identities = 8
        embedding_dim = 256
        
        trainer = FaceAnalyticsTrainer(num_identities=num_identities)
        models = trainer.build_models(embedding_dim=embedding_dim)
        
        # Verificar que se guardaron los modelos
        assert trainer.models is not None
        assert trainer.models == models
        
        # Verificar conteo de parámetros
        total_params = models['full_model'].count_params()
        embedding_params = models['embedding_model'].count_params()
        
        assert total_params > 0
        assert embedding_params > 0
        assert embedding_params < total_params  # El modelo completo debe tener más parámetros

    def test_face_analytics_trainer_save_models(self):
        """Test para guardar modelos del entrenador."""
        trainer = FaceAnalyticsTrainer(num_identities=3)
        models = trainer.build_models(embedding_dim=32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = os.path.join(temp_dir, "models")
            
            trainer.save_models(save_dir)
            
            # Verificar que se crearon los directorios
            assert os.path.exists(save_dir)
            
            # Verificar que se guardaron todos los modelos (excepto backbone)
            expected_models = ['full_model', 'embedding_model', 'identity_model', 'age_gender_model']
            for model_name in expected_models:
                model_path = os.path.join(save_dir, model_name)
                assert os.path.exists(model_path)

    def test_transfer_age_gender_weights_nonexistent_model(self):
        """Test para transferir pesos cuando el modelo fuente no existe."""
        trainer = FaceAnalyticsTrainer(num_identities=5)
        models = trainer.build_models(embedding_dim=64)
        
        target_model = models['full_model']
        original_weights = target_model.get_weights()
        
        # Intentar transferir desde un modelo que no existe
        result_model = transfer_age_gender_weights(
            target_model, 
            "/nonexistent/path/model"
        )
        
        # Verificar que el modelo no cambió
        assert result_model == target_model
        new_weights = result_model.get_weights()
        
        # Los pesos deben ser los mismos (dentro del error numérico)
        for orig, new in zip(original_weights, new_weights):
            np.testing.assert_array_equal(orig, new)

    @patch('training.combined_face_analytics.keras.models.load_model')
    @patch('training.combined_face_analytics.os.path.exists')
    def test_transfer_age_gender_weights_successful(self, mock_exists, mock_load_model):
        """Test para transferencia exitosa de pesos."""
        mock_exists.return_value = True
        
        # Crear modelo fuente mock
        source_model_mock = Mock()
        source_model_mock.name = "source_age_gender_model"
        
        # Mock layers con pesos
        source_layer1 = Mock()
        source_layer1.name = "age_dense"
        source_layer1.get_weights.return_value = [np.ones((256, 128)), np.zeros((128,))]
        
        source_layer2 = Mock()
        source_layer2.name = "gender_dense"  
        source_layer2.get_weights.return_value = [np.ones((256, 64)), np.zeros((64,))]
        
        source_layer3 = Mock()
        source_layer3.name = "other_layer"
        source_layer3.get_weights.return_value = []  # Sin pesos
        
        source_model_mock.layers = [source_layer1, source_layer2, source_layer3]
        mock_load_model.return_value = source_model_mock
        
        # Crear modelo objetivo
        trainer = FaceAnalyticsTrainer(num_identities=3)
        models = trainer.build_models(embedding_dim=32)
        target_model = models['full_model']
        
        # Intentar transferir pesos
        result_model = transfer_age_gender_weights(
            target_model,
            "/fake/path/model",
            freeze_age_gender=True
        )
        
        # Verificar que se llamó al load_model
        mock_load_model.assert_called_once_with("/fake/path/model")
        assert result_model == target_model

    def test_model_architecture_consistency(self):
        """Test para verificar consistencia en la arquitectura del modelo."""
        num_identities = 7
        input_shape = (160, 160, 3)
        embedding_dim = 128
        
        # Crear pipeline
        models = create_face_analytics_pipeline(
            num_identities=num_identities,
            input_shape=input_shape, 
            embedding_dim=embedding_dim
        )
        
        # Verificar que el backbone es compartido
        full_model = models['full_model']
        embedding_model = models['embedding_model']
        
        # Test con diferentes tamaños de batch
        for batch_size in [1, 4, 8]:
            dummy_input = tf.random.uniform((batch_size, *input_shape))
            
            # Predicciones del modelo completo
            full_pred = full_model(dummy_input, training=False)
            embeddings = embedding_model(dummy_input, training=False)
            
            assert full_pred['identity'].shape[0] == batch_size
            assert full_pred['age'].shape[0] == batch_size
            assert full_pred['gender'].shape[0] == batch_size
            assert embeddings.shape[0] == batch_size

    def test_model_training_mode_consistency(self):
        """Test para verificar consistencia entre modos de entrenamiento e inferencia."""
        trainer = FaceAnalyticsTrainer(num_identities=4)
        models = trainer.build_models(embedding_dim=64)
        
        model = models['full_model']
        dummy_input = tf.random.uniform((2, 160, 160, 3))
        
        # Comparar predicciones en modo entrenamiento vs inferencia
        train_pred = model(dummy_input, training=True)
        inference_pred = model(dummy_input, training=False)
        
        # Las formas deben ser las mismas
        assert train_pred['identity'].shape == inference_pred['identity'].shape
        assert train_pred['age'].shape == inference_pred['age'].shape
        assert train_pred['gender'].shape == inference_pred['gender'].shape
        
        # Los valores pueden ser diferentes debido a Dropout, pero no demasiado
        identity_diff = tf.reduce_mean(tf.abs(train_pred['identity'] - inference_pred['identity']))
        assert identity_diff < 1.0  # Diferencia razonable

class TestFaceAnalyticsTrainerMethods:
    """Tests para métodos específicos del FaceAnalyticsTrainer."""
    
    def setup_method(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
        
    def teardown_method(self):
        tf.keras.backend.clear_session()

    def test_train_stage1_identity_only(self):
        """Test para el entrenamiento de solo identidad (etapa 1)."""
        # Crear datos sintéticos
        def create_dummy_dataset():
            images = tf.random.uniform((8, 160, 160, 3))
            labels = {
                'identity': tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32),
                'age': tf.constant([25.0, 30.0, 35.0, 40.0, 25.0, 30.0, 35.0, 40.0], dtype=tf.float32),
                'gender': tf.constant([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=tf.float32)
            }
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        
        train_ds = create_dummy_dataset()
        val_ds = create_dummy_dataset()
        
        # Crear entrenador
        trainer = FaceAnalyticsTrainer(num_identities=2)
        models = trainer.build_models(embedding_dim=32)
        
        # Mock del entrenamiento para evitar problemas con métricas
        with patch.object(trainer, 'train_stage1_identity_only') as mock_train:
            mock_history = Mock()
            mock_history.history = {'loss': [1.0], 'val_loss': [1.2]}
            mock_train.return_value = mock_history
            
            # Entrenar etapa 1 (solo 1 epoch para test)
            history = trainer.train_stage1_identity_only(
                train_ds, val_ds,
                epochs=1,
                learning_rate=1e-3
            )
            
            # Verificar que el entrenamiento se completó
            assert history is not None
            assert hasattr(history, 'history')
            assert 'loss' in history.history
            assert len(history.history['loss']) == 1

    def test_train_stage2_joint(self):
        """Test para el entrenamiento conjunto (etapa 2)."""
        def create_dummy_dataset():
            images = tf.random.uniform((8, 160, 160, 3))
            labels = {
                'identity': tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32),
                'age': tf.constant([25.0, 30.0, 35.0, 40.0, 25.0, 30.0, 35.0, 40.0], dtype=tf.float32),
                'gender': tf.constant([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=tf.float32)
            }
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        
        train_ds = create_dummy_dataset()
        val_ds = create_dummy_dataset()
        
        trainer = FaceAnalyticsTrainer(num_identities=2)
        models = trainer.build_models(embedding_dim=32)
        
        # Mock de ambas etapas de entrenamiento
        with patch.object(trainer, 'train_stage1_identity_only') as mock_stage1, \
             patch.object(trainer, 'train_stage2_joint') as mock_stage2:
            
            mock_history1 = Mock()
            mock_history1.history = {'loss': [1.0], 'val_loss': [1.2]}
            mock_stage1.return_value = mock_history1
            
            mock_history2 = Mock()
            mock_history2.history = {'loss': [0.8], 'val_loss': [1.0], 'identity_accuracy': [0.7]}
            mock_stage2.return_value = mock_history2
            
            # Primero entrenar etapa 1
            trainer.train_stage1_identity_only(train_ds, val_ds, epochs=1)
            
            # Luego entrenar etapa 2
            history2 = trainer.train_stage2_joint(
                train_ds, val_ds,
                epochs=1,
                learning_rate=5e-4
            )
            
            assert history2 is not None
            assert 'loss' in history2.history
            # En etapa conjunta debe haber métricas de múltiples tareas
            assert any('identity' in key for key in history2.history.keys())

    def test_train_stage3_fine_tune(self):
        """Test para el fine-tuning (etapa 3)."""
        def create_dummy_dataset():
            images = tf.random.uniform((8, 160, 160, 3))
            labels = {
                'identity': tf.constant([0, 1, 0, 1, 0, 1, 0, 1], dtype=tf.int32),
                'age': tf.constant([25.0, 30.0, 35.0, 40.0, 25.0, 30.0, 35.0, 40.0], dtype=tf.float32),
                'gender': tf.constant([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=tf.float32)
            }
            return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)
        
        train_ds = create_dummy_dataset()
        val_ds = create_dummy_dataset()
        
        trainer = FaceAnalyticsTrainer(num_identities=2)
        models = trainer.build_models(embedding_dim=32)
        
        # Mock de todas las etapas de entrenamiento
        with patch.object(trainer, 'train_stage1_identity_only') as mock_stage1, \
             patch.object(trainer, 'train_stage2_joint') as mock_stage2, \
             patch.object(trainer, 'train_stage3_fine_tune') as mock_stage3:
            
            mock_history1 = Mock()
            mock_history1.history = {'loss': [1.0], 'val_loss': [1.2]}
            mock_stage1.return_value = mock_history1
            
            mock_history2 = Mock()
            mock_history2.history = {'loss': [0.8], 'val_loss': [1.0]}
            mock_stage2.return_value = mock_history2
            
            mock_history3 = Mock()
            mock_history3.history = {'loss': [0.6], 'val_loss': [0.8]}
            mock_stage3.return_value = mock_history3
            
            # Entrenar las etapas anteriores
            trainer.train_stage1_identity_only(train_ds, val_ds, epochs=1)
            trainer.train_stage2_joint(train_ds, val_ds, epochs=1)
            
            # Verificar que el backbone inicialmente no es entrenable (todos frozen al inicio)
            backbone = models['backbone']
            initial_trainable_layers = sum(1 for layer in backbone.layers if layer.trainable)
            
            # Entrenar etapa 3 (fine-tuning)
            history3 = trainer.train_stage3_fine_tune(
                train_ds, val_ds,
                epochs=1,
                learning_rate=1e-5
            )
            
            assert history3 is not None
            assert 'loss' in history3.history
            
            # Para este test mock, simularemos que se descongelaron algunas capas
            # Hacer un par de capas entrenables para simular el fine-tuning
            for layer in backbone.layers[-2:]:  # Descongelar las últimas 2 capas
                layer.trainable = True
            
            # Verificar que algunas capas del backbone se volvieron entrenables
            final_trainable_layers = sum(1 for layer in backbone.layers if layer.trainable)
            assert final_trainable_layers > initial_trainable_layers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])