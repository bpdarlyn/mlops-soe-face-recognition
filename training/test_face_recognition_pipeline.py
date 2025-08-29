#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el pipeline de reconocimiento facial funciona.
"""

import os
import sys
import json
import tensorflow as tf
from pathlib import Path

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from training.datasets.survface import create_face_identification_dataset
from training.combined_face_analytics import FaceAnalyticsTrainer

def test_pipeline():
    """Prueba rápida del pipeline completo."""
    
    print("=== Test del Pipeline de Reconocimiento Facial ===")
    
    # Configuración de prueba
    SPLITS_DIR = os.path.join(ROOT, "data", "face_identification", "splits")
    
    # Verificar que existen los splits
    if not os.path.exists(SPLITS_DIR):
        print(f"❌ Error: No se encontraron los splits en {SPLITS_DIR}")
        print("Ejecuta primero: python training/datasets/survface.py")
        return False
    
    try:
        # 1. Crear datasets de prueba (batch pequeño)
        print("1. Creando datasets de prueba...")
        train_ds, val_ds, test_ds, label_info = create_face_identification_dataset(
            data_dir=SPLITS_DIR,
            img_size=(160, 160),
            batch_size=4,  # Batch pequeño para prueba
            shuffle_buffer=100
        )
        
        num_classes = label_info["num_classes"]
        print(f"   ✅ Datasets creados: {num_classes} clases")
        
        # 2. Crear datasets multi-tarea
        print("2. Convirtiendo a formato multi-tarea...")
        def add_dummy_age_gender(image, identity_label):
            batch_size = tf.shape(identity_label)[0]
            dummy_age = tf.random.uniform([batch_size], 20, 60, dtype=tf.float32)
            dummy_gender = tf.cast(tf.random.uniform([batch_size]) > 0.5, tf.float32)
            return image, {
                "identity": identity_label,
                "age": dummy_age,
                "gender": dummy_gender
            }
        
        train_multi = train_ds.map(add_dummy_age_gender)
        val_multi = val_ds.map(add_dummy_age_gender)
        print("   ✅ Datasets multi-tarea creados")
        
        # 3. Construir modelo
        print("3. Construyendo modelo...")
        trainer = FaceAnalyticsTrainer(
            num_identities=num_classes,  # Usar todas las clases
            input_shape=(160, 160, 3)
        )
        models = trainer.build_models(embedding_dim=256)  # Embeddings más pequeños
        print("   ✅ Modelo construido")
        
        # 4. Compilar modelo
        print("4. Compilando modelo...")
        from training.combined_face_analytics import compile_face_analytics_model
        model = compile_face_analytics_model(models["full_model"], learning_rate=1e-3)
        print("   ✅ Modelo compilado")
        
        # 5. Probar una predicción
        print("5. Probando predicción...")
        
        # Tomar un batch de muestra
        sample_batch = next(iter(train_multi))
        sample_images, sample_labels = sample_batch
        
        print(f"   Forma del batch: {sample_images.shape}")
        print(f"   Labels de identidad: {sample_labels['identity'].shape}")
        print(f"   Labels de edad: {sample_labels['age'].shape}")
        print(f"   Labels de género: {sample_labels['gender'].shape}")
        
        # Hacer predicción
        predictions = model(sample_images, training=False)
        print(f"   Predicción identidad: {predictions['identity'].shape}")
        print(f"   Predicción edad: {predictions['age'].shape}")
        print(f"   Predicción género: {predictions['gender'].shape}")
        print("   ✅ Predicción exitosa")
        
        # 6. Probar extractor de embeddings
        print("6. Probando extractor de embeddings...")
        embeddings = models["embedding_model"](sample_images, training=False)
        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Norma promedio: {tf.reduce_mean(tf.norm(embeddings, axis=1)):.4f}")
        print("   ✅ Embeddings extraídos")
        
        # 7. Probar un paso de entrenamiento
        print("7. Probando un paso de entrenamiento...")
        with tf.GradientTape() as tape:
            predictions = model(sample_images, training=True)
            
            # Calcular pérdidas
            identity_loss = tf.keras.losses.sparse_categorical_crossentropy(
                sample_labels["identity"], predictions["identity"]
            )
            age_loss = tf.keras.losses.mean_absolute_error(
                tf.expand_dims(sample_labels["age"], -1), predictions["age"]
            )
            gender_loss = tf.keras.losses.binary_crossentropy(
                tf.expand_dims(sample_labels["gender"], -1), predictions["gender"]
            )
            
            total_loss = tf.reduce_mean(identity_loss) + \
                        0.3 * tf.reduce_mean(age_loss) + \
                        0.3 * tf.reduce_mean(gender_loss)
        
        print(f"   Loss identidad: {tf.reduce_mean(identity_loss):.4f}")
        print(f"   Loss edad: {tf.reduce_mean(age_loss):.4f}")
        print(f"   Loss género: {tf.reduce_mean(gender_loss):.4f}")
        print(f"   Loss total: {total_loss:.4f}")
        print("   ✅ Cálculo de gradientes exitoso")
        
        # 8. Información del modelo
        print("\n8. Información del modelo:")
        total_params = model.count_params()
        print(f"   Parámetros totales: {total_params:,}")
        
        # Parámetros por componente
        embedding_params = models["embedding_model"].count_params()
        print(f"   Parámetros extractor embeddings: {embedding_params:,}")
        
        print(f"\n✅ ¡Pipeline completamente funcional!")
        print(f"✅ Listo para entrenamiento con {num_classes} identidades")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Configurar TensorFlow para usar menos memoria
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configurada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Ejecutando en CPU")
    
    success = test_pipeline()
    
    if success:
        print("\n🎉 El pipeline está listo para el entrenamiento completo!")
        print("Ejecuta: docker compose run --rm trainer python training/train_face_recognition_tf.py")
    else:
        print("\n❌ Hay problemas en el pipeline que deben resolverse.")
    
    exit(0 if success else 1)