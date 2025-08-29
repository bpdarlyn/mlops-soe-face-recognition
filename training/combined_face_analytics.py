import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
import mlflow
import mlflow.keras
from pathlib import Path

def load_pretrained_age_gender_model(model_path: str):
    """
    Carga el modelo preentrenado de edad y género.
    
    Args:
        model_path: Ruta al modelo guardado
        
    Returns:
        model: Modelo de edad y género cargado
    """
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"Modelo de edad/género no encontrado en: {model_path}")

def create_face_analytics_pipeline(
    num_identities: int,
    age_gender_model_path: str = None,
    input_shape=(160, 160, 3),
    embedding_dim=512
):
    """
    Crea un pipeline completo de análisis facial que combina:
    1. Reconocimiento de identidad (face recognition)
    2. Estimación de edad
    3. Clasificación de género
    
    Args:
        num_identities: Número de identidades conocidas
        age_gender_model_path: Ruta al modelo preentrenado de edad/género
        input_shape: Forma de la imagen de entrada
        embedding_dim: Dimensión de los embeddings faciales
        
    Returns:
        dict: Diccionario con todos los modelos del pipeline
    """
    
    # Input común
    inputs = keras.Input(shape=input_shape, name="face_image")
    
    # === Backbone compartido ===
    # Usamos MobileNetV2 como backbone compartido para eficiencia
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        alpha=1.0  # Usar la versión completa para mejor precisión
    )
    backbone.trainable = False  # Empezamos congelado
    
    # Extracción de features compartidas
    shared_features = backbone(inputs, training=False)
    shared_features = layers.GlobalAveragePooling2D(name="shared_features")(shared_features)
    
    # === Branch de reconocimiento facial ===
    # Embeddings para reconocimiento
    face_branch = layers.Dropout(0.2, name="face_dropout1")(shared_features)
    face_embeddings = layers.Dense(
        embedding_dim, 
        activation=None, 
        name="face_embeddings",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(face_branch)
    
    # Normalización L2 para embeddings
    face_embeddings_norm = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name="face_embeddings_normalized"
    )(face_embeddings)
    
    # Clasificación de identidad
    face_branch = layers.Dropout(0.3, name="face_dropout2")(face_embeddings_norm)
    identity_output = layers.Dense(
        num_identities,
        activation="softmax",
        name="identity",
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(face_branch)
    
    # === Branch de edad ===
    age_branch = layers.Dropout(0.2, name="age_dropout")(shared_features)
    age_branch = layers.Dense(256, activation="relu", name="age_dense")(age_branch)
    age_branch = layers.Dropout(0.3)(age_branch)
    age_output = layers.Dense(1, name="age", activation="linear")(age_branch)
    
    # === Branch de género ===  
    gender_branch = layers.Dropout(0.2, name="gender_dropout")(shared_features)
    gender_branch = layers.Dense(128, activation="relu", name="gender_dense")(gender_branch)
    gender_branch = layers.Dropout(0.3)(gender_branch)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender")(gender_branch)
    
    # === Modelos ===
    
    # Modelo completo para entrenamiento conjunto
    full_model = keras.Model(
        inputs=inputs,
        outputs={
            "identity": identity_output,
            "age": age_output,
            "gender": gender_output
        },
        name="face_analytics_full"
    )
    
    # Modelo solo para extraer embeddings (útil para comparación de rostros)
    embedding_model = keras.Model(
        inputs=inputs,
        outputs=face_embeddings_norm,
        name="face_embedding_extractor"
    )
    
    # Modelo solo para reconocimiento de identidad
    identity_model = keras.Model(
        inputs=inputs,
        outputs=identity_output,
        name="face_identity_classifier"
    )
    
    # Modelo solo para edad y género (útil si queremos usar el preentrenado)
    age_gender_model = keras.Model(
        inputs=inputs,
        outputs=[age_output, gender_output],
        name="age_gender_classifier"
    )
    
    return {
        "full_model": full_model,
        "embedding_model": embedding_model,
        "identity_model": identity_model,
        "age_gender_model": age_gender_model,
        "backbone": backbone
    }

def compile_face_analytics_model(model, learning_rate=1e-3, stage="joint_training"):
    """
    Compila el modelo completo con pérdidas y métricas apropiadas.
    
    Args:
        model: Modelo de Keras
        learning_rate: Tasa de aprendizaje
        stage: Etapa de entrenamiento ("joint_training", "identity_only", "fine_tune")
    """
    
    if stage == "joint_training":
        # Entrenamiento conjunto con pesos balanceados
        loss_weights = {"identity": 1.0, "age": 0.3, "gender": 0.3}
    elif stage == "identity_only":
        # Solo entrenar la parte de identidad
        loss_weights = {"identity": 1.0, "age": 0.0, "gender": 0.0}
    elif stage == "fine_tune":
        # Fine-tuning con pesos más balanceados
        loss_weights = {"identity": 0.7, "age": 0.2, "gender": 0.1}
    else:
        loss_weights = {"identity": 1.0, "age": 0.3, "gender": 0.3}
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "identity": "sparse_categorical_crossentropy",
            "age": "mae", 
            "gender": "binary_crossentropy"
        },
        loss_weights=loss_weights,
        metrics={
            "identity": [
                "accuracy", 
                metrics.SparseTopKCategoricalAccuracy(k=5, name="sparse_top_k_categorical_accuracy")
            ],
            "age": ["mae", "mse"],
            "gender": [
                "accuracy", 
                metrics.Precision(name="precision"),
                metrics.Recall(name="recall")
            ]
        }
    )
    
    return model

def transfer_age_gender_weights(target_model, source_model_path, freeze_age_gender=True):
    """
    Transfiere pesos del modelo preentrenado de edad/género al modelo combinado.
    
    Args:
        target_model: Modelo objetivo (combinado)
        source_model_path: Ruta al modelo fuente preentrenado
        freeze_age_gender: Si congelar las capas de edad/género transferidas
        
    Returns:
        target_model: Modelo con pesos transferidos
    """
    
    if not os.path.exists(source_model_path):
        print(f"Advertencia: No se encontró el modelo preentrenado en {source_model_path}")
        return target_model
    
    try:
        # Cargar modelo fuente
        source_model = keras.models.load_model(source_model_path)
        print(f"Modelo fuente cargado: {source_model.name}")
        
        # Mapear capas por nombre (esto requiere que los nombres coincidan)
        source_weights = {}
        for layer in source_model.layers:
            source_weights[layer.name] = layer.get_weights()
        
        # Transferir pesos a capas correspondientes
        transferred_layers = []
        for layer in target_model.layers:
            if layer.name in source_weights and len(source_weights[layer.name]) > 0:
                try:
                    layer.set_weights(source_weights[layer.name])
                    transferred_layers.append(layer.name)
                    
                    # Congelar si se especifica
                    if freeze_age_gender and ("age" in layer.name or "gender" in layer.name):
                        layer.trainable = False
                        
                except Exception as e:
                    print(f"No se pudo transferir pesos para {layer.name}: {e}")
        
        print(f"Pesos transferidos para {len(transferred_layers)} capas: {transferred_layers}")
        
    except Exception as e:
        print(f"Error al transferir pesos: {e}")
    
    return target_model

class FaceAnalyticsTrainer:
    """
    Clase para manejar el entrenamiento del pipeline completo de análisis facial.
    """
    
    def __init__(self, num_identities, input_shape=(160, 160, 3)):
        self.num_identities = num_identities
        self.input_shape = input_shape
        self.models = None
        self.history = None
        
    def build_models(self, embedding_dim=512):
        """Construye todos los modelos del pipeline."""
        self.models = create_face_analytics_pipeline(
            num_identities=self.num_identities,
            input_shape=self.input_shape,
            embedding_dim=embedding_dim
        )
        return self.models
    
    def train_stage1_identity_only(self, train_ds, val_ds, epochs=10, learning_rate=1e-3):
        """
        Etapa 1: Entrenar solo el reconocimiento de identidad.
        """
        print("=== Etapa 1: Entrenamiento solo de identidad ===")
        
        model = self.models["full_model"]
        model = compile_face_analytics_model(
            model, 
            learning_rate=learning_rate, 
            stage="identity_only"
        )
        
        # Verificar si hay datos de validación
        val_cardinality = tf.data.experimental.cardinality(val_ds).numpy() if val_ds else 0
        has_validation = val_cardinality > 0
        
        print(f"Datos de entrenamiento: {tf.data.experimental.cardinality(train_ds).numpy()} batches")
        print(f"Datos de validación: {val_cardinality} batches")
        
        callbacks = []
        
        if has_validation:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="val_identity_accuracy",
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_identity_accuracy",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                )
            ])
        else:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="identity_accuracy",
                    patience=7,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="identity_accuracy",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ])
        
        self.history = model.fit(
            train_ds,
            validation_data=val_ds if has_validation else None,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return self.history
    
    def train_stage2_joint(self, train_ds, val_ds, epochs=15, learning_rate=5e-4):
        """
        Etapa 2: Entrenamiento conjunto de todas las tareas.
        """
        print("=== Etapa 2: Entrenamiento conjunto ===")
        
        model = self.models["full_model"]
        model = compile_face_analytics_model(
            model,
            learning_rate=learning_rate,
            stage="joint_training"
        )
        
        # Verificar si hay datos de validación
        val_cardinality = tf.data.experimental.cardinality(val_ds).numpy() if val_ds else 0
        has_validation = val_cardinality > 0
        
        callbacks = []
        
        if has_validation:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=7,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=4,
                    min_lr=1e-7
                )
            ])
        else:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.3,
                    patience=6,
                    min_lr=1e-7
                )
            ])
        
        history2 = model.fit(
            train_ds,
            validation_data=val_ds if has_validation else None,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history2
    
    def train_stage3_fine_tune(self, train_ds, val_ds, epochs=10, learning_rate=1e-5):
        """
        Etapa 3: Fine-tuning con backbone descongelado.
        """
        print("=== Etapa 3: Fine-tuning ===")
        
        # Descongelar backbone gradualmente
        backbone = self.models["backbone"]
        backbone.trainable = True
        
        # Congelar las primeras capas, descongelar las últimas
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        
        model = self.models["full_model"]
        model = compile_face_analytics_model(
            model,
            learning_rate=learning_rate,
            stage="fine_tune"
        )
        
        # Verificar si hay datos de validación
        val_cardinality = tf.data.experimental.cardinality(val_ds).numpy() if val_ds else 0
        has_validation = val_cardinality > 0
        
        callbacks = []
        
        if has_validation:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=3,
                    min_lr=1e-8
                )
            ])
        else:
            callbacks.extend([
                keras.callbacks.EarlyStopping(
                    monitor="loss",
                    patience=8,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-8
                )
            ])
        
        history3 = model.fit(
            train_ds,
            validation_data=val_ds if has_validation else None,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history3
    
    def save_models(self, save_dir):
        """Guarda todos los modelos entrenados."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar cada modelo por separado
        for model_name, model in self.models.items():
            if model_name != "backbone":  # El backbone es parte de los otros modelos
                model_path = save_path / f"{model_name}"
                model.save(model_path)
                print(f"Modelo guardado: {model_path}")

if __name__ == "__main__":
    # Ejemplo de uso
    NUM_IDENTITIES = 1000  # Ajustar según el dataset real
    
    print("Creando pipeline de análisis facial...")
    trainer = FaceAnalyticsTrainer(num_identities=NUM_IDENTITIES)
    models = trainer.build_models()
    
    print(f"Pipeline creado con {NUM_IDENTITIES} identidades")
    print(f"Modelo completo: {models['full_model'].count_params():,} parámetros")
    
    # Mostrar arquitectura
    models["full_model"].summary()