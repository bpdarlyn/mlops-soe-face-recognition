import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import mlflow.keras

def build_face_recognition_model(num_classes: int, input_shape=(160, 160, 3)):
    """
    Construye un modelo para reconocimiento facial basado en embeddings.
    
    Args:
        num_classes: Número de identidades únicas en el dataset
        input_shape: Forma de la imagen de entrada
    
    Returns:
        model: Modelo de Keras compilado
        backbone: Modelo base (para extraer features)
    """
    
    # Augmentación de datos
    augmenter = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
    ], name="augmentation")
    
    # Input
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = augmenter(inputs)
    
    # Backbone: MobileNetV2 pretrained en ImageNet
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False  # Empezamos con backbone congelado
    
    # Feature extraction
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Feature embedding layer
    x = layers.Dropout(0.2)(x)
    embedding = layers.Dense(512, activation=None, name="embedding")(x)  # Sin activación para embeddings
    embedding_normalized = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1), 
        name="l2_normalize"
    )(embedding)
    
    # Classification head
    x = layers.Dropout(0.3)(embedding_normalized)
    predictions = layers.Dense(num_classes, activation="softmax", name="classification")(x)
    
    # Modelo completo
    model = keras.Model(inputs=inputs, outputs=predictions, name="face_recognition")
    
    # Modelo para extraer embeddings
    embedding_model = keras.Model(inputs=inputs, outputs=embedding_normalized, name="face_embedder")
    
    return model, embedding_model, backbone

def create_combined_model(face_model, age_gender_model, input_shape=(160, 160, 3)):
    """
    Combina el modelo de reconocimiento facial con el modelo de edad/género.
    
    Args:
        face_model: Modelo de reconocimiento facial
        age_gender_model: Modelo preentrenado de edad/género
        input_shape: Forma de entrada
    
    Returns:
        combined_model: Modelo combinado que predice identidad, edad y género
    """
    
    inputs = keras.Input(shape=input_shape, name="combined_input")
    
    # Extraer predicciones de identidad
    identity_pred = face_model(inputs)
    
    # Extraer predicciones de edad y género
    age_pred, gender_pred = age_gender_model(inputs)
    
    # Modelo combinado
    combined_model = keras.Model(
        inputs=inputs,
        outputs={
            "identity": identity_pred,
            "age": age_pred,
            "gender": gender_pred
        },
        name="combined_face_analytics"
    )
    
    return combined_model

def compile_face_recognition_model(model, learning_rate=1e-3):
    """
    Compila el modelo de reconocimiento facial con las métricas apropiadas.
    
    Args:
        model: Modelo de Keras
        learning_rate: Tasa de aprendizaje
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", "top_k_categorical_accuracy"]
    )
    
    return model

def compile_combined_model(model, learning_rate=1e-3):
    """
    Compila el modelo combinado con pérdidas múltiples.
    
    Args:
        model: Modelo combinado
        learning_rate: Tasa de aprendizaje
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "identity": "sparse_categorical_crossentropy",
            "age": "mae",
            "gender": "binary_crossentropy"
        },
        loss_weights={
            "identity": 1.0,
            "age": 0.3,
            "gender": 0.3
        },
        metrics={
            "identity": ["accuracy", "top_k_categorical_accuracy"],
            "age": ["mae"],
            "gender": ["accuracy"]
        }
    )
    
    return model

class ArcFaceLayer(keras.layers.Layer):
    """
    Implementación de ArcFace loss para mejorar el aprendizaje de embeddings.
    """
    
    def __init__(self, num_classes, margin=0.5, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name="W",
            shape=(input_shape[-1], self.num_classes),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Normalizar features y weights
        x = tf.nn.l2_normalize(inputs, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        
        # Calcular cosine similarity
        logits = tf.matmul(x, W)
        
        if training:
            # Aplicar ArcFace margin durante entrenamiento
            theta = tf.acos(tf.clip_by_value(logits, -1.0 + 1e-7, 1.0 - 1e-7))
            target_logits = tf.cos(theta + self.margin)
            logits = target_logits
        
        return logits * self.scale

def build_arcface_model(num_classes: int, input_shape=(160, 160, 3)):
    """
    Construye un modelo con ArcFace loss para mejor separación de embeddings.
    
    Args:
        num_classes: Número de clases
        input_shape: Forma de entrada
        
    Returns:
        model: Modelo con ArcFace
        embedding_model: Modelo para extraer embeddings
        backbone: Backbone network
    """
    
    # Augmentación
    augmenter = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.1),
    ], name="augmentation")
    
    # Input
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = augmenter(inputs)
    
    # Backbone
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False
    
    # Feature extraction
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    # Embedding layer
    embeddings = layers.Dense(512, activation=None, name="embeddings")(x)
    
    # ArcFace layer
    predictions = ArcFaceLayer(num_classes, name="arcface")(embeddings)
    
    # Modelos
    model = keras.Model(inputs=inputs, outputs=predictions, name="arcface_model")
    embedding_model = keras.Model(
        inputs=inputs, 
        outputs=tf.nn.l2_normalize(embeddings, axis=1), 
        name="embedding_extractor"
    )
    
    return model, embedding_model, backbone

def create_triplet_loss_model(num_classes: int, input_shape=(160, 160, 3)):
    """
    Crea un modelo optimizado para triplet loss.
    
    Args:
        num_classes: Número de clases
        input_shape: Forma de entrada
        
    Returns:
        model: Modelo para triplet loss
        embedding_model: Extractor de embeddings
    """
    
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    # Backbone sin augmentación para triplet loss
    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False
    
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    
    # Embedding normalizado
    embeddings = layers.Dense(128, activation=None)(x)
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
    
    model = keras.Model(inputs=inputs, outputs=embeddings, name="triplet_embedder")
    
    return model, backbone

# Función de pérdida triplet
def triplet_loss(margin=0.5):
    """
    Implementa triplet loss con margen.
    
    Args:
        margin: Margen para la pérdida triplet
        
    Returns:
        loss_fn: Función de pérdida
    """
    
    def loss_fn(y_true, y_pred):
        # y_pred debe tener forma (batch_size * 3, embedding_dim)
        # donde cada triplet son 3 embeddings consecutivos: anchor, positive, negative
        
        anchor = y_pred[0::3]
        positive = y_pred[1::3]  
        negative = y_pred[2::3]
        
        # Distancias
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Triplet loss
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)
    
    return loss_fn

if __name__ == "__main__":
    # Ejemplo de uso
    NUM_CLASSES = 1000  # Ajustar según el dataset
    
    print("Construyendo modelo de reconocimiento facial...")
    model, embedding_model, backbone = build_face_recognition_model(NUM_CLASSES)
    
    print(f"Modelo construido con {NUM_CLASSES} clases")
    print(f"Parámetros del modelo completo: {model.count_params():,}")
    print(f"Parámetros del extractor de embeddings: {embedding_model.count_params():,}")
    
    model.summary()