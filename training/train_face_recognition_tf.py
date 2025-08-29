#!/usr/bin/env python3
"""
Script principal para entrenar el modelo de reconocimiento facial integrado
con análisis de edad y género usando el dataset QMUL-SurvFace.
"""

import os
import sys
import json
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from training.datasets.survface import (
    split_dataset, 
    create_face_identification_dataset,
    analyze_dataset_structure
)
from training.combined_face_analytics import (
    FaceAnalyticsTrainer,
    create_face_analytics_pipeline,
    transfer_age_gender_weights
)

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("FaceRecognition-SurvFace")

# Configuración de paths
DATA_ROOT = os.path.join(ROOT, "data", "face_identification")
ORIGINAL_DATASET = os.path.join(DATA_ROOT, "training_set")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
AGE_GENDER_MODEL_PATH = os.path.join(ROOT, "artifacts", "age_gender_savedmodel")

# Hiperparámetros
CONFIG = {
    "img_size": (160, 160),
    "batch_size": 32,
    "embedding_dim": 512,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "min_images_per_person": 5,
    "stage1_epochs": 1,
    "stage2_epochs": 1,
    "stage3_epochs": 1,
    "stage1_lr": 1e-3,
    "stage2_lr": 5e-4,
    "stage3_lr": 1e-5,
    "seed": 42
}

def prepare_dataset():
    """Prepara y divide el dataset si no existe."""
    
    print("=== Preparación del Dataset ===")
    
    # Verificar si ya existe la división
    if os.path.exists(SPLITS_DIR) and os.path.exists(os.path.join(SPLITS_DIR, "split_metadata.json")):
        print("División del dataset ya existe, cargando metadatos...")
        with open(os.path.join(SPLITS_DIR, "split_metadata.json"), "r") as f:
            metadata = json.load(f)
        return metadata
    
    # Analizar dataset original
    print("Analizando estructura del dataset original...")
    stats = analyze_dataset_structure(ORIGINAL_DATASET)
    print(f"Dataset original: {stats['total_persons']} personas, {stats['total_images']} imágenes")
    
    # Realizar división
    print("Dividiendo dataset...")
    split_stats = split_dataset(
        source_dir=ORIGINAL_DATASET,
        output_dir=SPLITS_DIR,
        train_split=CONFIG["train_split"],
        val_split=CONFIG["val_split"],
        test_split=CONFIG["test_split"],
        min_images_per_person=CONFIG["min_images_per_person"],
        seed=CONFIG["seed"]
    )
    
    print("División completada:")
    for key, value in split_stats.items():
        print(f"  {key}: {value}")
    
    return split_stats

def create_datasets(metadata):
    """Crea los datasets de TensorFlow."""
    
    print("=== Creación de Datasets ===")
    
    train_ds, val_ds, test_ds, label_info = create_face_identification_dataset(
        data_dir=SPLITS_DIR,
        img_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        shuffle_buffer=2000
    )
    
    print(f"Datasets creados:")
    print(f"  Clases de identidad: {label_info['num_classes']}")
    print(f"  Train batches: {tf.data.experimental.cardinality(train_ds)}")
    print(f"  Validation batches: {tf.data.experimental.cardinality(val_ds)}")
    print(f"  Test batches: {tf.data.experimental.cardinality(test_ds)}")
    
    return train_ds, val_ds, test_ds, label_info

def create_multi_task_datasets(train_ds, val_ds, test_ds):
    """
    Convierte los datasets de identificación a formato multi-tarea.
    Para esto necesitamos generar datos sintéticos de edad y género o usar un modelo preentrenado.
    """
    
    def add_dummy_age_gender(image, identity_label):
        """Añade etiquetas dummy para edad y género mientras no tengamos datos reales."""
        # Por ahora usamos valores aleatorios - en un caso real usarías etiquetas verdaderas
        batch_size = tf.shape(identity_label)[0]
        dummy_age = tf.random.uniform([batch_size], 18, 80, dtype=tf.float32)
        dummy_gender = tf.cast(tf.random.uniform([batch_size]) > 0.5, tf.float32)
        
        return image, {
            "identity": identity_label,
            "age": dummy_age,
            "gender": dummy_gender
        }
    
    # Convertir datasets
    train_multi = train_ds.map(add_dummy_age_gender)
    val_multi = val_ds.map(add_dummy_age_gender)
    test_multi = test_ds.map(add_dummy_age_gender)
    
    return train_multi, val_multi, test_multi

def plot_training_history(history, stage_name, save_dir):
    """Crea gráficos del historial de entrenamiento."""
    
    # Crear directorio de plots
    plots_dir = Path(save_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Métricas a graficar
    metrics = {
        "loss": "Loss",
        "identity_accuracy": "Identity Accuracy",
        "identity_sparse_top_k_categorical_accuracy": "Identity Top-5 Accuracy"
    }
    
    # Si tenemos métricas de edad y género
    if "age_mae" in history.history:
        metrics["age_mae"] = "Age MAE"
    if "gender_accuracy" in history.history:
        metrics["gender_accuracy"] = "Gender Accuracy"
    
    # Crear subplot para cada métrica
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 8))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_key, metric_name) in enumerate(metrics.items()):
        ax = axes[idx]
        
        if metric_key in history.history:
            ax.plot(history.history[metric_key], label=f'Train {metric_name}')
        
        val_key = f"val_{metric_key}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label=f'Validation {metric_name}')
        
        ax.set_title(f'{metric_name} - {stage_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True)
    
    # Ocultar ejes no usados
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plot_path = plots_dir / f"{stage_name.lower().replace(' ', '_')}_history.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return str(plot_path)

def evaluate_model(model, test_ds, label_info, save_dir):
    """Evalúa el modelo en el conjunto de prueba."""
    
    print("=== Evaluación del Modelo ===")
    
    # Verificar si hay datos de prueba
    test_cardinality = tf.data.experimental.cardinality(test_ds).numpy()
    if test_cardinality <= 0:
        print("⚠️ No hay datos de prueba disponibles. Saltando evaluación.")
        return {
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_top5_accuracy": 0.0
        }
    
    print(f"Evaluando en {test_cardinality} batches de prueba...")
    
    # Predicciones
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for batch_x, batch_y in test_ds:
        if isinstance(batch_y, dict):
            # Multi-task dataset
            predictions = model.predict(batch_x, verbose=0)
            if isinstance(predictions, dict):
                y_pred_batch = np.argmax(predictions["identity"], axis=1)
                y_pred_proba.append(predictions["identity"])
            else:
                y_pred_batch = np.argmax(predictions, axis=1)
                y_pred_proba.append(predictions)
            y_true_batch = batch_y["identity"].numpy()
        else:
            # Single task dataset
            predictions = model.predict(batch_x, verbose=0)
            y_pred_batch = np.argmax(predictions, axis=1)
            y_pred_proba.append(predictions)
            y_true_batch = batch_y.numpy()
        
        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)
    
    # Verificar que tenemos datos
    if len(y_true) == 0:
        print("⚠️ No se obtuvieron datos de evaluación.")
        return {
            "test_accuracy": 0.0,
            "test_precision": 0.0,
            "test_recall": 0.0,
            "test_f1": 0.0,
            "test_top5_accuracy": 0.0
        }
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.vstack(y_pred_proba)
    
    # Métricas
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Usar zero_division='warn' para manejar casos sin algunas clases
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Top-k accuracy
    def top_k_accuracy(y_true, y_pred_proba, k=5):
        if y_pred_proba.shape[1] < k:
            k = y_pred_proba.shape[1]  # Ajustar k si hay menos clases
        top_k_pred = np.argsort(y_pred_proba, axis=1)[:, -k:]
        return np.mean([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
    
    top5_accuracy = top_k_accuracy(y_true, y_pred_proba, k=min(5, y_pred_proba.shape[1]))
    
    metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_top5_accuracy": top5_accuracy
    }
    
    print("Métricas de evaluación:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Matriz de confusión (solo para un subset por tamaño)
    unique_labels = np.unique(y_true)
    if len(unique_labels) <= 50:  # Solo si no hay demasiadas clases
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            # Crear directorio si no existe
            plots_dir = Path(save_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            plt.figure(figsize=(12, 10))
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            cm_path = plots_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            print(f"Matriz de confusión guardada: {cm_path}")
            
        except Exception as e:
            print(f"⚠️ Error creando matriz de confusión: {e}")
    
    return metrics

def main():
    """Función principal de entrenamiento."""
    
    print("=== Iniciando Entrenamiento de Reconocimiento Facial ===")
    print(f"Configuración: {CONFIG}")
    
    # Preparar dataset
    metadata = prepare_dataset()
    
    # Crear datasets
    train_ds, val_ds, test_ds, label_info = create_datasets(metadata)
    num_classes = label_info["num_classes"]
    
    # Para entrenamiento multi-tarea, convertir datasets
    train_multi, val_multi, test_multi = create_multi_task_datasets(train_ds, val_ds, test_ds)
    
    # Iniciar experimento MLflow
    with mlflow.start_run(run_name=f"face_recognition_{num_classes}_classes") as run:
        
        # Log de parámetros
        mlflow.log_params(CONFIG)
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_params(metadata)
        
        # Crear entrenador
        trainer = FaceAnalyticsTrainer(
            num_identities=num_classes,
            input_shape=(*CONFIG["img_size"], 3)
        )
        
        # Construir modelos
        models = trainer.build_models(embedding_dim=CONFIG["embedding_dim"])
        
        # Transferir pesos de edad/género si existe modelo preentrenado
        if os.path.exists(AGE_GENDER_MODEL_PATH):
            print("Transfiriendo pesos del modelo de edad/género...")
            models["full_model"] = transfer_age_gender_weights(
                models["full_model"], 
                AGE_GENDER_MODEL_PATH
            )
        
        # === Etapa 1: Solo identidad ===
        print(f"\n=== ETAPA 1: Entrenamiento de Identidad (Solo) ===")
        history1 = trainer.train_stage1_identity_only(
            train_multi, val_multi,
            epochs=CONFIG["stage1_epochs"],
            learning_rate=CONFIG["stage1_lr"]
        )

        # Log métricas de etapa 1
        for metric, values in history1.history.items():
            mlflow.log_metric(f"stage1_{metric}_final", float(values[-1]))

        # Graficar etapa 1
        plot_path = plot_training_history(history1, "Stage 1 - Identity Only", "artifacts")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # === Etapa 2: Entrenamiento conjunto ===
        print(f"\n=== ETAPA 2: Entrenamiento Conjunto ===")
        history2 = trainer.train_stage2_joint(
            train_multi, val_multi,
            epochs=CONFIG["stage2_epochs"],
            learning_rate=CONFIG["stage2_lr"]
        )

        # Log métricas de etapa 2
        for metric, values in history2.history.items():
            mlflow.log_metric(f"stage2_{metric}_final", float(values[-1]))

        # Graficar etapa 2
        plot_path = plot_training_history(history2, "Stage 2 - Joint Training", "artifacts")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # === Etapa 3: Fine-tuning ===
        print(f"\n=== ETAPA 3: Fine-tuning ===")
        history3 = trainer.train_stage3_fine_tune(
            train_multi, val_multi,
            epochs=CONFIG["stage3_epochs"],
            learning_rate=CONFIG["stage3_lr"]
        )

        # Log métricas de etapa 3
        for metric, values in history3.history.items():
            mlflow.log_metric(f"stage3_{metric}_final", float(values[-1]))

        # Graficar etapa 3
        plot_path = plot_training_history(history3, "Stage 3 - Fine Tuning", "artifacts")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        
        # === Evaluación final ===
        eval_metrics = evaluate_model(
            models["full_model"], 
            test_multi, 
            label_info, 
            "artifacts"
        )
        
        # Log métricas de evaluación
        for metric, value in eval_metrics.items():
            mlflow.log_metric(metric, value)
        
        # === Guardar modelos ===
        print("\n=== Guardando Modelos ===")
        
        # Crear directorio de artefactos
        os.makedirs("artifacts", exist_ok=True)
        
        # Guardar todos los modelos
        trainer.save_models("artifacts/face_analytics_models")
        
        # Log de modelos en MLflow
        mlflow.log_artifacts("artifacts/face_analytics_models", artifact_path="models")
        
        # Exportar modelo de embeddings a ONNX
        try:
            import tf2onnx
            
            embedding_model = models["embedding_model"]
            spec = (tf.TensorSpec((None, *CONFIG["img_size"], 3), tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(embedding_model, input_signature=spec, opset=13)
            
            onnx_path = "artifacts/face_embeddings.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            mlflow.log_artifact(onnx_path, artifact_path="onnx")
            print(f"Modelo ONNX guardado: {onnx_path}")
            
        except ImportError:
            print("tf2onnx no disponible, saltando exportación ONNX")
        
        # Guardar mapeo de etiquetas
        label_mapping_path = "artifacts/label_mapping.json"
        with open(label_mapping_path, "w") as f:
            json.dump(label_info, f, indent=2)
        mlflow.log_artifact(label_mapping_path, artifact_path="metadata")
        
        print(f"\n=== Entrenamiento Completado ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"Modelos guardados en artifacts/")
        print(f"Accuracy final: {eval_metrics['test_accuracy']:.4f}")
        print(f"Top-5 Accuracy: {eval_metrics['test_top5_accuracy']:.4f}")

if __name__ == "__main__":
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU disponible: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("Ejecutando en CPU")
    
    # Ejecutar entrenamiento
    main()