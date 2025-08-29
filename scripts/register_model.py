#!/usr/bin/env python3
"""
Script para registrar y versionar el modelo de reconocimiento facial entrenado
usando MLflowClient. Busca el último run exitoso y lo registra en producción.
"""

import os
import sys
from mlflow import MlflowClient
from datetime import datetime

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "FaceRecognition-SurvFace"
MODEL_NAME = "face-analytics-model"

def find_latest_successful_run(client, experiment_name):
    """
    Encuentra el último run exitoso que contenga todos los artefactos necesarios.
    """
    print(f"Buscando experimento: {experiment_name}")
    
    # Buscar experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado")
    
    # Obtener runs ordenados por fecha (más recientes primero)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=50
    )
    
    print(f"Encontrados {len(runs)} runs completados")
    
    for run in runs:
        run_id = run.info.run_id
        print(f"\nRevisando run: {run_id}")
        print(f"  Fecha: {datetime.fromtimestamp(run.info.start_time / 1000)}")
        
        # Verificar que existan los artefactos principales
        required_artifacts = {
            "models": ["full_model", "embedding_model", "identity_model", "age_gender_model"],
            "onnx": ["face_embeddings.onnx"],
            "metadata": ["label_mapping.json"]
        }
        
        artifacts_found = True
        
        for artifact_path, expected_files in required_artifacts.items():
            try:
                artifacts = client.list_artifacts(run_id, path=artifact_path)
                found_files = [a.path.split('/')[-1] for a in artifacts]
                
                for expected_file in expected_files:
                    if expected_file not in found_files:
                        print(f"  ❌ Falta artefacto: {artifact_path}/{expected_file}")
                        artifacts_found = False
                        break
                
                if artifacts_found:
                    print(f"  ✅ Artefactos encontrados en {artifact_path}: {found_files}")
                else:
                    break
                    
            except Exception as e:
                print(f"  ❌ Error verificando {artifact_path}: {e}")
                artifacts_found = False
                break
        
        if artifacts_found:
            print(f"  🎯 Run válido encontrado: {run_id}")
            return run
    
    raise ValueError("No se encontró ningún run con todos los artefactos necesarios")

def register_model_version(client, run, model_name):
    """
    Registra una nueva versión del modelo desde el run especificado.
    """
    run_id = run.info.run_id
    
    # Preparar metadatos del modelo
    run_metrics = run.data.metrics
    run_params = run.data.params
    
    description = f"""
    Modelo de análisis facial multi-tarea entrenado el {datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')}
    
    Métricas de evaluación:
    - Test Accuracy: {run_metrics.get('test_accuracy', 'N/A')}
    - Test F1 Score: {run_metrics.get('test_f1', 'N/A')}
    - Test Top-5 Accuracy: {run_metrics.get('test_top5_accuracy', 'N/A')}
    
    Configuración del modelo:
    - Número de identidades: {run_params.get('num_classes', 'N/A')}
    - Dimensión embeddings: {run_params.get('embedding_dim', 'N/A')}
    - Imagen de entrada: {run_params.get('img_size', 'N/A')}
    """
    
    tags = {
        "stage": "prod",
        "model_type": "face_analytics",
        "framework": "tensorflow",
        "run_id": run_id,
        "training_date": datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d'),
        "test_accuracy": str(run_metrics.get('test_accuracy', 0)),
        "num_identities": str(run_params.get('num_classes', 0)),
        "embedding_dim": str(run_params.get('embedding_dim', 512))
    }
    
    print(f"\nRegistrando modelo desde run: {run_id}")
    
    # Registrar el modelo (usando el directorio de modelos como model_uri)
    model_uri = f"runs:/{run_id}/models"
    
    try:
        # Crear o actualizar el modelo registrado
        try:
            client.get_registered_model(model_name)
            print(f"  Modelo '{model_name}' ya existe, creando nueva versión...")
        except:
            print(f"  Creando nuevo modelo registrado: '{model_name}'")
            client.create_registered_model(model_name, description="Modelo de análisis facial multi-tarea")
        
        # Crear nueva versión
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id,
            description=description,
            tags=tags
        )
        
        print(f"  ✅ Modelo registrado: {model_name} v{mv.version}")
        return mv
        
    except Exception as e:
        print(f"  ❌ Error registrando modelo: {e}")
        raise

def promote_to_production(client, model_name, model_version):
    """
    Promueve la versión del modelo a producción y archiva las versiones anteriores.
    """
    print(f"\nPromoviendo {model_name} v{model_version.version} a producción...")
    
    # Archivar versiones anteriores en producción
    try:
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        for version in production_versions:
            print(f"  📦 Archivando versión anterior: v{version.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived",
                archive_existing_versions=False
            )
    except Exception as e:
        print(f"  ⚠️ Error archivando versiones anteriores: {e}")
    
    # Promover nueva versión a producción
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"  🚀 Promovido a Production: {model_name} v{model_version.version}")
        
        # Agregar alias de producción
        try:
            client.set_registered_model_alias(model_name, "prod", model_version.version)
            print(f"  🏷️ Alias 'prod' asignado a v{model_version.version}")
        except Exception as e:
            print(f"  ⚠️ Error asignando alias: {e}")
            
    except Exception as e:
        print(f"  ❌ Error promoviendo a producción: {e}")
        raise

def main():
    """
    Función principal del script.
    """
    print("=== Script de Registro y Versionado de Modelo ===")
    print(f"MLflow URI: {MLFLOW_URI}")
    print(f"Modelo: {MODEL_NAME}")
    
    # Inicializar cliente MLflow
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    
    try:
        # 1. Encontrar el último run exitoso
        print("\n1. Buscando último run exitoso...")
        latest_run = find_latest_successful_run(client, EXPERIMENT_NAME)
        
        # 2. Registrar nueva versión del modelo
        print("\n2. Registrando nueva versión del modelo...")
        model_version = register_model_version(client, latest_run, MODEL_NAME)
        
        # 3. Promover a producción
        print("\n3. Promoviendo a producción...")
        promote_to_production(client, MODEL_NAME, model_version)
        
        # 4. Mostrar resumen final
        print("\n=== Resumen Final ===")
        print(f"✅ Modelo registrado exitosamente:")
        print(f"   - Nombre: {MODEL_NAME}")
        print(f"   - Versión: v{model_version.version}")
        print(f"   - Run ID: {latest_run.info.run_id}")
        print(f"   - Estado: Production")
        print(f"   - Alias: prod")
        
        # Mostrar información para uso en FastAPI
        print(f"\n📋 Para usar en FastAPI:")
        print(f"   model_name = '{MODEL_NAME}'")
        print(f"   model_version = '{model_version.version}'")
        print(f"   model_alias = 'prod'")
        print(f"   model_uri = 'models:/{MODEL_NAME}/prod'")
        
    except Exception as e:
        print(f"\n❌ Error en el proceso: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()