#!/usr/bin/env python3
"""
Script para verificar y consultar modelos registrados en MLflow.
Útil para debugging y validación antes de usar en FastAPI.
"""

import os
import sys
from mlflow import MlflowClient
from datetime import datetime
import pandas as pd

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "face-analytics-model"

def list_all_registered_models(client):
    """Lista todos los modelos registrados."""
    print("=== Modelos Registrados ===")
    
    try:
        models = client.search_registered_models()
        
        if not models:
            print("No hay modelos registrados")
            return
        
        for model in models:
            print(f"\n📦 Modelo: {model.name}")
            print(f"   Descripción: {model.description or 'N/A'}")
            print(f"   Creado: {datetime.fromtimestamp(model.creation_timestamp / 1000)}")
            print(f"   Última modificación: {datetime.fromtimestamp(model.last_updated_timestamp / 1000)}")
            
            # Mostrar tags si existen
            if model.tags:
                print(f"   Tags: {model.tags}")
                
    except Exception as e:
        print(f"Error listando modelos: {e}")

def show_model_versions(client, model_name):
    """Muestra todas las versiones de un modelo específico."""
    print(f"\n=== Versiones del Modelo: {model_name} ===")
    
    try:
        # Obtener modelo registrado
        model = client.get_registered_model(model_name)
        
        # Obtener todas las versiones
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print("No hay versiones registradas para este modelo")
            return versions
        
        # Ordenar por versión
        versions.sort(key=lambda v: int(v.version), reverse=True)
        
        print(f"Total de versiones: {len(versions)}")
        
        for version in versions:
            print(f"\n🔖 Versión: v{version.version}")
            print(f"   Estado: {version.current_stage}")
            print(f"   Run ID: {version.run_id}")
            print(f"   Creado: {datetime.fromtimestamp(version.creation_timestamp / 1000)}")
            
            if version.description:
                print(f"   Descripción: {version.description[:100]}...")
            
            # Mostrar tags
            if version.tags:
                print(f"   Tags:")
                for key, value in version.tags.items():
                    print(f"     - {key}: {value}")
            
            # Mostrar alias si existen
            try:
                aliases = client.get_model_version_by_alias(model_name, "prod")
                if aliases and aliases.version == version.version:
                    print(f"   🏷️ Alias: prod")
            except:
                pass
        
        return versions
        
    except Exception as e:
        print(f"Error obteniendo versiones del modelo: {e}")
        return []

def show_production_model(client, model_name):
    """Muestra información del modelo actualmente en producción."""
    print(f"\n=== Modelo en Producción ===")
    
    try:
        # Intentar obtener por alias 'prod'
        try:
            prod_version = client.get_model_version_by_alias(model_name, "prod")
            print(f"✅ Modelo en producción (por alias 'prod'):")
            print(f"   - Nombre: {model_name}")
            print(f"   - Versión: v{prod_version.version}")
            print(f"   - Run ID: {prod_version.run_id}")
            print(f"   - URI: models:/{model_name}/prod")
            return prod_version
        except:
            pass
        
        # Intentar obtener por stage 'Production'
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if prod_versions:
            prod_version = prod_versions[0]
            print(f"✅ Modelo en producción (por stage 'Production'):")
            print(f"   - Nombre: {model_name}")
            print(f"   - Versión: v{prod_version.version}")
            print(f"   - Run ID: {prod_version.run_id}")
            print(f"   - URI: models:/{model_name}/Production")
            return prod_version
        else:
            print("❌ No hay modelo en producción")
            return None
            
    except Exception as e:
        print(f"❌ Error obteniendo modelo en producción: {e}")
        return None

def verify_model_artifacts(client, model_version):
    """Verifica que los artefactos del modelo estén disponibles."""
    print(f"\n=== Verificación de Artefactos (v{model_version.version}) ===")
    
    try:
        run_id = model_version.run_id
        
        # Artefactos esperados
        expected_artifacts = {
            "models": ["full_model", "embedding_model", "identity_model", "age_gender_model"],
            "onnx": ["face_embeddings.onnx"],
            "metadata": ["label_mapping.json"]
        }
        
        all_good = True
        
        for artifact_path, expected_files in expected_artifacts.items():
            print(f"\n📁 Verificando {artifact_path}/:")
            
            try:
                artifacts = client.list_artifacts(run_id, path=artifact_path)
                found_files = [a.path.split('/')[-1] for a in artifacts]
                
                for expected_file in expected_files:
                    if expected_file in found_files:
                        print(f"   ✅ {expected_file}")
                    else:
                        print(f"   ❌ {expected_file} (FALTANTE)")
                        all_good = False
                        
            except Exception as e:
                print(f"   ❌ Error accediendo a {artifact_path}: {e}")
                all_good = False
        
        if all_good:
            print(f"\n✅ Todos los artefactos están disponibles")
        else:
            print(f"\n❌ Faltan algunos artefactos")
            
        return all_good
        
    except Exception as e:
        print(f"Error verificando artefactos: {e}")
        return False

def show_fastapi_integration_info(client, model_name):
    """Muestra información para integrar con FastAPI."""
    print(f"\n=== Información para Integración FastAPI ===")
    
    prod_model = show_production_model(client, model_name)
    
    if not prod_model:
        print("❌ No hay modelo en producción para integrar")
        return
    
    print(f"\n📋 Código para FastAPI:")
    print(f"""
import mlflow
import mlflow.tensorflow

# Configuración del modelo
MLFLOW_URI = "{MLFLOW_URI}"
MODEL_NAME = "{model_name}"

# Opción 1: Cargar por alias
model = mlflow.tensorflow.load_model("models:/{model_name}/prod")

# Opción 2: Cargar por versión específica  
model = mlflow.tensorflow.load_model("models:/{model_name}/{prod_model.version}")

# Opción 3: Cargar por stage
model = mlflow.tensorflow.load_model("models:/{model_name}/Production")
""")

def main():
    """Función principal del script."""
    print("=== Verificación de Modelos MLflow ===")
    print(f"MLflow URI: {MLFLOW_URI}")
    
    # Inicializar cliente
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    
    try:
        # 1. Listar todos los modelos registrados
        list_all_registered_models(client)
        
        # 2. Verificar si existe el modelo específico
        try:
            model = client.get_registered_model(MODEL_NAME)
            print(f"\n✅ Modelo '{MODEL_NAME}' encontrado")
            
            # 3. Mostrar versiones
            versions = show_model_versions(client, MODEL_NAME)
            
            # 4. Mostrar modelo en producción
            prod_model = show_production_model(client, MODEL_NAME)
            
            # 5. Verificar artefactos si hay modelo en producción
            if prod_model:
                verify_model_artifacts(client, prod_model)
            
            # 6. Información para FastAPI
            show_fastapi_integration_info(client, MODEL_NAME)
            
        except Exception as e:
            print(f"\n❌ Modelo '{MODEL_NAME}' no encontrado: {e}")
            print("\n💡 Ejecuta 'register_model.py' primero para registrar un modelo")
    
    except Exception as e:
        print(f"❌ Error conectando con MLflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()