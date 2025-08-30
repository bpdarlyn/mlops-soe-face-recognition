#!/usr/bin/env python3
"""
Script completo para deploy de modelo: encuentra el mejor run, registra el modelo,
y lo pone en producción con todas las verificaciones necesarias.
Soporta múltiples tipos de modelos.
"""

import os
import sys
import argparse
from mlflow import MlflowClient
from datetime import datetime
import subprocess

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Configuraciones para diferentes tipos de modelos
MODEL_CONFIGS = {
    "face-recognition": {
        "enabled": False,
        "experiment_name": "FaceRecognition-SurvFace",
        "model_name": "face-analytics-model",
        "primary_metric": "test_accuracy",
        "required_artifacts": {
            "models": ["full_model", "embedding_model", "identity_model", "age_gender_model"],
            "onnx": ["face_embeddings.onnx"],
            "metadata": ["label_mapping.json"]
        },
        "validation_metrics": ["test_accuracy", "test_f1", "test_top5_accuracy"]
    },
    "age-gender": {
        "experiment_name": "AgeGender-UTKFace",
        "model_name": "nn-age-gender-model",  # Debe coincidir con registered_model_name del training
        "primary_metric": "val_gender_acc",
        "required_artifacts": {
            "onnx": ["nn-age-gender.onnx"],
            "plots": ["curve_gender_acc.png", "cm_gender.png"]
        },
        "validation_metrics": ["val_gender_acc", "val_age_mae"]
    }
}

def check_mlflow_connection(client):
    """Verifica que MLflow esté disponible."""
    print("🔍 Verificando conexión con MLflow...")
    try:
        experiments = client.search_experiments()
        print(f"✅ Conexión exitosa. Experimentos encontrados: {len(experiments)}")
        return True
    except Exception as e:
        print(f"❌ Error conectando con MLflow: {e}")
        print("💡 Asegúrate de que MLflow esté ejecutándose:")
        print("   docker compose up -d mlflow")
        return False

def find_best_run(client, config):
    """
    Encuentra el mejor run basado en la configuración del modelo.
    """
    experiment_name = config["experiment_name"]
    metric = config["primary_metric"]
    required_artifacts = config["required_artifacts"]
    validation_metrics = config["validation_metrics"]
    
    print(f"🔍 Buscando el mejor run basado en '{metric}'...")
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            raise ValueError(f"Experimento '{experiment_name}' no encontrado")
        
        # Buscar runs completados con la métrica específica
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"status = 'FINISHED' AND metrics.{metric} > 0",
            order_by=[f"metrics.{metric} DESC"],
            max_results=10
        )
        
        if not runs:
            raise ValueError(f"No se encontraron runs completados con la métrica '{metric}'")
        
        # Verificar artefactos para cada run hasta encontrar uno completo
        for run in runs:
            run_id = run.info.run_id
            metric_value = run.data.metrics.get(metric, 0)
            
            print(f"  Verificando run {run_id} ({metric}: {metric_value:.4f})")
            
            artifacts_complete = True
            
            for artifact_path, expected_files in required_artifacts.items():
                try:
                    artifacts = client.list_artifacts(run_id, path=artifact_path)
                    found_files = [a.path.split('/')[-1] for a in artifacts]
                    
                    for expected_file in expected_files:
                        if expected_file not in found_files:
                            print(f"    ❌ Falta: {artifact_path}/{expected_file}")
                            artifacts_complete = False
                            break
                except:
                    artifacts_complete = False
                    break
            
            if artifacts_complete:
                print(f"  ✅ Run válido encontrado: {run_id}")
                print(f"    Métricas:")
                for key in validation_metrics:
                    value = run.data.metrics.get(key, 0)
                    print(f"      {key}: {value:.4f}")
                return run
        
        raise ValueError("No se encontró ningún run con todos los artefactos")
        
    except Exception as e:
        print(f"❌ Error buscando el mejor run: {e}")
        raise

def validate_model_quality(run, model_type, config):
    """
    Valida que el modelo cumple con criterios mínimos de calidad.
    """
    print("🔍 Validando calidad del modelo...")
    
    metrics = run.data.metrics
    checks = []
    
    if model_type == "face-recognition":
        # Validaciones para modelo de reconocimiento facial
        accuracy = metrics.get('test_accuracy', 0)
        f1 = metrics.get('test_f1', 0)
        
        # Verificar accuracy mínimo
        if accuracy >= 0.5:
            checks.append(f"✅ Accuracy ({accuracy:.4f}) >= 0.5")
        else:
            checks.append(f"❌ Accuracy ({accuracy:.4f}) < 0.5")
        
        # Verificar F1 score
        if f1 >= 0.3:
            checks.append(f"✅ F1 Score ({f1:.4f}) >= 0.3")
        else:
            checks.append(f"❌ F1 Score ({f1:.4f}) < 0.3")
        
        # Verificar overfitting
        train_acc = metrics.get('stage1_identity_accuracy_final', metrics.get('identity_accuracy_final', 0))
        if train_acc > 0 and accuracy > 0:
            diff = train_acc - accuracy
            if diff < 0.3:
                checks.append(f"✅ No overfitting detectado (diff: {diff:.4f})")
            else:
                checks.append(f"⚠️ Posible overfitting (diff: {diff:.4f})")
    
    elif model_type == "age-gender":
        # Validaciones para modelo de edad/género
        val_age_mae = metrics.get('val_age_mae', float('inf'))
        val_gender_acc = metrics.get('val_gender_acc', 0)

        # Verificar MAE de edad
        if val_age_mae <= 15.0:  # MAE razonable para edad
            checks.append(f"✅ Age MAE ({val_age_mae:.2f}) <= 8.0 años")
        else:
            checks.append(f"❌ Age MAE ({val_age_mae:.2f}) > 8.0 años")
        
        # Verificar accuracy de género
        if val_gender_acc >= 0.5:
            checks.append(f"✅ Gender Accuracy ({val_gender_acc:.4f}) >= 0.8")
        else:
            checks.append(f"❌ Gender Accuracy ({val_gender_acc:.4f}) < 0.8")
    
    for check in checks:
        print(f"  {check}")
    
    # Determinar si pasa las validaciones
    failed_checks = [c for c in checks if c.startswith("❌")]
    
    if failed_checks:
        print(f"\n❌ El modelo no pasa las validaciones de calidad")
        return False
    else:
        print(f"\n✅ El modelo pasa todas las validaciones de calidad")
        return True

def main():
    """Función principal del deploy."""
    parser = argparse.ArgumentParser(description="Deploy automatizado de modelos")
    parser.add_argument(
        "--model-type", 
        choices=["face-recognition", "age-gender"],
        default="face-recognition",
        help="Tipo de modelo a deployar"
    )
    parser.add_argument(
        "--stage",
        choices=["dev", "pre", "prod"],
        default="prod",
        help="Stage del modelo (dev, pre, prod)"
    )
    args = parser.parse_args()
    
    # Obtener configuración del modelo
    config = MODEL_CONFIGS[args.model_type]
    
    print("🚀 === Deploy Automatizado del Modelo ===")
    print(f"MLflow URI: {MLFLOW_URI}")
    print(f"Tipo de modelo: {args.model_type}")
    print(f"Experimento: {config['experiment_name']}")
    print(f"Modelo: {config['model_name']}")
    
    # Inicializar cliente
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    
    try:
        # 1. Verificar conexión
        if not check_mlflow_connection(client):
            sys.exit(1)
        
        # 2. Encontrar el mejor run
        print(f"\n📊 === Búsqueda del Mejor Run ===")
        best_run = find_best_run(client, config)
        
        # 3. Validar calidad del modelo
        print(f"\n✅ === Validación de Calidad ===")
        if not validate_model_quality(best_run, args.model_type, config):
            print("❌ El modelo no cumple los criterios mínimos de calidad")
            response = input("¿Continuar con el deploy? (y/N): ").lower()
            if response != 'y':
                print("Deploy cancelado")
                sys.exit(1)
        
        # 4. Encontrar la versión del modelo ya registrada
        print(f"\n📦 === Búsqueda de Versión del Modelo ===")
        try:
            model_name = config['model_name']
            run_id = best_run.info.run_id
            
            # Buscar la versión del modelo asociada con este run
            model_versions = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
            
            if not model_versions:
                raise ValueError(f"No se encontró versión del modelo para el run {run_id}")
            
            # Usar la primera versión encontrada (debería ser única)
            model_version = model_versions[0]
            print(f"✅ Versión encontrada: v{model_version.version} (Run: {run_id[:8]})")
            
            # Forzar uso de aliases - crear si no existe
            stage_alias = args.stage
            print(f"🚀 Desplegando en stage '{stage_alias}' usando aliases...")
            
            # Verificar si el modelo registrado existe, si no crearlo
            try:
                registered_model = client.get_registered_model(model_name)
                print(f"✅ Modelo registrado '{model_name}' existe")
            except:
                print(f"📝 Creando modelo registrado '{model_name}'...")
                client.create_registered_model(
                    name=model_name,
                    description=f"Modelo {args.model_type} entrenado automáticamente"
                )
                print(f"✅ Modelo registrado '{model_name}' creado")
            
            # Limpiar TODOS los aliases existentes (dev, pre, prod)
            all_aliases = ["dev", "pre", "prod"]
            print(f"🧹 Limpiando aliases existentes...")
            
            try:
                client.delete_model_version_alias(model_name, stage_alias)
                print(f"✅ Alias '{stage_alias}' removido")
            except:
                print(f"ℹ️  Alias '{stage_alias}' no existía")
            
            # Establecer el nuevo alias para el stage especificado
            print(f"🏷️  Estableciendo alias '{stage_alias}'...")
            client.set_registered_model_alias(
                name=model_name,
                version=model_version.version,
                alias=stage_alias
            )
            print(f"✅ Versión v{model_version.version} establecida en stage '{stage_alias}'")
            
        except Exception as e:
            print(f"❌ Error en la transición del modelo: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # 5. Verificación final
        print(f"\n🔍 === Verificación Final ===")
        try:
            model_name = config['model_name']
            stage_alias = args.stage
            
            # Buscar usando el alias del stage especificado
            deployed_model = client.get_model_version_by_alias(model_name, stage_alias)
            print(f"✅ Modelo en stage '{stage_alias}' verificado:")
            print(f"   - Nombre: {model_name}")
            print(f"   - Versión: v{deployed_model.version}")
            print(f"   - Alias: {stage_alias}")
            print(f"   - URI: models:/{model_name}/{stage_alias}")
            
            # Verificar que los artefactos sean accesibles
            run_id = deployed_model.run_id
            required_artifacts = config['required_artifacts']
            
            for artifact_path, expected_files in required_artifacts.items():
                try:
                    artifacts = client.list_artifacts(run_id, path=artifact_path)
                    found_files = [a.path.split('/')[-1] for a in artifacts]
                    
                    for expected_file in expected_files:
                        if expected_file in found_files:
                            print(f"   - {artifact_path}/{expected_file}: ✅ Disponible")
                        else:
                            print(f"   - {artifact_path}/{expected_file}: ❌ No encontrado")
                            
                except Exception as e:
                    print(f"   - Error verificando {artifact_path}: {e}")
            
        except Exception as e:
            print(f"❌ Error en verificación final: {e}")
            sys.exit(1)
        
        # 6. Información para FastAPI
        model_name = config['model_name']
        stage_alias = args.stage
        model_uri = f"models:/{model_name}/{stage_alias}"
        
        print(f"\n🎯 === Información para FastAPI ===")
        print(f"""
Para usar en tu aplicación FastAPI (stage: {stage_alias}):

```python
import mlflow

# Configurar MLflow
mlflow.set_tracking_uri("{MLFLOW_URI}")

# Cargar modelo usando alias {stage_alias}
model = mlflow.tensorflow.load_model("{model_uri}")

# O usando pyfunc para predicciones
model_pyfunc = mlflow.pyfunc.load_model("{model_uri}")
predictions = model_pyfunc.predict(input_data)
```

Modelo {args.model_type} desplegado en stage '{stage_alias}'! 🎉

Comandos de ejemplo:
- Desarrollo: python scripts/deploy_model.py --model-type age-gender --stage dev
- Pre-producción: python scripts/deploy_model.py --model-type age-gender --stage pre  
- Producción: python scripts/deploy_model.py --model-type age-gender --stage prod
        """)
        
    except Exception as e:
        print(f"❌ Error en el proceso de deploy: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()