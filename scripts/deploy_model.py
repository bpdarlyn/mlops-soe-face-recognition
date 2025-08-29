#!/usr/bin/env python3
"""
Script completo para deploy de modelo: encuentra el mejor run, registra el modelo,
y lo pone en producción con todas las verificaciones necesarias.
"""

import os
import sys
from mlflow import MlflowClient
from datetime import datetime
import subprocess

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "FaceRecognition-SurvFace"
MODEL_NAME = "face-analytics-model"

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

def find_best_run(client, experiment_name, metric='test_accuracy'):
    """
    Encuentra el mejor run basado en una métrica específica.
    """
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
            
            # Verificar artefactos requeridos
            required_artifacts = {
                "models": ["full_model", "embedding_model", "identity_model", "age_gender_model"],
                "onnx": ["face_embeddings.onnx"],
                "metadata": ["label_mapping.json"]
            }
            
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
                key_metrics = ['test_accuracy', 'test_f1', 'test_top5_accuracy']
                for key in key_metrics:
                    value = run.data.metrics.get(key, 0)
                    print(f"      {key}: {value:.4f}")
                return run
        
        raise ValueError("No se encontró ningún run con todos los artefactos")
        
    except Exception as e:
        print(f"❌ Error buscando el mejor run: {e}")
        raise

def validate_model_quality(run, min_accuracy=0.5):
    """
    Valida que el modelo cumple con criterios mínimos de calidad.
    """
    print("🔍 Validando calidad del modelo...")
    
    metrics = run.data.metrics
    accuracy = metrics.get('test_accuracy', 0)
    f1 = metrics.get('test_f1', 0)
    
    checks = []
    
    # Verificar accuracy mínimo
    if accuracy >= min_accuracy:
        checks.append(f"✅ Accuracy ({accuracy:.4f}) >= {min_accuracy}")
    else:
        checks.append(f"❌ Accuracy ({accuracy:.4f}) < {min_accuracy}")
    
    # Verificar F1 score
    if f1 >= 0.3:  # F1 mínimo razonable
        checks.append(f"✅ F1 Score ({f1:.4f}) >= 0.3")
    else:
        checks.append(f"❌ F1 Score ({f1:.4f}) < 0.3")
    
    # Verificar que no haya overfitting extremo
    train_acc = metrics.get('stage1_identity_accuracy_final', metrics.get('identity_accuracy_final', 0))
    if train_acc > 0 and accuracy > 0:
        diff = train_acc - accuracy
        if diff < 0.3:  # Diferencia acceptable
            checks.append(f"✅ No overfitting detectado (diff: {diff:.4f})")
        else:
            checks.append(f"⚠️ Posible overfitting (diff: {diff:.4f})")
    
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
    print("🚀 === Deploy Automatizado del Modelo ===")
    print(f"MLflow URI: {MLFLOW_URI}")
    print(f"Experimento: {EXPERIMENT_NAME}")
    print(f"Modelo: {MODEL_NAME}")
    
    # Inicializar cliente
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    
    try:
        # 1. Verificar conexión
        if not check_mlflow_connection(client):
            sys.exit(1)
        
        # 2. Encontrar el mejor run
        print(f"\n📊 === Búsqueda del Mejor Run ===")
        best_run = find_best_run(client, EXPERIMENT_NAME)
        
        # 3. Validar calidad del modelo
        print(f"\n✅ === Validación de Calidad ===")
        if not validate_model_quality(best_run):
            print("❌ El modelo no cumple los criterios mínimos de calidad")
            response = input("¿Continuar con el deploy? (y/N): ").lower()
            if response != 'y':
                print("Deploy cancelado")
                sys.exit(1)
        
        # 4. Ejecutar el script de registro
        print(f"\n📦 === Registro del Modelo ===")
        try:
            # Usar el script de registro que ya creamos
            script_path = os.path.join(os.path.dirname(__file__), 'register_model.py')
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Modelo registrado exitosamente")
                print(result.stdout)
            else:
                print("❌ Error registrando modelo:")
                print(result.stderr)
                sys.exit(1)
                
        except Exception as e:
            print(f"❌ Error ejecutando script de registro: {e}")
            sys.exit(1)
        
        # 5. Verificación final
        print(f"\n🔍 === Verificación Final ===")
        try:
            prod_model = client.get_model_version_by_alias(MODEL_NAME, "prod")
            print(f"✅ Modelo en producción verificado:")
            print(f"   - Nombre: {MODEL_NAME}")
            print(f"   - Versión: v{prod_model.version}")
            print(f"   - URI: models:/{MODEL_NAME}/prod")
            
            # Verificar que los artefactos sean accesibles
            run_id = prod_model.run_id
            artifacts = client.list_artifacts(run_id, path="onnx")
            onnx_found = any(a.path.endswith("face_embeddings.onnx") for a in artifacts)
            
            if onnx_found:
                print(f"   - ONNX: ✅ Disponible")
            else:
                print(f"   - ONNX: ❌ No encontrado")
            
        except Exception as e:
            print(f"❌ Error en verificación final: {e}")
            sys.exit(1)
        
        # 6. Información para FastAPI
        print(f"\n🎯 === Información para FastAPI ===")
        print(f"""
Para usar en tu aplicación FastAPI:

```python
import mlflow

# Configurar MLflow
mlflow.set_tracking_uri("{MLFLOW_URI}")

# Cargar modelo
model = mlflow.tensorflow.load_model("models:/{MODEL_NAME}/prod")

# O usando pyfunc para predicciones
model_pyfunc = mlflow.pyfunc.load_model("models:/{MODEL_NAME}/prod")
predictions = model_pyfunc.predict(input_data)
```

Modelo listo para producción! 🎉
        """)
        
    except Exception as e:
        print(f"❌ Error en el proceso de deploy: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()