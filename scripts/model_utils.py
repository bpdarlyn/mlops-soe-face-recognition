#!/usr/bin/env python3
"""
Utilidades para gestión de modelos MLflow: rollback, limpieza, etc.
"""

import os
import sys
import argparse
from mlflow import MlflowClient
from datetime import datetime

# Configuración MLflow
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "face-analytics-model"

def rollback_to_version(client, model_name, target_version):
    """
    Hace rollback a una versión específica del modelo.
    """
    print(f"=== Rollback del Modelo ===")
    print(f"Modelo: {model_name}")
    print(f"Versión objetivo: v{target_version}")
    
    try:
        # Verificar que la versión objetivo existe
        target_model = client.get_model_version(model_name, target_version)
        print(f"✅ Versión v{target_version} encontrada")
        print(f"   Run ID: {target_model.run_id}")
        print(f"   Creado: {datetime.fromtimestamp(target_model.creation_timestamp / 1000)}")
        
        # Archivar versión actual en producción
        try:
            current_prod = client.get_latest_versions(model_name, stages=["Production"])
            for version in current_prod:
                print(f"📦 Archivando versión actual: v{version.version}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
        except Exception as e:
            print(f"⚠️ Error archivando versión actual: {e}")
        
        # Promover versión objetivo a producción
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production"
        )
        
        # Actualizar alias 'prod'
        try:
            client.set_registered_model_alias(model_name, "prod", target_version)
            print(f"🏷️ Alias 'prod' actualizado a v{target_version}")
        except Exception as e:
            print(f"⚠️ Error actualizando alias: {e}")
        
        print(f"✅ Rollback completado exitosamente a v{target_version}")
        
    except Exception as e:
        print(f"❌ Error en rollback: {e}")
        sys.exit(1)

def clean_old_versions(client, model_name, keep_versions=5):
    """
    Limpia versiones antiguas, manteniendo solo las más recientes.
    """
    print(f"=== Limpieza de Versiones Antiguas ===")
    print(f"Manteniendo las {keep_versions} versiones más recientes")
    
    try:
        # Obtener todas las versiones
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if len(versions) <= keep_versions:
            print(f"Solo hay {len(versions)} versiones, no es necesario limpiar")
            return
        
        # Ordenar por versión (más recientes primero)
        versions.sort(key=lambda v: int(v.version), reverse=True)
        
        # Identificar versiones a eliminar
        versions_to_delete = versions[keep_versions:]
        
        print(f"Versiones a eliminar: {len(versions_to_delete)}")
        
        for version in versions_to_delete:
            # No eliminar versiones en Production
            if version.current_stage == "Production":
                print(f"⚠️ Saltando v{version.version} (en Production)")
                continue
            
            try:
                print(f"🗑️ Eliminando v{version.version} (stage: {version.current_stage})")
                client.delete_model_version(model_name, version.version)
            except Exception as e:
                print(f"❌ Error eliminando v{version.version}: {e}")
        
        print("✅ Limpieza completada")
        
    except Exception as e:
        print(f"❌ Error en limpieza: {e}")

def list_model_performance(client, model_name):
    """
    Lista el rendimiento de todas las versiones del modelo.
    """
    print(f"=== Rendimiento de Versiones ===")
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        versions.sort(key=lambda v: int(v.version), reverse=True)
        
        print(f"{'Versión':<8} {'Stage':<12} {'Accuracy':<10} {'F1':<10} {'Top5':<10} {'Fecha':<12}")
        print("-" * 70)
        
        for version in versions:
            # Obtener métricas del run
            try:
                run = client.get_run(version.run_id)
                metrics = run.data.metrics
                
                accuracy = metrics.get('test_accuracy', 0)
                f1 = metrics.get('test_f1', 0)
                top5 = metrics.get('test_top5_accuracy', 0)
                date = datetime.fromtimestamp(version.creation_timestamp / 1000).strftime('%Y-%m-%d')
                
                print(f"v{version.version:<7} {version.current_stage:<12} {accuracy:<10.4f} {f1:<10.4f} {top5:<10.4f} {date}")
                
            except Exception as e:
                print(f"v{version.version:<7} {version.current_stage:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A'}")
        
    except Exception as e:
        print(f"❌ Error listando rendimiento: {e}")

def compare_versions(client, model_name, version1, version2):
    """
    Compara dos versiones del modelo.
    """
    print(f"=== Comparación de Versiones ===")
    print(f"Comparando v{version1} vs v{version2}")
    
    try:
        # Obtener información de ambas versiones
        v1 = client.get_model_version(model_name, version1)
        v2 = client.get_model_version(model_name, version2)
        
        # Obtener métricas
        run1 = client.get_run(v1.run_id)
        run2 = client.get_run(v2.run_id)
        
        metrics1 = run1.data.metrics
        metrics2 = run2.data.metrics
        
        print(f"\n📊 Métricas:")
        print(f"{'Métrica':<20} {'v' + version1:<12} {'v' + version2:<12} {'Diferencia'}")
        print("-" * 60)
        
        key_metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_top5_accuracy']
        
        for metric in key_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            diff = val2 - val1
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            
            print(f"{metric:<20} {val1:<12.4f} {val2:<12.4f} {diff_str}")
        
        # Información adicional
        print(f"\n📅 Fechas:")
        print(f"v{version1}: {datetime.fromtimestamp(v1.creation_timestamp / 1000)}")
        print(f"v{version2}: {datetime.fromtimestamp(v2.creation_timestamp / 1000)}")
        
        print(f"\n📦 Stages:")
        print(f"v{version1}: {v1.current_stage}")
        print(f"v{version2}: {v2.current_stage}")
        
    except Exception as e:
        print(f"❌ Error comparando versiones: {e}")

def main():
    """Función principal con argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Utilidades para gestión de modelos MLflow")
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Rollback
    rollback_parser = subparsers.add_parser('rollback', help='Hacer rollback a una versión específica')
    rollback_parser.add_argument('version', type=str, help='Versión objetivo para rollback')
    rollback_parser.add_argument('--model', default=MODEL_NAME, help='Nombre del modelo')
    
    # Limpieza
    clean_parser = subparsers.add_parser('clean', help='Limpiar versiones antiguas')
    clean_parser.add_argument('--keep', type=int, default=5, help='Número de versiones a mantener')
    clean_parser.add_argument('--model', default=MODEL_NAME, help='Nombre del modelo')
    
    # Performance
    perf_parser = subparsers.add_parser('performance', help='Mostrar rendimiento de versiones')
    perf_parser.add_argument('--model', default=MODEL_NAME, help='Nombre del modelo')
    
    # Comparar
    compare_parser = subparsers.add_parser('compare', help='Comparar dos versiones')
    compare_parser.add_argument('version1', type=str, help='Primera versión')
    compare_parser.add_argument('version2', type=str, help='Segunda versión')
    compare_parser.add_argument('--model', default=MODEL_NAME, help='Nombre del modelo')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Inicializar cliente
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    
    try:
        if args.command == 'rollback':
            rollback_to_version(client, args.model, args.version)
        
        elif args.command == 'clean':
            clean_old_versions(client, args.model, args.keep)
        
        elif args.command == 'performance':
            list_model_performance(client, args.model)
        
        elif args.command == 'compare':
            compare_versions(client, args.model, args.version1, args.version2)
    
    except Exception as e:
        print(f"❌ Error ejecutando comando: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()