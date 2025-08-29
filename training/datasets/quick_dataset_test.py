#!/usr/bin/env python3
"""
Script rápido para probar el procesamiento del dataset sin hacer la división completa.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from training.datasets.survface import analyze_dataset_structure

def main():
    """Función principal de prueba rápida."""
    
    print("=== Análisis Rápido del Dataset QMUL-SurvFace ===")
    
    # Configurar paths
    source_dir = os.path.join(ROOT, "data", "face_identification", "training_set")
    
    if not os.path.exists(source_dir):
        print(f"ERROR: Dataset no encontrado en {source_dir}")
        return
    
    print(f"Analizando: {source_dir}")
    
    # Análisis básico
    try:
        stats = analyze_dataset_structure(source_dir)
        
        print(f"\n=== Estadísticas del Dataset ===")
        print(f"Total de personas: {stats['total_persons']:,}")
        print(f"Total de imágenes: {stats['total_images']:,}")
        print(f"Promedio de imágenes por persona: {stats['avg_images_per_person']:.1f}")
        print(f"Mínimo de imágenes por persona: {stats['min_images_per_person']}")
        print(f"Máximo de imágenes por persona: {stats['max_images_per_person']}")
        
        # Estadísticas adicionales
        data_path = Path(source_dir)
        person_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
        # Contar personas por número de imágenes
        image_counts = {}
        for person_dir in person_dirs[:100]:  # Solo primeras 100 para velocidad
            images = list(person_dir.glob("*.jpg"))
            count = len(images)
            if count not in image_counts:
                image_counts[count] = 0
            image_counts[count] += 1
        
        print(f"\n=== Distribución de Imágenes (muestra de 100 personas) ===")
        for count in sorted(image_counts.keys()):
            print(f"  {count} imágenes: {image_counts[count]} personas")
        
        # Verificar algunas imágenes de muestra
        print(f"\n=== Muestra de Archivos ===")
        sample_person = person_dirs[0]
        sample_images = list(sample_person.glob("*.jpg"))[:5]
        print(f"Persona ID: {sample_person.name}")
        for img in sample_images:
            print(f"  - {img.name}")
        
        # Estimación para splits
        min_images = 5
        valid_persons = sum(1 for pd in person_dirs 
                          if len(list(pd.glob("*.jpg"))) >= min_images)
        
        print(f"\n=== Estimación para Splits (min {min_images} imágenes) ===")
        print(f"Personas válidas: {valid_persons:,}")
        print(f"Estimado train (70%): {int(valid_persons * 0.7):,} personas")
        print(f"Estimado val (15%): {int(valid_persons * 0.15):,} personas")
        print(f"Estimado test (15%): {int(valid_persons * 0.15):,} personas")
        
        print(f"\n✅ Dataset válido y listo para procesamiento")
        
    except Exception as e:
        print(f"❌ Error analizando dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()