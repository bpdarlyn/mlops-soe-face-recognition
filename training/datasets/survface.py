import os
import random
from pathlib import Path
import shutil
from typing import Tuple, List
import tensorflow as tf

def analyze_dataset_structure(data_dir: str) -> dict:
    """
    Analiza la estructura del dataset QMUL-SurvFace
    
    Returns:
        dict: Estadísticas del dataset incluyendo número de personas e imágenes
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    person_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    person_count = len(person_dirs)
    
    total_images = 0
    min_images_per_person = float('inf')
    max_images_per_person = 0
    
    for person_dir in person_dirs:
        images = list(person_dir.glob("*.jpg"))
        num_images = len(images)
        total_images += num_images
        min_images_per_person = min(min_images_per_person, num_images)
        max_images_per_person = max(max_images_per_person, num_images)
    
    avg_images_per_person = total_images / person_count if person_count > 0 else 0
    
    return {
        "total_persons": person_count,
        "total_images": total_images,
        "min_images_per_person": min_images_per_person,
        "max_images_per_person": max_images_per_person,
        "avg_images_per_person": avg_images_per_person
    }

def split_dataset(
    source_dir: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    min_images_per_person: int = 3,
    seed: int = 42
) -> dict:
    """
    Divide el dataset QMUL-SurvFace en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        source_dir: Directorio con el dataset original
        output_dir: Directorio donde guardar los splits
        train_split: Proporción para entrenamiento
        val_split: Proporción para validación  
        test_split: Proporción para prueba
        min_images_per_person: Mínimo número de imágenes por persona para incluir
        seed: Semilla para reproducibilidad
        
    Returns:
        dict: Estadísticas de la división
    """
    if abs(train_split + val_split + test_split - 1.0) > 1e-6:
        raise ValueError("Los splits deben sumar 1.0")
    
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Crear directorios de salida
    train_dir = output_path / "train"
    val_dir = output_path / "validation"
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Filtrar personas con suficientes imágenes
    valid_persons = []
    for person_dir in source_path.iterdir():
        if person_dir.is_dir():
            images = list(person_dir.glob("*.jpg"))
            if len(images) >= min_images_per_person:
                valid_persons.append(person_dir)
    
    print(f"Personas válidas (>={min_images_per_person} imágenes): {len(valid_persons)}")
    
    # Mezclar y dividir personas
    random.shuffle(valid_persons)
    
    n_total = len(valid_persons)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_persons = valid_persons[:n_train]
    val_persons = valid_persons[n_train:n_train + n_val]
    test_persons = valid_persons[n_train + n_val:]
    
    stats = {
        "total_valid_persons": n_total,
        "train_persons": len(train_persons),
        "val_persons": len(val_persons),
        "test_persons": len(test_persons),
        "train_images": 0,
        "val_images": 0,
        "test_images": 0
    }
    
    # Copiar archivos a los directorios correspondientes
    def copy_person_data(persons_list, target_dir, split_name):
        images_count = 0
        for person_dir in persons_list:
            target_person_dir = target_dir / person_dir.name
            target_person_dir.mkdir(exist_ok=True)
            
            images = list(person_dir.glob("*.jpg"))
            for image_path in images:
                target_image_path = target_person_dir / image_path.name
                shutil.copy2(image_path, target_image_path)
                images_count += 1
        
        print(f"{split_name}: {len(persons_list)} personas, {images_count} imágenes")
        return images_count
    
    stats["train_images"] = copy_person_data(train_persons, train_dir, "Train")
    stats["val_images"] = copy_person_data(val_persons, val_dir, "Validation")  
    stats["test_images"] = copy_person_data(test_persons, test_dir, "Test")
    
    # Guardar metadatos
    metadata = {
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "min_images_per_person": min_images_per_person,
        "seed": seed,
        **stats
    }
    
    import json
    with open(output_path / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return stats

def create_face_identification_dataset(
    data_dir: str,
    img_size: Tuple[int, int] = (160, 160),
    batch_size: int = 32,
    shuffle_buffer: int = 1000
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Crea datasets de TensorFlow para identificación facial desde los splits.
    
    Args:
        data_dir: Directorio con los splits (train/, validation/, test/)
        img_size: Tamaño de redimensionamiento de imágenes
        batch_size: Tamaño de batch
        shuffle_buffer: Tamaño del buffer de mezcla
        
    Returns:
        Tuple[train_ds, val_ds, test_ds, label_mapping]: Datasets y mapeo de labels
    """
    data_path = Path(data_dir)
    
    # Crear mapeo de person_id a label numérico
    train_persons = sorted([d.name for d in (data_path / "train").iterdir() if d.is_dir()])
    person_to_label = {person_id: idx for idx, person_id in enumerate(train_persons)}
    num_classes = len(train_persons)
    
    def parse_image(image_path, label):
        """Procesa una imagen individual"""
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    def create_dataset_from_split(split_name: str) -> tf.data.Dataset:
        """Crea un dataset desde un split específico"""
        split_path = data_path / split_name
        
        image_paths = []
        labels = []
        
        for person_dir in split_path.iterdir():
            if person_dir.is_dir():
                person_id = person_dir.name
                if person_id in person_to_label:  # Solo personas del conjunto de entrenamiento
                    label = person_to_label[person_id]
                    for img_path in person_dir.glob("*.jpg"):
                        image_paths.append(str(img_path))
                        labels.append(label)
        
        # Crear dataset con tipos explícitos
        dataset = tf.data.Dataset.from_tensor_slices({
            "image_paths": tf.constant(image_paths, dtype=tf.string),
            "labels": tf.constant(labels, dtype=tf.int32)
        })
        
        # Extraer los valores del diccionario
        dataset = dataset.map(lambda x: (x["image_paths"], x["labels"]))
        
        if split_name == "train":
            dataset = dataset.shuffle(shuffle_buffer)
        
        dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    train_ds = create_dataset_from_split("train")
    val_ds = create_dataset_from_split("validation")
    test_ds = create_dataset_from_split("test")
    
    label_info = {
        "num_classes": num_classes,
        "person_to_label": person_to_label,
        "label_to_person": {v: k for k, v in person_to_label.items()}
    }
    
    return train_ds, val_ds, test_ds, label_info

if __name__ == "__main__":
    # Análisis del dataset original
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source_dir = os.path.join(root, "data", "face_identification", "training_set")
    output_dir = os.path.join(root, "data", "face_identification", "splits")
    
    print("Analizando estructura del dataset...")
    stats = analyze_dataset_structure(source_dir)
    print(f"Total de personas: {stats['total_persons']}")
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Promedio de imágenes por persona: {stats['avg_images_per_person']:.1f}")
    print(f"Rango de imágenes por persona: {stats['min_images_per_person']}-{stats['max_images_per_person']}")
    
    print("\nDividiendo dataset...")
    split_stats = split_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        min_images_per_person=5  # Mínimo 5 imágenes para tener suficientes datos
    )
    
    print("\nEstadísticas de la división:")
    for key, value in split_stats.items():
        print(f"{key}: {value}")
    
    print(f"\nDatasets guardados en: {output_dir}")