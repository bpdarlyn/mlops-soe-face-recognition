# Face Recognition Integration Guide

Este documento explica cómo usar el nuevo sistema de reconocimiento facial integrado con análisis de edad y género.

## 📋 Resumen del Proyecto

### Parte 1: Detección de Rostros (COMPLETA)
- ✅ **Dataset**: QMUL-SurvFace (5,319 personas, 220,888 imágenes)
- ✅ **Arquitectura**: MobileNetV2 + capas de embeddings + clasificación multi-tarea
- ✅ **Pipeline de Entrenamiento**: 3 etapas (identidad → conjunto → fine-tuning)
- ✅ **Scripts**: División de datos, entrenamiento y evaluación

### Parte 2: Identificación Conocido/Desconocido
Esta parte se implementará en la aplicación FastAPI, no en el entrenamiento, como especificaste.

## 🚀 Uso Rápido

### 1. Análisis del Dataset
```bash
# Análisis rápido del dataset
docker compose run --rm trainer python training/datasets/quick_dataset_test.py
```

### 2. División de Datos (Primera vez)
```bash
# Crear splits de entrenamiento/validación/test
docker compose run --rm trainer python training/datasets/survface.py
```

### 3. Entrenamiento del Modelo
```bash
# Entrenamiento completo (3 etapas)
docker compose run --rm trainer python training/train_face_recognition_tf.py
```

## 📊 Estadísticas del Dataset

- **Total de Personas**: 5,319
- **Total de Imágenes**: 220,888
- **Promedio por Persona**: 41.5 imágenes
- **Rango**: 2-489 imágenes por persona

### División Propuesta (min 5 imágenes/persona)
- **Personas Válidas**: 2,940
- **Training (70%)**: ~2,058 personas
- **Validation (15%)**: ~441 personas  
- **Test (15%)**: ~441 personas

## 🏗️ Arquitectura del Modelo

### Modelo Unificado de Análisis Facial
```
Input Image (160x160x3)
        ↓
   MobileNetV2 Backbone
        ↓
   Shared Features
        ↓
    ┌─────────┬─────────┬─────────┐
    │         │         │         │
    │ Face    │  Age    │ Gender  │
    │Embedding│Branch   │ Branch  │
    │         │         │         │
    │Identity │ Age     │ Gender  │
    │(Softmax)│(Linear) │(Sigmoid)│
    └─────────┴─────────┴─────────┘
```

### Componentes Clave
1. **Face Embedding**: Vectores de 512 dimensiones para comparación
2. **Identity Classification**: Clasificación entre identidades conocidas
3. **Age Estimation**: Regresión de edad
4. **Gender Classification**: Clasificación binaria

## 📁 Estructura de Archivos

```
training/
├── datasets/
│   ├── survface.py              # Procesamiento dataset QMUL-SurvFace
│   ├── quick_dataset_test.py    # Test rápido del dataset
│   └── utkface.py              # Dataset UTKFace (edad/género)
├── face_recognition_model.py    # Arquitecturas de modelos
├── combined_face_analytics.py   # Pipeline unificado
└── train_face_recognition_tf.py # Script de entrenamiento principal
```

## 🎯 Pipeline de Entrenamiento

### Etapa 1: Solo Identidad (15 epochs)
- Entrena solo la clasificación de identidad
- Backbone congelado
- LR: 1e-3

### Etapa 2: Multi-tarea (20 epochs)  
- Entrenamiento conjunto: identidad + edad + género
- Backbone aún congelado
- LR: 5e-4
- Pesos: identidad(1.0), edad(0.3), género(0.3)

### Etapa 3: Fine-tuning (10 epochs)
- Descongela últimas 30 capas del backbone
- LR: 1e-5
- Refinamiento final

## 📈 Métricas y Evaluación

### Métricas de Identidad
- **Accuracy**: Precisión en clasificación
- **Top-5 Accuracy**: Precisión en top-5 predicciones
- **Precision/Recall/F1**: Por clase

### Métricas de Edad
- **MAE (Mean Absolute Error)**: Error promedio en años
- **MSE (Mean Squared Error)**: Error cuadrático medio

### Métricas de Género
- **Accuracy**: Precisión binaria
- **Precision/Recall**: Por clase (M/F)

## 🔧 Integración con Modelo Existente

El sistema integra automáticamente el modelo preentrenado de edad/género:
- Transfiere pesos del modelo UTKFace existente
- Mantiene compatibilidad con el pipeline actual
- Permite fine-tuning conjunto

## 📱 Para la Aplicación FastAPI (Parte 2)

Los modelos entrenados generan:

### 1. Modelo de Embeddings (`face_embedding_extractor`)
- Convierte rostros en vectores de 512D
- Usar para comparar rostros desconocidos

### 2. Modelo Completo (`face_analytics_full`)
- Predicción completa: identidad + edad + género
- Usar cuando el rostro ya está en la BD

### 3. Mapeo de Etiquetas (`label_mapping.json`)
- Convierte IDs numéricos a IDs de personas
- Esencial para la aplicación

## 🏃‍♂️ Siguientes Pasos

### Implementación en FastAPI:
1. **Carga de Modelos**: Cargar modelo de embeddings y modelo completo
2. **Detección de Rostros Nuevos**: 
   - Extraer embedding del rostro
   - Calcular similitud con embeddings existentes
   - Si similitud < umbral → "Desconocido"
   - Si similitud ≥ umbral → Usar ID existente
3. **Registro de Nuevas Identidades**:
   - Guardar embedding en BD
   - Asignar nuevo ID
   - Opcional: re-entrenar modelo periódicamente

### Optimizaciones Futuras:
- **ArcFace Loss**: Para mejor separación de embeddings
- **Triplet Loss**: Entrenamiento contrastivo
- **Model Compression**: Cuantización para deployment
- **Online Learning**: Actualización incremental

## 🎉 Conclusión

El sistema está listo para la **Parte 1: Detección de Rostros**. La arquitectura unificada permite reconocimiento de identidad junto con análisis de edad y género en un solo modelo eficiente.

La **Parte 2: Conocido/Desconocido** se implementará en tu aplicación FastAPI usando los embeddings generados por este modelo.