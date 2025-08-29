# Scripts de Gestión de Modelos MLflow

Esta carpeta contiene scripts para gestionar el ciclo de vida de los modelos de reconocimiento facial entrenados con MLflow.

## Scripts Disponibles

### 1. `deploy_model.py` - Deploy Automatizado ⭐
Script principal que automatiza todo el proceso de deploy:
- Encuentra el mejor run basado en métricas
- Valida la calidad del modelo
- Registra el modelo en MLflow
- Lo pone en producción

```bash
# Deploy automático del mejor modelo
python scripts/deploy_model.py
```

### 2. `register_model.py` - Registro Manual
Registra manualmente un modelo en MLflow desde el último run exitoso:

```bash
python scripts/register_model.py
```

**Lo que hace:**
- Busca el último run con todos los artefactos
- Registra una nueva versión del modelo
- Archiva versiones anteriores en producción
- Asigna alias 'prod' al modelo

### 3. `check_models.py` - Verificación y Consulta
Verifica modelos registrados y muestra información detallada:

```bash
python scripts/check_models.py
```

**Lo que muestra:**
- Todos los modelos registrados
- Versiones disponibles de cada modelo
- Modelo actualmente en producción
- Verificación de artefactos
- Código para integrar con FastAPI

### 4. `model_utils.py` - Utilidades de Gestión
Herramientas avanzadas para gestión de modelos:

```bash
# Hacer rollback a una versión específica
python scripts/model_utils.py rollback 3

# Limpiar versiones antiguas (mantener solo las 5 más recientes)
python scripts/model_utils.py clean --keep 5

# Ver rendimiento de todas las versiones
python scripts/model_utils.py performance

# Comparar dos versiones
python scripts/model_utils.py compare 2 4
```

## Flujo de Trabajo Recomendado

### 1. Después del Entrenamiento
```bash
# Opción A: Deploy automático (recomendado)
docker compose run --rm trainer python scripts/deploy_model.py

# Opción B: Deploy manual
docker compose run --rm trainer python scripts/register_model.py
```

### 2. Verificación
```bash
# Verificar que el modelo esté correctamente registrado
docker compose run --rm trainer python scripts/check_models.py
```

### 3. En caso de problemas
```bash
# Ver rendimiento de versiones
docker compose run --rm trainer python scripts/model_utils.py performance

# Hacer rollback si es necesario
docker compose run --rm trainer python scripts/model_utils.py rollback <version>
```

## Variables de Entorno

Los scripts usan las siguientes variables de entorno:

- `MLFLOW_TRACKING_URI`: URI del servidor MLflow (default: `http://mlflow:5000`)

## Configuración

Los scripts están configurados para:
- **Experimento**: `FaceRecognition-SurvFace`
- **Nombre del modelo**: `face-analytics-model`
- **Alias de producción**: `prod`

## Integración con FastAPI

Una vez que el modelo esté registrado, puedes usarlo en FastAPI:

```python
import mlflow
import mlflow.tensorflow

# Configurar MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

# Cargar modelo en producción
model = mlflow.tensorflow.load_model("models:face-analytics-model/prod")

# Usar para predicciones
predictions = model(input_data)
```

## Estructura de Artefactos Esperada

Los scripts buscan esta estructura de artefactos en cada run:

```
run_artifacts/
├── models/
│   ├── full_model/           # Modelo completo multi-tarea
│   ├── embedding_model/      # Modelo de embeddings
│   ├── identity_model/       # Modelo solo de identidad
│   └── age_gender_model/     # Modelo de edad y género
├── onnx/
│   └── face_embeddings.onnx  # Modelo ONNX para inferencia
└── metadata/
    └── label_mapping.json    # Mapeo de etiquetas
```

## Solución de Problemas

### Error: "Experimento no encontrado"
```bash
# Verificar que el experimento existe
docker compose run --rm trainer python -c "import mlflow; print(mlflow.search_experiments())"
```

### Error: "No se encontraron runs"
```bash
# Ejecutar primero el entrenamiento
docker compose run --rm trainer python training/train_face_recognition_tf.py
```

### Error de conexión con MLflow
```bash
# Verificar que MLflow esté ejecutándose
docker compose up -d mlflow
docker compose logs mlflow
```

### Verificar estado de contenedores
```bash
docker compose ps
```

## Logs y Debug

Todos los scripts incluyen logging detallado. En caso de errores:

1. Verificar que MLflow esté ejecutándose
2. Comprobar que existe al menos un run completado
3. Verificar que los artefactos estén presentes
4. Revisar los logs de los contenedores

## Ejemplos de Uso Completo

### Escenario 1: Primer Deploy
```bash
# 1. Entrenar modelo
docker compose run --rm trainer python training/train_face_recognition_tf.py

# 2. Deploy automático
docker compose run --rm trainer python scripts/deploy_model.py

# 3. Verificar
docker compose run --rm trainer python scripts/check_models.py
```

### Escenario 2: Nuevo Modelo Entrenado
```bash
# 1. Entrenar nueva versión
docker compose run --rm trainer python training/train_face_recognition_tf.py

# 2. Comparar con versión actual
docker compose run --rm trainer python scripts/model_utils.py performance

# 3. Deploy si es mejor
docker compose run --rm trainer python scripts/deploy_model.py
```

### Escenario 3: Rollback
```bash
# 1. Ver versiones disponibles
docker compose run --rm trainer python scripts/check_models.py

# 2. Comparar versiones
docker compose run --rm trainer python scripts/model_utils.py compare 1 2

# 3. Hacer rollback
docker compose run --rm trainer python scripts/model_utils.py rollback 1
```