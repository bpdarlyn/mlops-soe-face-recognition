# Face Analytics Home

Sistema completo de análisis facial con reconocimiento de personas, estimación de edad y género usando TensorFlow, MLflow y FastAPI.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de análisis facial que incluye:

- **Detección de rostros** con coordenadas de bounding box
- **Reconocimiento facial** usando embeddings y similarity matching
- **Estimación de edad** mediante regresión
- **Clasificación de género** (masculino/femenino)
- **Identificación conocido/desconocido** con base de datos
- **API REST** para integración con aplicaciones
- **Seguimiento de experimentos** con MLflow
- **Base de datos MySQL** para persistencia

## 🏗️ Arquitectura

El proyecto está compuesto por 4 servicios Docker:

- **db**: Base de datos MySQL para MLflow y almacenamiento de rostros
- **mlflow**: Servidor MLflow para gestión de experimentos y modelos
- **trainer**: Entorno Python para entrenamiento y testing
- **face-api**: API FastAPI para análisis facial en tiempo real

## 🚀 Configuración Inicial

### Prerrequisitos

- Docker y Docker Compose
- Al menos 8GB RAM disponible
- 10GB espacio en disco

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Base de datos MySQL
MYSQL_ROOT_PASSWORD=root_password_secure_123
MYSQL_DATABASE=mlflow
MYSQL_USER=mlflow
MYSQL_PASSWORD=mlflow_password_secure_123

# Kaggle API (opcional, para descargar datasets)
KAGGLE_USERNAME=tu_usuario_kaggle
KAGGLE_KEY=tu_api_key_kaggle
```

## 🔧 Instalación Paso a Paso

### 1. Clonar y Preparar el Proyecto

```bash
# Clonar repositorio
git clone <url-del-repo>
cd face-analytics-home

# Crear archivo de variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### 2. Construir las Imágenes Docker

```bash
# Construir todas las imágenes
docker compose build

# Ver las imágenes creadas
docker images | grep face-analytics
```

### 3. Inicializar los Servicios Base

```bash
# Levantar base de datos y MLflow
docker compose up -d db mlflow

# Verificar que MLflow esté funcionando
docker compose logs -f mlflow
```

**Espera hasta ver:** `"Listening at: http://0.0.0.0:5000"`

Luego accede a: http://localhost:5001 (MLflow UI)

### 4. Verificar la Instalación

```bash
# Ejecutar tests para verificar que todo funciona
docker compose run --rm trainer pytest -q

# Test más detallado con coverage
docker compose run --rm trainer pytest --cov
```

## 📊 Preparación de Datos

### Datasets Utilizados

1. **UTKFace**: Para entrenamiento de edad y género
2. **QMUL-SurvFace**: Para reconocimiento facial

### Descargar y Preparar UTKFace

```bash
# Descarga completa (~3GB, ~20K imágenes)
docker compose run --rm trainer python scripts/download_utkface.py

# O descarga limitada para pruebas rápidas
docker compose run --rm trainer python scripts/download_utkface.py --limit 500

# Preparar dataset
docker compose run --rm trainer python scripts/prepare_utk_face.py
```

### Preparar QMUL-SurvFace

```bash
# Crear directorio y colocar dataset manualmente
mkdir -p data/face_identification/training_set

# El dataset debe tener estructura:
# data/face_identification/training_set/
#   ├── PersonID_1/
#   │   ├── 001_Camera1_image1.jpg
#   │   └── 002_Camera2_image2.jpg
#   └── PersonID_2/
#       ├── 001_Camera1_image1.jpg
#       └── 002_Camera2_image2.jpg

# Procesar y dividir dataset
docker compose run --rm trainer python training/datasets/survface.py
```

## 🤖 Entrenamiento de Modelos

### 1. Modelo de Prueba (Smoke Test)

```bash
# Entrenar modelo simple para verificar pipeline
docker compose run --rm trainer python training/train_smoke_tf.py
```

### 2. Modelo de Edad y Género

```bash
# Entrenar con UTKFace dataset
docker compose run --rm trainer python training/train_age_gender_tf.py
```

### 3. Modelo de Reconocimiento Facial

```bash
# Entrenar con QMUL-SurvFace dataset
docker compose run --rm trainer python training/train_face_recognition_tf.py
```

### 4. Verificar Entrenamientos

Accede a MLflow UI: http://localhost:5001

- Ve experimentos: `AgeGender-UTKFace`, `FaceRecognition-SurvFace`
- Revisa métricas, parámetros y artefactos
- Compara diferentes runs

## 📦 Gestión de Modelos con Scripts

### Registrar Modelos

```bash
# Registrar modelo desde el último run exitoso
docker compose run --rm trainer python scripts/register_model.py

# Verificar modelos registrados
docker compose run --rm trainer python scripts/check_models.py
```

### Deploy Automático

```bash
# Encuentra el mejor modelo y lo despliega a producción
docker compose run --rm trainer python scripts/deploy_model.py
```

### Utilidades de Gestión

```bash
# Ver rendimiento de todas las versiones
docker compose run --rm trainer python scripts/model_utils.py performance

# Hacer rollback a una versión específica
docker compose run --rm trainer python scripts/model_utils.py rollback 3

# Limpiar versiones antiguas (mantener solo 5)
docker compose run --rm trainer python scripts/model_utils.py clean --keep 5

# Comparar dos versiones
docker compose run --rm trainer python scripts/model_utils.py compare 2 4
```

## 🔌 API de Análisis Facial

### Iniciar el Servicio API

```bash
# Levantar la API (requiere modelos entrenados)
docker compose up -d face-api

# Verificar estado
docker compose logs -f face-api

# Health check
curl http://localhost:8000/health
```

### Endpoints Disponibles

#### 1. Analizar Imagen
```bash
# Subir imagen para análisis completo
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8000/analyze
```

**Respuesta:**
```json
{
  "success": true,
  "faces": [
    {
      "bbox": {"x": 100, "y": 50, "width": 150, "height": 200},
      "confidence": 0.95,
      "age": 25.5,
      "gender": "female",
      "gender_confidence": 0.85,
      "identity": "unknown",
      "person_id": "unknown_1703123456_7890"
    }
  ],
  "message": "Successfully analyzed 1 face(s)"
}
```

#### 2. Registrar Rostro Conocido
```bash
curl -X POST \
  -F "file=@john.jpg" \
  -F "person_name=John Doe" \
  http://localhost:8000/register
```

#### 3. Estadísticas
```bash
curl http://localhost:8000/stats
```

#### 4. Buscar por Nombre
```bash
curl "http://localhost:8000/search?name=John"
```

### Pruebas de la API

```bash
# Script de pruebas integrado
python test_api.py path/to/test/image.jpg

# Solo health check
python test_api.py
```

## 🐍 Uso con Python

```python
import requests

# Analizar una imagen
with open('photo.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    result = response.json()
    
    for face in result['faces']:
        print(f"Edad: {face['age']}")
        print(f"Género: {face['gender']}")
        print(f"Conocido: {face['identity']}")
```

## 🔧 Desarrollo y Depuración

### Acceso Interactivo

```bash
# Shell en el contenedor trainer
docker compose run --rm trainer bash

# Shell en el contenedor API
docker compose exec face-api bash

# Logs en tiempo real
docker compose logs -f face-api
```

### Reiniciar Servicios

```bash
# Reiniciar solo la API
docker compose restart face-api

# Reiniciar todo
docker compose down
docker compose up -d
```

### Limpieza

```bash
# Detener todos los servicios
docker compose down

# Limpiar volúmenes (CUIDADO: borra datos)
docker compose down -v

# Limpiar imágenes no utilizadas
docker system prune
```

## 📁 Estructura del Proyecto

```
face-analytics-home/
├── api/                          # FastAPI application
│   ├── main.py                   # API endpoints
│   ├── face_detector.py          # Face detection logic
│   ├── face_analytics.py         # Age/gender/recognition
│   ├── database.py               # MySQL integration
│   └── README.md                 # API documentation
├── training/                     # Training modules
│   ├── train_smoke_tf.py         # Smoke test training
│   ├── train_age_gender_tf.py    # Age/gender training
│   ├── train_face_recognition_tf.py  # Face recognition training
│   ├── combined_face_analytics.py   # Multi-task model
│   └── datasets/                 # Dataset processors
│       ├── utkface.py
│       └── survface.py
├── scripts/                      # Model management
│   ├── download_utkface.py       # Download UTKFace dataset
│   ├── prepare_utk_face.py       # Prepare UTKFace
│   ├── register_model.py         # Register model to MLflow
│   ├── deploy_model.py           # Auto-deploy best model
│   ├── check_models.py           # Check model status
│   └── model_utils.py            # Model utilities
├── tests/                        # Unit tests
├── docker-compose.yml            # Services definition
├── Dockerfile.trainer            # Trainer container
├── Dockerfile.mlflow             # MLflow container
├── Dockerfile.api                # API container
├── requirements.trainer.txt      # Training dependencies
├── requirements.api.txt          # API dependencies
├── test_api.py                   # API integration tests
└── README.md                     # This file
```

## 🚨 Solución de Problemas

### MLflow no se conecta
```bash
# Verificar que la DB esté lista
docker compose logs db

# Reiniciar MLflow
docker compose restart mlflow
```

### Error de memoria en entrenamiento
```bash
# Reducir batch size en los scripts de entrenamiento
# O aumentar memoria Docker
```

### Base de datos llena
```bash
# Limpiar detecciones antiguas
docker compose exec db mysql -u mlflow -p mlflow -e "DELETE FROM face_detections WHERE detected_at < DATE_SUB(NOW(), INTERVAL 30 DAY);"
```

## 🔒 Consideraciones de Seguridad

- Cambiar contraseñas por defecto en producción
- No commitear archivos `.env`
- Usar HTTPS en producción
- Implementar autenticación en la API
- Configurar firewall para los puertos

## 📈 Monitoreo

### Métricas Importantes

- Precisión de detección facial
- Tiempo de respuesta de API
- Uso de memoria y CPU
- Crecimiento de base de datos

### Logs

```bash
# Logs de la API
docker compose logs face-api

# Logs de entrenamiento
docker compose logs trainer

# Logs de MLflow
docker compose logs mlflow
```

## 🤝 Contribución

1. Fork del proyecto
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👨‍💻 Soporte

Para soporte y preguntas:
- Crear un issue en GitHub
- Revisar la documentación en `api/README.md`
- Consultar logs con `docker compose logs [servicio]`