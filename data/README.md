# Directorio de Datos

Este directorio contiene los datasets utilizados para el entrenamiento de modelos de análisis facial.

## Estructura de Directorios

```
data/
├── UTKFace/                          # Dataset UTKFace (descargado automáticamente)
│   ├── 20_0_0_20170109142408075.jpg  # Formato: edad_género_raza_timestamp.jpg
│   ├── 25_1_3_20170116174525125.jpg
│   └── ...
│
├── face_identification/              # Dataset QMUL-SurvFace (manual)
│   ├── training_set/                 # Dataset original
│   │   ├── PersonID_1/
│   │   │   ├── 001_Camera1_image1.jpg
│   │   │   ├── 002_Camera2_image2.jpg
│   │   │   └── ...
│   │   ├── PersonID_2/
│   │   │   ├── 001_Camera1_image1.jpg
│   │   │   └── ...
│   │   └── ...
│   │
│   └── splits/                       # Dataset procesado (generado automáticamente)
│       ├── train/
│       ├── validation/
│       ├── test/
│       └── split_metadata.json
│
└── README.md                         # Este archivo
```

## Datasets

### 1. UTKFace Dataset

**Descripción:** Dataset para entrenamiento de edad y género
- **Imágenes:** ~20,000 rostros
- **Formato:** edad_género_raza_timestamp.jpg
- **Edad:** 0-116 años
- **Género:** 0=masculino, 1=femenino
- **Raza:** 0=blanco, 1=negro, 2=asiático, 3=indio, 4=otros

**Descarga automática:**
```bash
# Descarga completa
docker compose run --rm trainer python scripts/download_utkface.py

# Descarga limitada (para pruebas)
docker compose run --rm trainer python scripts/download_utkface.py --limit 500
```

**Procesamiento:**
```bash
docker compose run --rm trainer python scripts/prepare_utk_face.py
```

### 2. QMUL-SurvFace Dataset

**Descripción:** Dataset para reconocimiento facial
- **Personas:** 5,319 identidades
- **Imágenes:** 220,888 rostros
- **Formato:** PersonID_CameraID_ImageName.jpg

**Instalación manual:**
1. Descargar el dataset desde: http://qmul-survface.cs.qmul.ac.uk/
2. Extraer en `data/face_identification/training_set/`
3. Verificar estructura de directorios

**Procesamiento automático:**
```bash
# Analizar dataset y crear splits
docker compose run --rm trainer python training/datasets/survface.py
```

## Archivos Generados

### UTKFace
- `data/UTKFace/labels.csv`: Metadatos extraídos de nombres de archivo
- Splits automáticos durante entrenamiento

### QMUL-SurvFace
- `data/face_identification/splits/`: Directorios train/validation/test
- `data/face_identification/splits/split_metadata.json`: Estadísticas de división

## Uso en Entrenamiento

### Edad y Género (UTKFace)
```python
from training.datasets.utkface import make_datasets
train_ds, val_ds = make_datasets(ROOT, img_size=(160,160), batch=32, val_split=0.1)
```

### Reconocimiento Facial (QMUL-SurvFace)
```python
from training.datasets.survface import create_face_identification_dataset
train_ds, val_ds, test_ds, label_info = create_face_identification_dataset(
    data_dir="data/face_identification/splits",
    img_size=(160, 160),
    batch_size=32
)
```

## Estadísticas de Datasets

### UTKFace (después de descarga)
- Imágenes totales: ~20,000
- Rango de edad: 0-116 años
- Distribución de género: ~50/50
- Formato: 200x200 píxeles (aprox.)

### QMUL-SurvFace (después de procesamiento)
- Personas válidas: ~4,000+ (con ≥5 imágenes)
- División por defecto: 70% train, 15% validation, 15% test
- Imágenes por persona: 3-100+ (muy variable)

## Notas Importantes

### Espacio en Disco
- UTKFace: ~3GB
- QMUL-SurvFace: ~10GB+ (dependiendo del subset descargado)
- Modelos entrenados: ~2GB adicionales

### Privacidad y Ética
- Estos datasets contienen rostros reales de personas
- Úsalos únicamente para investigación y desarrollo
- No redistribuyas las imágenes
- Respeta las licencias de los datasets originales

### Rendimiento
- Para entrenamiento rápido, usar datasets limitados durante desarrollo
- Para producción, usar datasets completos
- Considerar usar SSDs para mejor I/O durante entrenamiento

## Solución de Problemas

### Dataset UTKFace no se descarga
```bash
# Verificar credenciales Kaggle
echo "KAGGLE_USERNAME: $KAGGLE_USERNAME"
echo "KAGGLE_KEY: [hidden]"

# Configurar credenciales manualmente
docker compose run --rm trainer bash
mkdir -p ~/.kaggle
echo '{"username":"tu_usuario","key":"tu_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### QMUL-SurvFace estructura incorrecta
```bash
# Verificar estructura
docker compose run --rm trainer python -c "
from training.datasets.survface import analyze_dataset_structure
stats = analyze_dataset_structure('data/face_identification/training_set')
print(stats)
"
```

### Espacio en disco insuficiente
```bash
# Verificar espacio
df -h

# Limpiar archivos temporales
docker system prune -a
```

Para más información, consulta la documentación de cada dataset processor en `training/datasets/`.