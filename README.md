# 1) Construir imágenes
`docker compose build`

# 2) Levantar DB y MLflow (en segundo plano)
`docker compose up -d db mlflow`

# 3) Revisar MLflow listo
`docker compose logs -f mlflow`
# cuando veas "Listening at: http://0.0.0.0:5000", abre http://localhost:5000

# 4) Ejecutar tests (pytest) en el contenedor trainer
`docker compose run --rm trainer pytest -q`

# 5) Ejecutar el entrenamiento "smoke" y loguear a MLflow+MySQL
`docker compose run --rm trainer python training/train_smoke_tf.py`

# (opcional) crea carpeta de datos en el host
mkdir -p data/UTKFace

# descarga completa
docker compose run --rm trainer python scripts/download_utkface.py
# o para probar rápido:
docker compose run --rm trainer python scripts/download_utkface.py --limit 200
