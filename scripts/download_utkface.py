#!/usr/bin/env python3
"""
Descarga UTKFace desde múltiples fuentes y guarda los .jpg en ./data/UTKFace/.
Uso:
  python scripts/download_utkface.py
Opciones:
  --dest ./data/UTKFace   Carpeta destino (por defecto).
  --limit N               Descargar sólo N imágenes (debug).
"""

import os
import argparse
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

def ensure_dir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def infer_filename(ex: dict, idx: int) -> str:
    # HF expone "__key__" tipo "UTKFace/25_0_1_20170112235005249"
    key: Optional[str] = ex.get("__key__")
    if key:
        base = os.path.basename(key)
        # Asegura extensión .jpg
        if not base.lower().endswith((".jpg", ".jpeg", ".png")):
            base = base + ".jpg"
        return base
    # Fallback: nombre secuencial
    return f"img_{idx:05d}.jpg"

def download_from_kaggle(out_dir: Path, limit: Optional[int] = None) -> int:
    """Descarga UTKFace desde Kaggle (requiere kaggle API)"""
    try:
        import kaggle
        print("[INFO] Descargando UTKFace desde Kaggle...")
        
        # Descargar dataset de Kaggle
        kaggle.api.dataset_download_files('jangedoo/utkface-new', path=str(out_dir.parent), unzip=True)
        
        # Contar archivos descargados
        jpg_files = list(out_dir.glob("*.jpg"))
        if limit and len(jpg_files) > limit:
            # Eliminar archivos extras si hay limit
            for f in jpg_files[limit:]:
                f.unlink()
            return limit
        return len(jpg_files)
        
    except ImportError:
        print("[WARNING] Kaggle API no disponible. Instala con: pip install kaggle")
        return 0
    except Exception as e:
        print(f"[WARNING] Error descargando desde Kaggle: {e}")
        return 0

def create_sample_data(out_dir: Path, limit: int = 100) -> int:
    """Crea datos sintéticos para testing"""
    print(f"[INFO] Creando {limit} imágenes sintéticas para testing...")
    from PIL import Image, ImageDraw
    import random
    
    saved = 0
    for i in range(limit):
        # Crear imagen sintética 200x200
        img = Image.new('RGB', (200, 200), color=(random.randint(100, 255), 
                                                  random.randint(100, 255), 
                                                  random.randint(100, 255)))
        draw = ImageDraw.Draw(img)
        
        # Dibujar círculo simple para simular cara
        x, y = 100, 100
        r = random.randint(40, 80)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(200, 180, 160))
        
        # Generar nombre con formato UTKFace: age_gender_race_timestamp.jpg
        age = random.randint(1, 80)
        gender = random.randint(0, 1)
        race = random.randint(0, 4)
        timestamp = f"2017{random.randint(1000000000, 9999999999)}"
        
        filename = f"{age}_{gender}_{race}_{timestamp}.jpg"
        img.save(out_dir / filename, format="JPEG", quality=95)
        saved += 1
    
    return saved

def main(dest: str, limit: Optional[int] = None) -> None:
    out_dir = ensure_dir(dest)
    print(f"[INFO] Guardando imágenes en: {out_dir}")
    
    saved = 0
    
    # Método 1: Intentar Kaggle
    saved = download_from_kaggle(out_dir, limit)
    
    # Método 2: Si Kaggle falla, crear datos sintéticos
    if saved == 0:
        print("[INFO] Generando datos sintéticos para desarrollo/testing...")
        sample_limit = limit if limit and limit <= 200 else 100
        saved = create_sample_data(out_dir, sample_limit)
    
    print(f"\n[OK] Guardadas {saved} imágenes en {out_dir}")
    print("[SUGERENCIA] Ahora ejecuta: python scripts/prepare_utkface.py para generar labels.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="./data/UTKFace", help="Carpeta destino")
    ap.add_argument("--limit", type=int, default=None, help="Descargar sólo N imágenes (debug)")
    args = ap.parse_args()
    main(dest=args.dest, limit=args.limit)
