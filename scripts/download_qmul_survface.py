#!/usr/bin/env python3
"""
Descarga un ZIP desde Google Drive (SurvFace train), descomprime y re-formatea a:

  data/face_identification/<id>/<id>_<cam_id>_<count>.jpg

Uso:
  python scripts/download_qmul_survface.py
  python scripts/download_qmul_survface.py --limit 100
  python scripts/download_qmul_survface.py --url https://drive.google.com/open?id=<FILE_ID>

Notas:
- --limit limita el número de <id> procesados (directorios dentro de training_set/).
- Si hay imágenes .jpeg/.jpe, se guardan como .jpg (calidad 95).
"""

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import gdown
from PIL import Image
from tqdm import tqdm


DEFAULT_URL = "https://drive.google.com/open?id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl"


def extract_file_id(url_or_id: str) -> str:
    """Acepta un id o una URL de Drive y devuelve el file_id."""
    # Caso ya es un ID (sin esquema)
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
        return url_or_id

    # URL típica: open?id=FILEID
    parsed = urlparse(url_or_id)
    q = parse_qs(parsed.query)
    if "id" in q and q["id"]:
        return q["id"][0]

    # URL alternativa: /file/d/FILEID/
    m = re.search(r"/file/d/([^/]+)/?", parsed.path)
    if m:
        return m.group(1)

    raise ValueError(f"No pude extraer file_id desde: {url_or_id}")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_as_jpg(src: Path, dst: Path) -> bool:
    """Copia/convierte a .jpg. Devuelve True si se generó un archivo."""
    try:
        ext = src.suffix.lower()
        if ext == ".jpg":
            shutil.copy2(src, dst)
            return True
        elif ext in (".jpeg", ".jpe", ".png", ".bmp"):
            # convertir a JPEG
            with Image.open(src) as im:
                if im.mode in ("RGBA", "P"):
                    im = im.convert("RGB")
                im.save(dst, format="JPEG", quality=95, optimize=True)
            return True
        else:
            return False
    except Exception as e:
        print(f"[WARN] No pude procesar {src}: {e}")
        return False


def main(url: str, dest_root: str, limit: int | None) -> None:
    dest_root = Path(dest_root).resolve()
    safe_mkdir(dest_root)

    file_id = extract_file_id(url)
    print(f"[INFO] File ID: {file_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "survface_train.zip"
        out_dir = tmpdir / "unzipped"

        # 1) Descargar
        print("[INFO] Descargando ZIP desde Google Drive...")
        gdown.download(id=file_id, output=str(zip_path), quiet=False)

        # 2) Descomprimir
        print("[INFO] Descomprimiendo...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        # 3) Localizar carpeta training_set/
        cand = list(out_dir.rglob("training_set"))
        if not cand:
            raise RuntimeError("No encuentro la carpeta 'training_set' en el ZIP.")
        training_set = cand[0]
        print(f"[INFO] training_set: {training_set}")

        # 4) Enumerar IDs
        id_dirs = sorted([p for p in training_set.iterdir() if p.is_dir()])
        if limit is not None:
            id_dirs = id_dirs[:limit]
            print(f"[INFO] Aplicando --limit: {limit} ids")
        print(f"[INFO] IDs a procesar: {len(id_dirs)}")

        # 5) Copiar/convertir a data/face_identification/<id>/*.jpg
        total_imgs = 0
        for id_dir in tqdm(id_dirs, desc="Copiando por ID"):
            person_id = id_dir.name  # e.g., "1", "00234"
            target_dir = dest_root / person_id
            safe_mkdir(target_dir)

            imgs = sorted(
                [p for p in id_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".jpe", ".png", ".bmp")]
            )
            for src in imgs:
                # nombre ya viene como <id>_<cam>_<count>.<ext>
                stem = src.stem  # sin extensión
                dst = target_dir / f"{stem}.jpg"
                if dst.exists():
                    continue
                ok = copy_as_jpg(src, dst)
                if ok:
                    total_imgs += 1

        print(f"[OK] IDs: {len(id_dirs)} | Imágenes copiados/convertidos: {total_imgs}")
        print(f"[DONE] Dataset en: {dest_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Descargar SurvFace train y formatear a data/face_identification/<id>/*.jpg")
    ap.add_argument("--url", default=DEFAULT_URL, help="URL o file_id de Google Drive")
    ap.add_argument("--dest", default="./data/face_identification", help="Directorio destino")
    ap.add_argument("--limit", type=int, default=None, help="Número de IDs a incluir (opcional)")
    args = ap.parse_args()
    main(args.url, args.dest, args.limit)
