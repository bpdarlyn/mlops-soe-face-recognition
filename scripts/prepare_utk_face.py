import os, re, csv, sys

# Espera imágenes nombradas como: age_gender_race_date&time.jpg (ej. 25_0_0_20170116174525125.jpg)
# gender: 0=female, 1=male (según convención UTKFace)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "UTKFace")
OUT_CSV = os.path.join(DATA_DIR, "labels.csv")
PAT = re.compile(r"^(\d+)_(\d+)_\d+_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

def main():
    if not os.path.isdir(DATA_DIR):
        print("No encuentro data/UTKFace. Coloca ahí tus imágenes.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fn in os.listdir(DATA_DIR):
        m = PAT.match(fn)
        if not m:
            continue
        age = int(m.group(1))
        gender = int(m.group(2))
        rows.append([fn, age, gender])

    if not rows:
        print("No se encontraron archivos válidos con patrón UTKFace.", file=sys.stderr)
        sys.exit(2)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename","age","gender"])
        w.writerows(rows)
    print(f"Escrito: {OUT_CSV} ({len(rows)} filas)")

if __name__ == "__main__":
    main()
