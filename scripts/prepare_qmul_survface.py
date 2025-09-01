import os, csv, random, glob
from pathlib import Path

ROOT = Path("data/face_identification")
OUT = Path("artifacts")
OUT.mkdir(parents=True, exist_ok=True)

def main(seed=7, val_ratio=0.1, test_ratio=0.1):
    rows = []  # image_path, identity
    exts = (".jpg", ".jpeg", ".jpe", ".png")
    for id_dir in sorted(ROOT.iterdir()):
        if not id_dir.is_dir(): continue
        for p in sorted(id_dir.rglob("*")):
            if p.suffix.lower() in exts:
                rows.append((str(p), id_dir.name))

    random.Random(seed).shuffle(rows)
    n = len(rows); nv = int(n*val_ratio); nt = int(n*test_ratio)
    val = rows[:nv]; test = rows[nv:nv+nt]; train = rows[nv+nt:]

    for name, part in [("train", train), ("val", val), ("test", test)]:
        with open(OUT/f"{name}_manifest.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["image_path", "identity"])
            w.writerows(part)
        print(name, len(part))
    print("OK →", OUT)

if __name__ == "__main__":
    main()
