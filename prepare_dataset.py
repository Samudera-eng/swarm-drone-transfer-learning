"""Konversi dataset deteksi format YOLO (bbox) -> dataset klasifikasi (folder per kelas).

Kelas keluaran = kelas di data.yaml + 'background' (crop acak yang tidak overlap bbox).
Struktur keluaran: dst/{train,val,test}/{bird,drone,background}/*.jpg
"""
import argparse
import random
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLIT_ALIAS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def load_names(src: Path, fallback):
    yml = src / "data.yaml"
    if yml.exists():
        import yaml
        names = yaml.safe_load(yml.read_text()).get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        if names:
            return list(names)
    if fallback:
        return fallback.split(",")
    raise SystemExit("data.yaml tidak ada. Pakai --names bird,drone (urutan = id kelas di label).")


def label_path(img: Path) -> Path:
    parts = list(img.with_suffix(".txt").parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder dataset YOLO (berisi train/valid/test)")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--names", default=None, help="fallback nama kelas, mis. bird,drone")
    ap.add_argument("--pad", type=float, default=0.15, help="padding relatif di sekitar bbox")
    ap.add_argument("--min-size", type=int, default=24, help="skip crop lebih kecil (px)")
    ap.add_argument("--bg-per-image", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    names = load_names(src, args.names)
    print("Kelas dari label:", names, "(cek urutan vs data.yaml!)")
    counts = {}

    for split_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name in SPLIT_ALIAS):
        split = SPLIT_ALIAS[split_dir.name]
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            lbl = label_path(img_path)
            if not lbl.exists():
                continue
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            boxes = []
            for line in lbl.read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                bw, bh = w * W, h * H
                x1, y1 = cx * W - bw / 2, cy * H - bh / 2
                boxes.append((c, x1, y1, x1 + bw, y1 + bh))

            for i, (c, x1, y1, x2, y2) in enumerate(boxes):
                if min(x2 - x1, y2 - y1) < args.min_size:
                    continue
                px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                crop = im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))
                out = dst / split / names[c]
                out.mkdir(parents=True, exist_ok=True)
                crop.save(out / f"{img_path.stem}_{i}.jpg", quality=95)
                counts[(split, names[c])] = counts.get((split, names[c]), 0) + 1

            # background: crop acak dengan ukuran mirip bbox, tanpa overlap
            if not boxes:
                continue
            for k in range(args.bg_per_image):
                _, bx1, by1, bx2, by2 = random.choice(boxes)
                bw, bh = int(bx2 - bx1), int(by2 - by1)
                if bw < args.min_size or bh < args.min_size or bw >= W or bh >= H:
                    continue
                for _ in range(10):
                    rx, ry = random.randint(0, W - bw), random.randint(0, H - bh)
                    cand = (rx, ry, rx + bw, ry + bh)
                    if all(iou(cand, b[1:]) < 0.02 for b in boxes):
                        out = dst / split / "background"
                        out.mkdir(parents=True, exist_ok=True)
                        im.crop(cand).save(out / f"{img_path.stem}_bg{k}.jpg", quality=95)
                        counts[(split, "background")] = counts.get((split, "background"), 0) + 1
                        break

    for k in sorted(counts):
        print(k, counts[k])


if __name__ == "__main__":
    main()
