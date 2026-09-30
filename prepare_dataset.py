"""Konversi dataset Bird vs Drone (YOLO, label campuran polygon/bbox) -> dataset klasifikasi.

Catatan dataset: SEMUA label berisi id 0, jadi kelas diambil dari AWALAN NAMA FILE
(B* = bird, D* = drone). Baris label 5 kolom = bbox; baris >5 kolom = polygon (diubah ke bbox).
Kelas keluaran: bird, drone, background (crop acak yang tidak overlap objek).

Struktur keluaran: dst/{train,val,test}/{bird,drone,background}/*.jpg
"""
import argparse
import random
from pathlib import Path

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png"}
SPLIT_ALIAS = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_prefix_map(s):
    return {kv.split("=")[0].upper(): kv.split("=")[1] for kv in s.split(",")}


def label_path(img: Path) -> Path:
    parts = list(img.with_suffix(".txt").parts)
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def parse_boxes(lbl: Path, W, H):
    """Kembalikan list bbox piksel (x1,y1,x2,y2) dari baris bbox maupun polygon."""
    boxes = []
    for line in lbl.read_text().strip().splitlines():
        p = line.split()
        try:
            v = [float(x) for x in p[1:]]
        except ValueError:
            continue
        if len(v) == 4:                       # cx cy w h
            cx, cy, w, h = v
            x1, y1, x2, y2 = (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H
        elif len(v) >= 6:                     # polygon: x1 y1 x2 y2 ...
            v = v[: len(v) // 2 * 2]
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H
        else:
            continue
        boxes.append((max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
    return boxes


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder yang berisi train/valid/test (mis. .../archive/Dataset)")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--prefix-map", default="B=bird,D=drone", help="huruf awal nama file -> kelas")
    ap.add_argument("--pad", type=float, default=0.15, help="padding relatif di sekitar bbox")
    ap.add_argument("--min-size", type=int, default=24, help="skip crop lebih kecil (px)")
    ap.add_argument("--bg-per-image", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    pmap = parse_prefix_map(args.prefix_map)
    counts, skipped = {}, 0

    for split_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name in SPLIT_ALIAS):
        split = SPLIT_ALIAS[split_dir.name]
        for img_path in sorted(split_dir.rglob("*")):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            cls = pmap.get(img_path.stem[0].upper())
            lbl = label_path(img_path)
            if cls is None or not lbl.exists():
                skipped += 1
                continue
            im = Image.open(img_path).convert("RGB")
            W, H = im.size
            boxes = parse_boxes(lbl, W, H)

            for i, (x1, y1, x2, y2) in enumerate(boxes):
                if min(x2 - x1, y2 - y1) < args.min_size:
                    continue
                px, py = (x2 - x1) * args.pad, (y2 - y1) * args.pad
                crop = im.crop((max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)))
                out = dst / split / cls
                out.mkdir(parents=True, exist_ok=True)
                crop.save(out / f"{img_path.stem}_{i}.jpg", quality=95)
                counts[(split, cls)] = counts.get((split, cls), 0) + 1

            if not boxes:
                continue
            for k in range(args.bg_per_image):
                x1, y1, x2, y2 = random.choice(boxes)
                bw, bh = int(x2 - x1), int(y2 - y1)
                if bw < args.min_size or bh < args.min_size or bw >= W or bh >= H:
                    continue
                for _ in range(10):
                    rx, ry = random.randint(0, W - bw), random.randint(0, H - bh)
                    cand = (rx, ry, rx + bw, ry + bh)
                    if all(iou(cand, b) < 0.02 for b in boxes):
                        out = dst / split / "background"
                        out.mkdir(parents=True, exist_ok=True)
                        im.crop(cand).save(out / f"{img_path.stem}_bg{k}.jpg", quality=95)
                        counts[(split, "background")] = counts.get((split, "background"), 0) + 1
                        break

    for k in sorted(counts):
        print(k, counts[k])
    print("gambar dilewati (awalan tak dikenal / label tak ada):", skipped)


if __name__ == "__main__":
    main()
