"""Diagnosa selisih val vs test.

1. Kemiripan terdekat tiap crop val/test ke crop train (cosine, fitur ResNet-18 sebelum fc).
   Kalau satu split jauh lebih mirip ke train -> indikasi kebocoran (near-duplicate).
2. Simpan crop yang salah prediksi ke errors/<split>/<benar>_as_<prediksi>/ untuk dicek visual.

Jalankan SETELAH training selesai (biar CPU/GPU tidak rebutan):
  python analyze_gap.py --mode partial
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


@torch.no_grad()
def extract(model, ds, dev, bs, workers):
    fc = model.fc
    model.fc = nn.Identity()
    feats, preds = [], []
    for x, _ in DataLoader(ds, bs, shuffle=False, num_workers=workers):
        f = model(x.to(dev))
        feats.append(F.normalize(f, dim=1))
        preds.append(fc(f).argmax(1))
    model.fc = fc
    return torch.cat(feats), torch.cat(preds).cpu().numpy()


@torch.no_grad()
def nearest_sim(query, ref, chunk=1024):
    out = []
    for i in range(0, len(query), chunk):
        out.append((query[i:i + chunk] @ ref.T).max(dim=1).values)
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", default="partial")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-copy", type=int, default=30, help="maks crop salah disimpan per pasangan kelas")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(args.runs) / args.mode
    classes = json.loads((run / "classes.json").read_text())
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.to(dev).eval()

    root = Path(args.data)
    ds = {s: datasets.ImageFolder(root / s, TF) for s in ("train", "val", "test")}
    assert ds["train"].classes == classes, "urutan kelas data != classes.json"

    print("ekstrak fitur train (beberapa menit)...")
    f_tr, _ = extract(model, ds["train"], dev, args.bs, args.workers)

    for split in ("val", "test"):
        f, pred = extract(model, ds[split], dev, args.bs, args.workers)
        y = np.array([t for _, t in ds[split].samples])
        sim = nearest_sim(f, f_tr)
        acc = float((pred == y).mean())
        print(f"\n== {split}: n={len(y)} acc={acc:.4f}")
        print(f"   kemiripan ke train terdekat: median={np.median(sim):.3f} mean={sim.mean():.3f} "
              f"p10={np.percentile(sim, 10):.3f} p90={np.percentile(sim, 90):.3f}")
        print(f"   fraksi sim>0.95: {(sim > 0.95).mean():.3f} | sim>0.90: {(sim > 0.90).mean():.3f}")
        for i, c in enumerate(classes):
            m = y == i
            print(f"   {c:<11} n={m.sum():<5} acc={(pred[m] == y[m]).mean():.4f} median_sim={np.median(sim[m]):.3f}")

        out = Path("errors") / split
        shutil.rmtree(out, ignore_errors=True)
        cnt = {}
        for (path, t), p in zip(ds[split].samples, pred):
            if t == p:
                continue
            key = f"{classes[t]}_as_{classes[p]}"
            cnt[key] = cnt.get(key, 0) + 1
            if cnt[key] <= args.max_copy:
                (out / key).mkdir(parents=True, exist_ok=True)
                shutil.copy(path, out / key / Path(path).name)
        print("   salah prediksi:", cnt if cnt else "tidak ada", "-> errors/%s/" % split)


if __name__ == "__main__":
    main()
