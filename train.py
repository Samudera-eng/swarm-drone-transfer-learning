"""Transfer learning ResNet-18 untuk klasifikasi objek udara (swarm drone perception).

mode:
  scratch : tanpa pretrained, semua layer dilatih (baseline)
  fe      : feature extraction  -> backbone beku, hanya fc baru dilatih
  partial : fine-tuning parsial -> layer4 + fc, discriminative LR
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # HARUS sama dgn deployment


def build_model(mode, n_cls):
    weights = None if mode == "scratch" else models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=weights)
    if mode in ("fe", "partial"):
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, n_cls)  # head baru -> requires_grad=True
    if mode == "partial":
        for p in m.layer4.parameters():
            p.requires_grad = True
    return m


def param_groups(m, mode, lr):
    if mode == "partial":
        return [{"params": m.layer4.parameters(), "lr": lr * 0.1},
                {"params": m.fc.parameters(), "lr": lr}]
    return [{"params": [p for p in m.parameters() if p.requires_grad], "lr": lr}]


def freeze_bn(m):
    """BatchNorm pada layer beku tetap eval() supaya statistik ImageNet tidak berubah."""
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d) and not any(p.requires_grad for p in mod.parameters()):
            mod.eval()


@torch.no_grad()
def evaluate(m, loader, dev, crit):
    m.eval()
    loss, preds, gts = 0.0, [], []
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        out = m(x)
        loss += crit(out, y).item() * x.size(0)
        preds += out.argmax(1).cpu().tolist()
        gts += y.cpu().tolist()
    n = len(gts)
    acc = float(np.mean(np.array(preds) == np.array(gts)))
    return loss / n, acc, gts, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", choices=["scratch", "fe", "partial"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out) / args.mode
    out.mkdir(parents=True, exist_ok=True)

    tf_train = T.Compose([
        T.Resize((224, 224)), T.RandomHorizontalFlip(), T.RandomRotation(15),
        T.ColorJitter(0.3, 0.3, 0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_eval = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])

    root = Path(args.data)
    ds_tr = datasets.ImageFolder(root / "train", tf_train)
    ds_va = datasets.ImageFolder(root / "val", tf_eval)
    classes = ds_tr.classes
    (out / "classes.json").write_text(json.dumps(classes))
    dl_tr = DataLoader(ds_tr, args.bs, shuffle=True, num_workers=args.workers)
    dl_va = DataLoader(ds_va, args.bs, num_workers=args.workers)

    m = build_model(args.mode, len(classes)).to(dev)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"mode={args.mode} kelas={classes} trainable_params={n_train:,} device={dev}")

    opt = torch.optim.Adam(param_groups(m, args.mode, args.lr))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    hist, best, t0 = [], 0.0, time.time()
    for ep in range(1, args.epochs + 1):
        m.train(); freeze_bn(m)
        tl, n, te = 0.0, 0, time.time()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(m(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        vl, va, _, _ = evaluate(m, dl_va, dev, crit)
        hist.append(dict(epoch=ep, train_loss=tl / n, val_loss=vl, val_acc=va, time=time.time() - te))
        print(f"ep {ep:02d} | train {tl/n:.4f} | val {vl:.4f} acc {va:.4f}")
        if va > best:
            best = va
            torch.save(m.state_dict(), out / "best.pt")

    result = dict(mode=args.mode, trainable_params=n_train, best_val_acc=best,
                  total_time=time.time() - t0, epochs=hist, test_acc=None)

    if (root / "test").exists():
        m.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        dl_te = DataLoader(datasets.ImageFolder(root / "test", tf_eval), args.bs, num_workers=args.workers)
        _, ta, gts, preds = evaluate(m, dl_te, dev, crit)
        result["test_acc"] = ta
        rep = classification_report(gts, preds, target_names=classes, digits=4)
        cm = confusion_matrix(gts, preds)
        (out / "report.txt").write_text(rep + "\nConfusion matrix:\n" + str(cm))
        print(rep); print(cm)

    (out / "history.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
