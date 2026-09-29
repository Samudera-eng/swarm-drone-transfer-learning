"""Bandingkan run scratch vs fe vs partial -> results.md + compare.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = {}
for mode in ("scratch", "fe", "partial"):
    f = Path("runs") / mode / "history.json"
    if f.exists():
        runs[mode] = json.loads(f.read_text())

rows = ["| Mode | Trainable params | Best val acc | Test acc | Total time (s) |", "|---|---|---|---|---|"]
for k, r in runs.items():
    ta = f"{r['test_acc']:.4f}" if r["test_acc"] is not None else "-"
    rows.append(f"| {k} | {r['trainable_params']:,} | {r['best_val_acc']:.4f} | {ta} | {r['total_time']:.0f} |")
Path("results.md").write_text("\n".join(rows) + "\n")
print("\n".join(rows))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, r in runs.items():
    e = [h["epoch"] for h in r["epochs"]]
    ax[0].plot(e, [h["val_acc"] for h in r["epochs"]], label=k)
    ax[1].plot(e, [h["val_loss"] for h in r["epochs"]], label=k)
ax[0].set_title("Val accuracy"); ax[1].set_title("Val loss")
for a in ax:
    a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("compare.png", dpi=150)
