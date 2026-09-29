# Swarm Drone — Transfer Learning untuk Persepsi Visual Antar-Drone

RET503 Computer Vision and Deep Learning · Pertemuan 3 · Politeknik Negeri Batam

## Tujuan
Klasifikasi crop objek udara menjadi `drone` / `bird` / `background` sebagai modul persepsi
swarm: mengenali drone tetangga dari kamera onboard dan menolak false positive (burung, langit, dsb).
Output classifier dipakai sebagai *gate* setelah detektor (YOLO, minggu 6), bukan pengganti kontrol.

## Pipeline
```
raw YOLO dataset -> prepare_dataset.py -> data/{train,val,test}/{kelas}/ -> train.py (3 mode) -> compare.py
```

## Dataset
1. Publik: **Bird vs Drone** (Kaggle, format YOLO, 640x640, sumber gambar Pexels).
   Info: https://hyper.ai/en/datasets/38191 — unduh dari Kaggle, ekstrak ke `raw/`.
2. **Wajib tambah data sendiri** dari kamera drone proyek (min. ±100 crop/kelas), taruh di
   `data/train|val|test/<kelas>/` setelah langkah prepare. Ini domain target sebenarnya.

## Jalankan
```bash
pip install -r requirements.txt
python prepare_dataset.py --src raw/bird-vs-drone --dst data      # cek urutan kelas di output
python train.py --mode scratch --epochs 15
python train.py --mode fe      --epochs 15
python train.py --mode partial --epochs 15
python compare.py                                                 # -> results.md, compare.png
```

## Strategi
| Mode | Yang dilatih | LR |
|---|---|---|
| scratch | semua layer, tanpa pretrained (baseline) | 1e-3 |
| fe | `fc` saja | 1e-3 |
| partial | `layer4` + `fc` (discriminative LR) | 1e-4 / 1e-3 |

Catatan teknis: BatchNorm layer beku dipaksa `eval()`, optimizer hanya menerima parameter
`requires_grad=True`, preprocessing 224x224 RGB + mean/std ImageNet (harus identik saat deployment).

## Hasil
Isi dari `results.md` dan `compare.png` setelah training.

## Batasan
- Dataset publik berasal dari frame video + augmentasi -> risiko *near-duplicate leakage* antara train/val.
  Angka val bisa terlalu optimis; jadikan **test set dari kamera sendiri** sebagai angka utama.
- Model dilatih pada gambar drone RGB siang hari; malam/IR = domain berbeda (risiko negative transfer).
