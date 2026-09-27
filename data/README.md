# GLID Dataset — Download and Setup Instructions

## Source

- **Dataset**: Glacial Lake Image Dataset (GLID)
- **Paper**: Ma D, Li J, Jiang L. 2025. *Efficient glacial lake mapping by leveraging deep
  transfer learning and a new annotated glacial lake dataset.* Journal of Hydrology 657: 133072.
- **Zenodo DOI**: [10.5281/zenodo.14838695](https://doi.org/10.5281/zenodo.14838695)
- **License**: CC BY 4.0

---

## Files to Download

| File | URL | Contents |
|---|---|---|
| `GLID.rar` | https://zenodo.org/records/14838695/files/GLID.rar?download=1 | Training set — 16,000 samples |
| `val.zip` | https://zenodo.org/records/14838695/files/val.zip?download=1 | Validation set — 2,367 samples |

Optional (not needed for training):
- `GLID_annotation.zip` — GIS shapefiles
- `Optical_images_source.xlsx` — image provenance metadata
- `Transferability validation.zip` — geographic transfer test scenes

---

## Download Steps

```bash
# From the project root
mkdir -p data/raw

# Download training set (large — several GB)
curl -L "https://zenodo.org/records/14838695/files/GLID.rar?download=1" \
     -o data/raw/GLID.rar

# Download validation set
curl -L "https://zenodo.org/records/14838695/files/val.zip?download=1" \
     -o data/raw/val.zip
```

---

## Extraction

```bash
# Extract validation set (zip)
unzip data/raw/val.zip -d data/raw/

# Extract training set (rar) — install unrar if needed
# macOS: brew install rar
# Ubuntu: sudo apt install unrar
unrar x data/raw/GLID.rar data/raw/
```

---

## Expected Directory Structure After Extraction

```
data/raw/
├── GLID.rar          # Keep original archive
├── val.zip           # Keep original archive
├── GLID/             # Extracted training set
│   ├── images/       # 16,000 RGB PNG patches (512×512)
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   └── labels/       # 16,000 binary mask PNGs (512×512)
│       ├── 1.png     # lake=255, background=0
│       ├── 2.png
│       └── ...
└── val/              # Extracted validation set
    ├── images/       # 2,367 RGB PNG patches (512×512)
    │   ├── 1.png
    │   └── ...
    └── labels/       # 2,367 binary mask PNGs (512×512)
        ├── 1.png
        └── ...
```

> **Note**: The exact subdirectory names (`labels/` vs `masks/`) must be verified after
> extraction. Run `python scripts/inspect_dataset.py --config configs/default.yaml` to
> auto-detect and validate the structure.

---

## Image Specifications

| Property | Value |
|---|---|
| Spatial resolution | 512 × 512 pixels |
| Image channels | 3 (RGB) |
| Image format | PNG |
| Mask: glacial lake | pixel value **255** |
| Mask: background | pixel value **0** |
| Filename pairing | Same base filename in `images/` and `labels/` |

## Satellite Sensors

GLID contains patches from four sensors, mixed without per-sample labels in the dataset:
- WorldView-2 (2 m resolution)
- Gaofen-2 (4 m)
- Sentinel-2 (10 m)
- Landsat-8 (30 m)
