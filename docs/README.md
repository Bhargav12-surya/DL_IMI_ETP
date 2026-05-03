# IMI — Hybrid Multi-Modal Crystal Property Predictor

> **Predict crystalline material properties from EBSD microstructure images using a multi-branch deep learning fusion pipeline.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://www.tensorflow.org)
[![React 19](https://img.shields.io/badge/React-19-61DAFB)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

IMI is a research-grade framework that combines **three complementary neural network branches** to predict physical properties of polycrystalline materials from a single EBSD (Electron Backscatter Diffraction) dataset:

| Branch | Architecture | Input | What It Learns |
|--------|-------------|-------|----------------|
| **Microstructure** | ResNet50 | 2D EBSD grain-map images | Grain morphology, size distribution, phase boundaries |
| **Texture** | MobileNetV2 | Auto-generated pole-figure PNGs | Crystallographic orientation distribution |
| **Atomic Structure** | CGCNN | Grain-boundary graphs from h5 files | Inter-grain misorientation topology |
| **Fusion** | Dense NN | Concatenated 768-d embeddings | Combined multi-scale prediction |

Each branch produces a **256-dimensional embedding** vector. The Fusion Layer concatenates all three (768-d) and predicts material properties through a dense head with physics-informed augmentation.

### Properties Predicted

| Property | Unit | Source |
|----------|------|--------|
| Grain Size | µm | EBSD grain segmentation |
| Texture Index | dimensionless | ODF second moment J = ∫f(g)²dg |
| Avg Misorientation | degrees | Neighbour-pair Euler angle difference |
| Elastic Modulus | GPa | Fusion + physics derivation |
| Yield Strength | MPa | Hall-Petch relation from grain size |
| Hardness | HV | Empirical correlation with yield strength |
| Bulk Modulus | GPa | Derived from elastic modulus (K ≈ E / 3(1-2ν)) |
| Dislocation Density | m⁻² | Derived from misorientation via Nye tensor |

---

## Repository Structure

```
d:\imi\
│
├── backend/                          ← All Python code
│   ├── app.py                        ← Flask REST API server (port 5000)
│   ├── data_utils.py                 ← Shared EBSD data utilities
│   ├── train_resnet_microstructure.py  ← Branch 1 training
│   ├── train_mobilenet_texture.py      ← Branch 2 training
│   ├── train_cgcnn_atomic.py           ← Branch 3 training
│   ├── train_fusion_layer.py           ← Fusion Layer training
│   └── requirements.txt               ← Python dependencies
│
├── frontend/                         ← React + Vite web app
│   ├── src/
│   │   ├── App.jsx                   ← Main UI component
│   │   ├── App.css                   ← Component styles
│   │   ├── main.jsx                  ← React entry point
│   │   └── index.css                 ← Tailwind CSS v4 + animations
│   ├── index.html                    ← HTML entry point
│   ├── package.json                  ← Node.js dependencies
│   └── vite.config.js                ← Vite + Tailwind plugin config
│
├── data/                             ← Dataset (not committed)
│   ├── 2D_EBSD/                      ← 2D EBSD grain-map images
│   └── h5_slices/                    ← 3D EBSD h5 slices
│
├── docs/                             ← Documentation
│   ├── README.md                     ← This file
│   └── COMPLETE_BEGINNER_GUIDE.md    ← Step-by-step beginner tutorial
│
├── checkpoints/                      ← Training checkpoints (auto-created)
├── outputs/                          ← Trained models & metrics (auto-created)
└── logs/                             ← Training logs & TensorBoard (auto-created)
```

---

## Quick Start

### 1. Install Python Dependencies

```bash
cd d:\imi
pip install -r backend/requirements.txt
```

### 2. Verify GPU (Optional but Recommended)

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 3. Train the Pipeline

**All commands must be run from the project root (`d:\imi\`):**

```bash
# Step 1 — Microstructure branch (requires data/2D_EBSD + data/h5_slices)
python backend/train_resnet_microstructure.py

# Step 2 — Texture branch (auto-generates pole figures from h5 files)
python backend/train_mobilenet_texture.py

# Step 3 — Atomic structure branch (builds grain graphs from h5 files)
python backend/train_cgcnn_atomic.py

# Step 4 — Fusion layer (combines all three branch embeddings)
python backend/train_fusion_layer.py
```

> **⏱ Training takes ~1 hour per branch on an RTX 3080.** Each script supports checkpoint resume — if interrupted, just re-run the same command.

### 4. Start the API Server

```bash
python backend/app.py
# Server starts at http://localhost:5000
```

### 5. Start the Frontend

```bash
cd frontend
npm install    # first time only
npm run dev
# UI opens at http://localhost:5173
```

---

## Architecture

```
                    ┌──────────────────────┐
                    │  EBSD Microstructure  │
                    │  (2D grain-map PNG)   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ResNet50 Branch     │
                    │   Multi-scale GAP     │
                    │   SE Attention        │
                    │   → 256-d embedding   │──────┐
                    └──────────────────────┘      │
                                                   │
                    ┌──────────────────────┐       │
                    │  Pole Figure (auto)   │       │
                    │  {100},{110},{111}    │       │
                    └──────────┬───────────┘       │
                               │                   │    ┌──────────────────┐
                    ┌──────────▼───────────┐       ├───►│  Fusion Layer    │
                    │  MobileNetV2 Branch   │       │    │  768-d → 512    │
                    │  Multi-scale GAP      │       │    │  SE → Dense(256)│
                    │  SE Attention         │       │    │  → Dense(3)     │
                    │  → 256-d embedding    │──────┤    │                  │
                    └──────────────────────┘       │    │  + Physics:      │
                                                   │    │  Yield Strength  │
                    ┌──────────────────────┐       │    │  Hardness        │
                    │  Grain Graph (auto)   │       │    │  Bulk Modulus    │
                    │  nodes=grains         │       │    │  Disloc. Density │
                    │  edges=boundaries     │       │    └────────┬─────────┘
                    └──────────┬───────────┘       │             │
                               │                   │    ┌────────▼─────────┐
                    ┌──────────▼───────────┐       │    │  Flask REST API   │
                    │  CGCNN Branch         │       │    │  Port 5000        │
                    │  Graph convolutions   │       │    └────────┬─────────┘
                    │  SE Attention         │       │             │
                    │  → 256-d embedding    │──────┘    ┌────────▼─────────┐
                    └──────────────────────┘            │  React Frontend   │
                                                        │  Port 5173        │
                                                        └──────────────────┘
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server status, GPU info, loaded branches |
| `POST` | `/api/predict/resnet` | Predict from microstructure image (`micro_image`) |
| `POST` | `/api/predict/mobilenet` | Predict from pole figure (`pole_figure`) |
| `POST` | `/api/predict/cgcnn` | Predict from h5 file (`h5_file`) |
| `POST` | `/api/predict/fusion` | Full multi-modal prediction (all 3 files) |
| `POST` | `/api/predict/formula` | Formula-based prediction (demo mode) |
| `GET` | `/api/metrics/<branch>` | Get training metrics for a branch |
| `GET` | `/api/branches` | List all loaded model branches |

### Example: Predict from EBSD Image

```bash
curl -X POST http://localhost:5000/api/predict/resnet \
     -F "micro_image=@data/2D_EBSD/sample.png"
```

Response:
```json
{
  "branch": "resnet",
  "predictions": {
    "grain_size":         {"value": 24.51, "unit": "µm"},
    "texture_index":      {"value":  1.87, "unit": "dimensionless"},
    "avg_misorientation": {"value":  3.24, "unit": "degrees"},
    "elastic_modulus":    {"value": 198.3, "unit": "GPa"},
    "yield_strength":     {"value": 412.0, "unit": "MPa"},
    "hardness":           {"value": 126.0, "unit": "HV"},
    "bulk_modulus":       {"value": 161.2, "unit": "GPa"}
  },
  "inference_ms": 142.3
}
```

---

## Training Details

### Two-Phase Transfer Learning

Each branch follows the same training protocol:

1. **Warmup Phase** (10 epochs) — Backbone weights frozen, only custom head learns. LR = 1e-3.
2. **Fine-tune Phase** (60–70 epochs) — Top 50% of backbone unfrozen with cosine-decay LR (5e-5 → 1e-7). Early stopping (patience=20) restores best weights.

### Model Architecture Highlights

- **Multi-scale feature extraction**: GAP from 3 backbone stages → concatenated
- **Squeeze-and-Excitation (SE) attention**: Channel-wise recalibration
- **Huber loss**: Robust to outlier labels
- **Mixed precision**: FP16 compute with FP32 accumulation (2× GPU throughput)
- **Checkpoint resume**: `training_state.json` tracks epoch, phase, and best val_loss

### Gradient Accumulation

All training scripts support gradient accumulation for training with larger effective batch sizes on limited VRAM:

```python
# In each script's CONFIG dict:
"grad_accum_steps": 4,    # effective batch = batch_size × 4
```

| Script | batch_size | grad_accum_steps | Effective Batch |
|--------|-----------|-----------------|-----------------|
| ResNet | 32 | 4 | 128 |
| MobileNet | 32 | 4 | 128 |
| CGCNN | 16 | 4 | 64 |
| Fusion | 32 | 4 | 128 |

Set `"grad_accum_steps": 1` to disable. No code changes needed — the wrapper is transparent to `model.fit()`.

### CUDA Device Selection

The pipeline automatically detects and configures GPUs. You can control device selection via:

```python
# In CONFIG — select specific GPU(s)
"cuda_visible_devices": "0",      # use GPU 0 only
"cuda_visible_devices": "0,1",    # multi-GPU
"cuda_visible_devices": "",       # force CPU

# Or via environment variable
CUDA_VISIBLE_DEVICES=0 python backend/train_resnet_microstructure.py
```

At startup, each script prints a diagnostic banner showing:
- TensorFlow version, CUDA version, cuDNN version
- GPU name(s), compute capability
- Mixed precision status

### Enhanced Checkpoints

Beyond the standard `best_model.keras` and `last_model.keras`, the pipeline now saves:

- **Epoch-N snapshots**: `ckpt_epoch_0010.keras`, `ckpt_epoch_0020.keras`, etc.
- **VERSION.json**: Full reproducibility metadata (git hash, config, device info)
- **Automatic pruning**: Old snapshots are deleted when `max_kept_ckpts` is exceeded

```python
"ckpt_every_n_epochs": 10,    # snapshot frequency
"max_kept_ckpts": 5,          # keep at most 5 epoch snapshots
```

### Training Logs

Each training run creates a structured log file in `logs/<branch>/train_<run_id>.log` containing:
- All Python `log.info()` output
- System info (OS, Python, TF version, GPU)
- Run ID and timestamp

The CSV training log (`training_log.csv`) now includes a `run_id` column for tracking across restarts.

### Git Version Control

Each training run automatically captures:
- **Git commit hash** and branch name
- **Dirty flag** (whether there are uncommitted changes)
- **Commit message**

This info is saved in both `training_state.json` and `VERSION.json`.

Optional: auto-tag experiments in git:
```python
"git_tag_runs": True,    # creates tag: run/resnet/20260502T153000_a1b2c3
```

### Fusion Layer Modes

| Mode | Speed | Description |
|------|-------|-------------|
| **Offline** | Fast | Reads pre-saved `embeddings.npy` from each branch. No GPU needed for branch models. |
| **Online** | Slow | Loads all three branch models, runs inference live, and fine-tunes top layers jointly. |

---

## Dataset

**Source:** [Dryad — doi:10.5061/dryad.83bk3j9sj](https://datadryad.org/dataset/doi:10.5061/dryad.83bk3j9sj)

| File | Size | Required By |
|------|------|-------------|
| `2D_EBSD_Dataset.zip` | 316 MB | ResNet branch |
| `3D_EBSD_Dataset_Slices1to176_h5.zip` | 14.18 GB | All branches |
| `3D_EBSD_Dataset_Slices177to352_h5.zip` | 14.11 GB | Optional (more data) |
| `3D_EBSD_Dataset_Slices353to528_h5.zip` | 14.16 GB | Optional (more data) |

Extract to:
- Images → `data/2D_EBSD/`
- H5 files → `data/h5_slices/`

---

## Frontend

The React frontend provides a premium crystalline-themed UI for uploading EBSD microstructure images and viewing predictions. Built with:

- **React 19** + **Vite 8** (fast HMR development)
- **Tailwind CSS v4** (utility-first styling)
- **Lucide React** icons (science-themed icon ring animations)
- **JetBrains Mono** + **Plus Jakarta Sans** typography

Features:
- Drag-and-drop or click-to-upload EBSD images
- Real-time prediction results with animated counters
- Confidence bars for each predicted property
- Copy-to-clipboard and JSON export of results
- Prediction history (last 5 analyses)
- Graceful fallback to demo mode when API is offline

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 8 GB VRAM | NVIDIA 16–24 GB |
| RAM | 32 GB | 64 GB |
| Storage | 60 GB free | 250 GB free |
| CUDA | 11.8 | 12.x |

> **No GPU?** Training works on CPU (10–50× slower). Set `batch_size: 8` in each script's CONFIG.

---

## Dependencies

### Backend (Python)
```
tensorflow>=2.16    h5py>=3.8      numpy>=1.23
pandas>=1.5         Pillow>=9.4    scikit-learn>=1.2
scipy>=1.9          matplotlib>=3.6  tqdm>=4.64
flask>=2.3          flask-cors>=4.0  werkzeug>=2.3
```

### Frontend (Node.js)
```
react@19            react-dom@19     vite@8
@tailwindcss/vite@4  tailwindcss@4   lucide-react@1
@vitejs/plugin-react@6
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `OOM when allocating tensor` | Reduce `batch_size` to 16 or 8 in CONFIG |
| `No .h5 files found` | Extract h5 ZIP into `data/h5_slices/` |
| `No images found` | Extract 2D EBSD ZIP into `data/2D_EBSD/` |
| `ModuleNotFoundError: data_utils` | Run from project root: `python backend/train_*.py` |
| `GPUs found: 0` | Reinstall TF: `pip install tensorflow[and-cuda]` |
| `Mixed precision error` | Set `"use_mixed_precision": False` in CONFIG |
| `Connection refused (API)` | Start Flask: `python backend/app.py` |

---

## Physics-Informed Derivations

The API augments DL model outputs with physics-based derived properties:

| Property | Formula | Source |
|----------|---------|--------|
| Yield Strength | σ_y = σ₀ + k_y / √d (Hall-Petch) | Grain size prediction |
| Hardness | HV ≈ σ_y / 3 (Tabor relation) | Yield strength |
| Bulk Modulus | K = E / 3(1 - 2ν) | Elastic modulus + Poisson's ratio |
| Dislocation Density | ρ = θ / (b × d) (Nye tensor) | Misorientation + Burgers vector |

---

## License

This project is for research and educational purposes. Dataset sourced from Dryad under CC0 public domain dedication.
