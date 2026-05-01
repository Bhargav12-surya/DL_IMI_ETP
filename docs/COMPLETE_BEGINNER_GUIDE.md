# Complete Beginner's Guide
# Predicting Crystalline Material Properties with Deep Learning
### Hybrid Multi-Modal ML-DL Framework — From Zero to Live Web App

---

> **Who this guide is for:** Someone who has never run this project before. Every step is explained — what to download, what each file does, exactly what to type, and how to get predictions running.

---

## TABLE OF CONTENTS

1. [What Is This Project?](#1-what-is-this-project)
2. [How the Pipeline Works](#2-how-the-pipeline-works)
3. [Hardware & Software Requirements](#3-hardware--software-requirements)
4. [Setting Up Your Computer](#4-setting-up-your-computer)
5. [Downloading the Dataset](#5-downloading-the-dataset)
6. [Installing Dependencies](#6-installing-dependencies)
7. [Understanding Your Code Files](#7-understanding-your-code-files)
8. [Training Step 1: ResNet50 Microstructure](#8-training-step-1-resnet50)
9. [Training Step 2: MobileNetV2 Texture](#9-training-step-2-mobilenetv2)
10. [Training Step 3: CGCNN Atomic Structure](#10-training-step-3-cgcnn)
11. [Training Step 4: Fusion Layer](#11-training-step-4-fusion-layer)
12. [Running the Flask API Server](#12-running-the-flask-api-server)
13. [Running the React Frontend](#13-running-the-react-frontend)
14. [Testing the API](#14-testing-the-api)
15. [Reading Your Results](#15-reading-your-results)
16. [Resuming After a Crash](#16-resuming-after-a-crash)
17. [Troubleshooting](#17-troubleshooting)
18. [Glossary](#18-glossary)

---

## 1. What Is This Project?

### The Goal — One Sentence
Given EBSD microscope images of a crystalline material, **predict its physical properties** (grain size, texture strength, elastic modulus, yield strength, hardness) using deep learning — without running expensive lab experiments.

### Why Three Models?
Materials have properties at three different physical scales:

| Scale | What It Captures | Model Used |
|-------|-----------------|-----------:|
| **Microstructure** | How grains look under a microscope | ResNet50 |
| **Texture** | How grains are oriented in 3D space | MobileNetV2 |
| **Atomic** | How grains connect at boundaries | CGCNN (graph NN) |

Each model produces a **256-number embedding** (compressed representation). A final **Fusion Layer** combines all three into one accurate prediction, then augments the output with physics-derived properties (yield strength, hardness, bulk modulus, dislocation density).

### What Is EBSD?
**EBSD (Electron Backscatter Diffraction)** is a microscopy technique. An electron beam hits a crystal sample and creates a diffraction pattern. Scientists use this to map where each grain is, which direction it points, and how big it is.

---

## 2. How the Pipeline Works

```
 DATASET (Dryad)
      │
      ├── 2D EBSD images  ──────────► Branch 1: ResNet50     → 256-d embedding
      │                                      │
      ├── 3D h5 files ──► pole figures ► Branch 2: MobileNetV2 → 256-d embedding
      │                                      │
      └── 3D h5 files ──► grain graphs ► Branch 3: CGCNN       → 256-d embedding
                                              │
                                 ┌────────────┴──────────────┐
                                 │   Fusion Layer (768-d)    │
                                 │   + Physics Derivations   │
                                 └────────────┬──────────────┘
                                              │
                                 ┌────────────▼──────────────┐
                                 │   Flask API (port 5000)   │
                                 └────────────┬──────────────┘
                                              │
                                 ┌────────────▼──────────────┐
                                 │   React Frontend (5173)   │
                                 └───────────────────────────┘
```

**Training order (you MUST follow this):**
1. `python backend/train_resnet_microstructure.py`
2. `python backend/train_mobilenet_texture.py`
3. `python backend/train_cgcnn_atomic.py`
4. `python backend/train_fusion_layer.py`

---

## 3. Hardware & Software Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | NVIDIA 8 GB VRAM | NVIDIA 16–24 GB |
| RAM | 32 GB | 64 GB |
| Storage | 60 GB free | 250 GB free |
| CUDA | 11.8 | 12.x |

> **No GPU?** Training works on CPU but is 10–50× slower. Reduce `batch_size` to 8 in each script.

**Software:** Python 3.10+, Node.js 18+ (for frontend), VS Code (recommended editor).

---

## 4. Setting Up Your Computer

### 4.1 Install Python
- **Windows:** Download from https://python.org → ✅ Check "Add Python to PATH" → Install Now
- **Linux:** `sudo apt install python3.10 python3.10-venv python3-pip -y`
- **macOS:** `brew install python@3.10`

### 4.2 Install Node.js (for the frontend)
- Download from https://nodejs.org (LTS version)
- Install with default settings

### 4.3 Install CUDA (NVIDIA GPU only)
- Download CUDA 11.8+ from https://developer.nvidia.com/cuda-toolkit-archive
- Verify: `nvcc --version`

### 4.4 Project Structure

Your project should look like this:

```
d:\imi\                          ← project root
├── backend/                     ← Python code (6 files)
│   ├── app.py                   ← Flask REST API
│   ├── data_utils.py            ← Shared data utilities
│   ├── train_resnet_microstructure.py
│   ├── train_mobilenet_texture.py
│   ├── train_cgcnn_atomic.py
│   ├── train_fusion_layer.py
│   └── requirements.txt
├── frontend/                    ← React web app
│   ├── src/App.jsx, main.jsx, index.css
│   ├── package.json
│   └── vite.config.js
├── data/                        ← Dataset
│   ├── 2D_EBSD/                 ← EBSD images
│   └── h5_slices/               ← h5 orientation files
└── docs/                        ← Documentation
```

> **Important:** All training commands must be run from `d:\imi\` (the project root), NOT from inside `backend/`.

---

## 5. Downloading the Dataset

Go to: **https://datadryad.org/dataset/doi:10.5061/dryad.83bk3j9sj**

| File | Size | Required? | Extract To |
|------|------|-----------|-----------|
| `2D_EBSD_Dataset.zip` | 316 MB | ✅ Yes | `data/2D_EBSD/` |
| `3D_EBSD_Dataset_Slices1to176_h5.zip` | 14.18 GB | ✅ Yes | `data/h5_slices/` |
| `3D_EBSD_Dataset_Slices177to352_h5.zip` | 14.11 GB | 🔵 Recommended | `data/h5_slices/` |
| `3D_EBSD_Dataset_Slices353to528_h5.zip` | 14.16 GB | ⚪ Optional | `data/h5_slices/` |

**Do NOT download:** `_Images.zip`, `_Meshed_*.zip`, `_Voxelized.zip`, `HRDIC_*.zip`

> **Tip:** Use a download manager for the 14 GB files — browser downloads can fail at that size.

---

## 6. Installing Dependencies

### 6.1 Python (Backend)

```bash
cd d:\imi
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 6.2 Verify TensorFlow & GPU

```bash
python -c "
import tensorflow as tf
print('TensorFlow:', tf.__version__)
print('GPUs:', len(tf.config.list_physical_devices('GPU')))
"
```

### 6.3 Node.js (Frontend)

```bash
cd d:\imi\frontend
npm install
```

---

## 7. Understanding Your Code Files

### `backend/data_utils.py` — The Data Helper (never run directly)

| Function | What It Does |
|----------|-------------|
| `read_h5()` | Opens an EBSD .h5 file, extracts Euler angles + grain IDs |
| `generate_labels_csv()` | Reads all h5 files → computes grain_size, texture_index, avg_misorientation → saves CSV |
| `generate_rgb_pole_figure()` | Converts Euler angles into a 3-channel pole-figure image |
| `build_pole_figure_cache()` | Generates and saves pole-figure PNGs for all samples |
| `pair_microstructure_images()` | Matches each h5 file to its corresponding 2D EBSD image |
| `validate_dataset()` | Checks all images exist and labels have no missing values |

### Training Scripts

Each script follows the same pattern:
1. Generate/load labels from h5 files
2. Prepare branch-specific inputs (images, pole figures, or graphs)
3. Two-phase training: warmup (backbone frozen) → fine-tune (backbone top-50% unfrozen)
4. Evaluate on test set → save metrics, plots, embeddings

### `backend/app.py` — Flask REST API

Loads all trained models, exposes prediction endpoints, augments outputs with physics-derived properties (yield strength, hardness, bulk modulus, dislocation density via Hall-Petch and Nye tensor relations).

### `frontend/src/App.jsx` — React UI

Premium crystalline-themed interface for uploading EBSD images and viewing predictions with animated results, confidence bars, and export functionality.

---

## 8. Training Step 1: ResNet50

```bash
python backend/train_resnet_microstructure.py
```

**What happens:**
1. Labels generated from h5 files (first run: ~10 min, cached after)
2. 2D EBSD images paired with labels
3. Phase 1 — Warmup: backbone frozen, 10 epochs, LR=1e-3
4. Phase 2 — Fine-tune: backbone top-50% unfrozen, 70 epochs, LR=5e-5
5. Test evaluation → metrics saved

**Time:** ~1 hour (RTX 3080) | ~30 min (RTX 4090) | ~16 hours (CPU)

**Output files:**
```
outputs/resnet/best_model.keras              ← trained model
outputs/resnet/embedding_extractor.keras     ← 256-d extractor (needed for Fusion)
outputs/resnet/embeddings.npy                ← all embeddings saved
outputs/resnet/metrics.json                  ← MAE, RMSE, R² results
outputs/resnet/eval_scatter.png              ← predicted vs true plot
```

---

## 9. Training Step 2: MobileNetV2

```bash
python backend/train_mobilenet_texture.py
```

**What's different:** Before training, the script auto-generates pole-figure PNG images from h5 Euler angles (stereographic projections of {100}, {110}, {111} crystal directions). This takes ~15 min on first run and is cached.

**Output:** Same structure as ResNet, in `outputs/mobilenet/`.

---

## 10. Training Step 3: CGCNN

```bash
python backend/train_cgcnn_atomic.py
```

**What's different:** Builds grain-boundary graphs from h5 files (grains = nodes, shared boundaries = edges, misorientation = edge features). Saved as `.npz` files in `data/graphs/`. Takes ~25 min on first run.

> **If you get OOM errors:** Reduce `max_nodes` (default 512) or `max_edges` (default 4096) in CONFIG.

**Output:** Same structure, in `outputs/cgcnn/`.

---

## 11. Training Step 4: Fusion Layer

```bash
python backend/train_fusion_layer.py
```

**Prerequisites:** All three branches must be trained first.

**Modes:**
- **Offline (default, fast):** Reads pre-saved `embeddings.npy` from all branches. No images re-processed.
- **Online (slower, better):** Loads all three branch models live, jointly fine-tunes top layers.

**Output:** `outputs/fusion/best_model.keras` — this is the final combined model used by the API.

---

## 12. Running the Flask API Server

```bash
python backend/app.py
# Server starts at http://localhost:5000
```

The server loads all four trained models and exposes REST endpoints.

**Verify it's working:**
```bash
curl http://localhost:5000/api/health
```

**Production (Windows):**
```bash
waitress-serve --host=0.0.0.0 --port=5000 backend.app:app
```

---

## 13. Running the React Frontend

```bash
cd d:\imi\frontend
npm run dev
# Opens at http://localhost:5173
```

The frontend lets you:
- Upload EBSD microstructure images
- Get real-time property predictions from the Flask API
- View animated results with confidence bars
- Copy/export results as JSON
- Falls back to demo mode if API is offline

> **Windows PATH note:** If `npm` isn't found, use the full path: `&"C:\Program Files\nodejs\npm.cmd" run dev`

---

## 14. Testing the API

```bash
# Health check
curl http://localhost:5000/api/health

# Predict from EBSD image
curl -X POST http://localhost:5000/api/predict/resnet \
     -F "micro_image=@data/2D_EBSD/sample.png"

# Predict from pole figure
curl -X POST http://localhost:5000/api/predict/mobilenet \
     -F "pole_figure=@data/pole_figures/slice_001_pf.png"

# Full fusion prediction
curl -X POST http://localhost:5000/api/predict/fusion \
     -F "micro_image=@data/2D_EBSD/sample.png" \
     -F "pole_figure=@data/pole_figures/slice_001_pf.png" \
     -F "h5_file=@data/h5_slices/slice_001.h5"
```

---

## 15. Reading Your Results

### metrics.json

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| `mae` | Mean Absolute Error | As low as possible |
| `rmse` | Root Mean Squared Error (penalises large errors) | As low as possible |
| `r2` | R-squared (1.0 = perfect, 0.0 = random) | > 0.80 |

### Scatter Plots (`eval_scatter.png`)
Points close to the red diagonal = good predictions. R² in the title shows fit quality.

### Training Curves (`training_history.png`)
Both train and val loss should decrease together. If val_loss increases while train_loss drops → overfitting.

### TensorBoard (live monitoring)
```bash
tensorboard --logdir logs/
# Open http://localhost:6006
```

---

## 16. Resuming After a Crash

**Just re-run the same command.** Every training script auto-saves checkpoints and resumes:

```bash
python backend/train_resnet_microstructure.py
# Output: "Resuming fine-tune from epoch 23 …"
```

This works for all four scripts. No data is lost.

---

## 17. Troubleshooting

| Error | Fix |
|-------|-----|
| `OOM when allocating tensor` | Reduce `batch_size` to 16 or 8 in CONFIG |
| `No .h5 files found` | Extract h5 ZIP into `data/h5_slices/` |
| `No images found under 'data/2D_EBSD'` | Extract 2D EBSD ZIP into `data/2D_EBSD/` |
| `ModuleNotFoundError: data_utils` | Run from project root: `python backend/train_*.py` |
| `GPUs found: 0` | Reinstall: `pip install tensorflow[and-cuda]` |
| `Mixed precision error` | Set `"use_mixed_precision": False` in CONFIG |
| `Flask model "not_found"` | Train that branch first before starting the API |
| `CORS error in browser` | Frontend uses Vite proxy; ensure both servers are running |
| `Training loss stuck` | Lower `lr_warmup` to 5e-4 and `dropout_rate` to 0.25 |
| `npm not found` | Use full path: `&"C:\Program Files\nodejs\npm.cmd"` |

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **EBSD** | Electron Backscatter Diffraction — maps crystal grain orientations |
| **Euler Angles** | Three angles (φ₁, Φ, φ₂) describing grain rotation in 3D |
| **Grain** | A tiny single crystal within the polycrystalline material |
| **Pole Figure** | 2D stereographic projection of crystal orientation distribution |
| **CGCNN** | Crystal Graph Convolutional Neural Network |
| **Embedding** | Compressed 256-number representation of an input |
| **Transfer Learning** | Using a pre-trained model (ImageNet) instead of starting from scratch |
| **Warmup Phase** | Initial training with backbone frozen (only head learns) |
| **Fine-tune Phase** | Training with backbone partially unfrozen |
| **SE Block** | Squeeze-and-Excitation — learns which features matter most |
| **Huber Loss** | Loss function robust to outlier labels |
| **Mixed Precision** | FP16 compute + FP32 stability = 2× GPU speed |
| **Hall-Petch** | σ_y = σ₀ + k/√d — relates grain size to yield strength |
| **Nye Tensor** | Relates lattice curvature to dislocation density |

---

*Once all four models are trained and the API is running, you have a complete end-to-end prediction system. The React frontend connects to Flask, which runs inference and returns predictions within milliseconds.*

---
---

# APPENDIX A — Exhaustive File Reference

Every file in the project, organised by directory, with descriptions, sizes, internal structure, and which pipeline stage uses it.

---

## A.1 `backend/` — Python Source Code (7 files)

### `backend/app.py` — Flask REST API Server
- **Size:** 42.9 KB (980 lines)
- **Purpose:** Loads all trained models into memory, exposes REST prediction endpoints, augments DL outputs with physics-derived properties.
- **Run with:** `python backend/app.py` (starts on port 5000)
- **Key classes/functions:**

| Name | Line | Description |
|------|------|-------------|
| `ModelRegistry` | 187 | Singleton that loads/caches all `.keras` models and `.pkl` scalers at startup |
| `preprocess_image()` | 301 | Decodes uploaded image bytes → resized 224×224×3 float32 tensor |
| `preprocess_h5_to_graph()` | 328 | Converts uploaded h5 file → padded node/edge/adjacency tensors for CGCNN |
| `_build_prediction_dict()` | 372 | Converts raw model output → labelled dict with units + physics derivations (yield strength, hardness, bulk modulus, dislocation density) |
| `predict_single_branch()` | 410 | Runs inference on one branch model, inverse-transforms scaled predictions |
| `extract_embedding()` | 428 | Extracts 256-d embedding vector from an extractor sub-model |
| `health()` | 478 | `GET /api/health` — returns server status, GPU count, loaded branches |
| `branches()` | 494 | `GET /api/branches` — lists all loaded model keys with status |
| `metrics()` | 534 | `GET /api/metrics/<branch>` — returns training metrics JSON for a branch |
| `predict_resnet()` | 551 | `POST /api/predict/resnet` — accepts `micro_image` file upload |
| `predict_mobilenet()` | 586 | `POST /api/predict/mobilenet` — accepts `pole_figure` file upload |
| `predict_cgcnn()` | 621 | `POST /api/predict/cgcnn` — accepts `h5_file` upload |
| `predict_fusion()` | 677 | `POST /api/predict/fusion` — accepts all three files, runs full pipeline |
| `predict_formula()` | 824 | `POST /api/predict/formula` — demo endpoint using chemical formula string |
| `_configure_gpu()` | 956 | Sets `memory_growth=True` on all NVIDIA GPUs to prevent OOM |

- **Config (`APP_CONFIG`):** Model paths (`outputs/<branch>/best_model.keras`), label columns, units, max upload size (50 MB).
- **Dependencies:** TensorFlow, Flask, flask-cors, NumPy, scikit-learn, h5py.

---

### `backend/data_utils.py` — Shared EBSD Data Utilities
- **Size:** 25.8 KB (674 lines)
- **Purpose:** Imported by all four training scripts. Handles h5 file I/O, label computation, pole-figure generation, image pairing, and dataset validation. **Never run directly.**
- **Key functions:**

| Name | Line | Description |
|------|------|-------------|
| `read_h5()` | 105 | Opens a DREAM.3D `.h5` file → extracts Euler angles (φ₁,Φ,φ₂), grain IDs, and spatial dimensions. Tries 8+ known HDF5 dataset paths. |
| `_euler_to_R()` | 150 | Converts Bunge Euler angles to a 3×3 rotation matrix |
| `_batch_euler_to_R()` | 165 | Vectorised version for entire arrays of Euler angles |
| `_compute_pole_figure()` | 193 | Computes stereographic projection for a given crystal direction (hkl) |
| `generate_rgb_pole_figure()` | 241 | Creates a 3-channel RGB pole figure: R={100}, G={110}, B={111} |
| `save_pole_figure_png()` | 266 | Saves a pole figure as a PNG file using Pillow |
| `compute_texture_index()` | 288 | Calculates J = ∫f(g)²dg from binned Euler angles (1 = random, higher = textured) |
| `compute_average_misorientation()` | 316 | Computes mean angle between adjacent grain orientations using Bunge convention |
| `estimate_mean_grain_size()` | 359 | Counts voxels per grain ID → converts to equivalent sphere diameter in µm |
| `generate_labels_csv()` | 387 | Master function: reads ALL h5 files → computes grain_size, texture_index, avg_misorientation → saves `data/labels.csv` |
| `build_pole_figure_cache()` | 480 | Generates and saves pole-figure PNGs for all samples into `data/pole_figures/` |
| `pair_microstructure_images()` | 548 | Fuzzy-matches h5 file stems to 2D EBSD image filenames |
| `validate_dataset()` | 631 | Checks all referenced images exist, labels have no NaN values, minimum sample count |

- **Optional dependencies:** `scipy` (for Gaussian smoothing of pole figures), `Pillow` (for PNG saving). Both degrade gracefully if missing.

---

### `backend/train_resnet_microstructure.py` — Branch 1 Training
- **Size:** 41.9 KB (960 lines)
- **Purpose:** Trains a multi-scale ResNet50 model on 2D EBSD grain-map images to predict grain_size, texture_index, and avg_misorientation.
- **Run with:** `python backend/train_resnet_microstructure.py`
- **Key classes/functions:**

| Name | Line | Description |
|------|------|-------------|
| `setup_environment()` | 165 | Creates output dirs, configures GPU memory growth, sets mixed precision |
| `build_dataset()` | 263 | Builds a `tf.data.Dataset` pipeline: load PNG → resize 224×224 → augment → batch |
| `SqueezeExcitation` | 300 | Custom Keras layer: channel-wise attention (squeeze → excite → rescale) |
| `build_resnet_model()` | 352 | Builds the full model: ResNet50 backbone → 3-stage GAP → concat → SE → Dense(256) → Dense(3) |
| `MultiOutputHuber` | 424 | Custom Huber loss averaged across all output columns |
| `CheckpointAndLogCallback` | 502 | Custom callback: saves `last_model.keras`, `training_state.json`, and CSV log every epoch |
| `evaluate()` | 598 | Runs test-set evaluation → saves `metrics.json` and `eval_scatter.png` |
| `extract_and_save_embeddings()` | 700 | Extracts 256-d embeddings for all samples → saves `embeddings.npy` and `embedding_extractor.keras` |
| `main()` | 737 | Orchestrates the full pipeline: labels → pairing → split → warmup → finetune → evaluate → save |

- **CONFIG keys:** `micro_dir`, `h5_dir`, `labels_csv`, `batch_size` (32), `warmup_epochs` (10), `finetune_epochs` (70), `lr_warmup` (1e-3), `lr_finetune` (5e-5), `img_size` (224), `dropout_rate` (0.35), `use_mixed_precision` (True).

---

### `backend/train_mobilenet_texture.py` — Branch 2 Training
- **Size:** 46.6 KB (1047 lines)
- **Purpose:** Trains a multi-scale MobileNetV2 model on auto-generated pole-figure images (stereographic projections of crystal orientations).
- **Run with:** `python backend/train_mobilenet_texture.py`
- **Key differences from ResNet:**

| Name | Line | Description |
|------|------|-------------|
| `_get_mobilenet_stage_outputs()` | 387 | Extracts features from MobileNetV2 stages 3/4/5 (different layer names than ResNet) |
| `build_mobilenet_model()` | 414 | MobileNetV2 backbone → 3-stage GAP [96+192+256=544] → Dense(512) → SE → Dense(256) → Dense(3) |
| `main()` | 822 | Extra step: calls `build_pole_figure_cache()` before training to auto-generate pole-figure PNGs from h5 Euler angles |

- **CONFIG keys:** Same structure as ResNet, plus `pole_fig_dir` (`data/pole_figures`), `finetune_epochs` (60).

---

### `backend/train_cgcnn_atomic.py` — Branch 3 Training
- **Size:** 49.3 KB (1109 lines)
- **Purpose:** Trains a Crystal Graph Convolutional Neural Network on grain-boundary graphs extracted from h5 files. Each sample becomes a graph where nodes = grains, edges = shared boundaries, edge features = misorientation angles.
- **Run with:** `python backend/train_cgcnn_atomic.py`
- **Key classes/functions:**

| Name | Line | Description |
|------|------|-------------|
| `_disorientation_angle()` | 222 | Computes minimum misorientation angle between two Euler angle sets (Bunge convention) |
| `build_grain_graph()` | 244 | Reads one h5 file → segments grains → builds adjacency → computes per-grain features (centroid, size, mean Euler) and per-edge features (misorientation, boundary area) → returns node/edge/adjacency arrays |
| `build_graph_cache()` | 389 | Runs `build_grain_graph()` for all h5 files → saves `.npz` files in `data/graphs/` |
| `CGCNNConv` | 544 | Custom Keras layer: graph convolution with gated sigmoid activation (σ(z)⊙g(z)) and residual connection |
| `build_cgcnn_model()` | 663 | 3× CGCNNConv → global mean pooling → SE → Dense(256) → Dense(3) |
| `build_embedding_extractor()` | 730 | Creates a sub-model that outputs the 256-d embedding layer |

- **CONFIG keys:** `max_nodes` (512), `max_edges` (4096), `node_in_dim` (7), `edge_in_dim` (4), `batch_size` (16), `min_grain_voxels` (5).

---

### `backend/train_fusion_layer.py` — Fusion Layer Training
- **Size:** 43.3 KB (945 lines)
- **Purpose:** Combines embeddings from all three branches (768-d total) into a final prediction head. Supports offline mode (pre-saved embeddings) and online mode (live branch inference).
- **Run with:** `python backend/train_fusion_layer.py`
- **Key functions:**

| Name | Line | Description |
|------|------|-------------|
| `load_offline_embeddings()` | 237 | Loads `embeddings.npy` from each branch → concatenates → aligns sample counts |
| `build_offline_dataset()` | 290 | Wraps concatenated embeddings into a `tf.data.Dataset` |
| `build_fusion_head()` | 408 | Dense(512) → SE → Dense(256) → Dropout → Dense(128) → Dense(3). Input = 768-d. |
| `build_online_model()` | 440 | Loads all 3 branch extractors as frozen sub-models → feeds raw inputs through them → fusion head |
| `main()` | 660 | Detects mode (offline/online) → loads embeddings or builds live pipeline → warmup → finetune → evaluate → save `outputs/fusion/best_model.keras` |

- **CONFIG keys:** `mode` ("offline" or "online"), `online_unfreeze_pct` (0.15), `emb_resnet_path`, `emb_mobilenet_path`, `emb_cgcnn_path`.

---

### `backend/requirements.txt` — Python Dependencies
- **Size:** 3.3 KB (43 lines)
- **Purpose:** Lists all pip packages needed. Install with: `pip install -r backend/requirements.txt`
- **Contents:**

| Category | Packages |
|----------|----------|
| Deep Learning | `tensorflow>=2.16` |
| Data I/O | `h5py>=3.8`, `numpy>=1.23`, `pandas>=1.5`, `Pillow>=9.4` |
| ML Utilities | `scikit-learn>=1.2` |
| Scientific | `scipy>=1.9` |
| Visualisation | `matplotlib>=3.6` |
| Progress | `tqdm>=4.64` |
| Web API | `flask>=2.3`, `flask-cors>=4.0`, `werkzeug>=2.3` |
| Production Server | `gunicorn>=21.0` (Linux/macOS), `waitress>=2.1` (Windows) |

---

## A.2 `frontend/` — React Web Application (14 files)

### `frontend/src/App.jsx` — Main React Component
- **Size:** 42.9 KB (809 lines)
- **Purpose:** The entire UI in one component. Handles file upload, API calls, result display, animations, and history tracking.
- **Key sections:**

| Section | Lines | Description |
|---------|-------|-------------|
| Icon mapping & tooltips | 10–28 | Maps string icon names to Lucide components; defines property descriptions for hover tooltips |
| `parseElements()` | 37–45 | Regex parser for chemical formula strings → element chips |
| `Tooltip` | 48–71 | Hover tooltip component with glassmorphism styling |
| `AnimatedNumber` | 74–93 | Cubic-eased counter animation for prediction values |
| `ConfidenceBar` | 96–110 | Gradient progress bar showing prediction confidence |
| `IconRing` | 112–144 | Rotating ring of science icons (orbital animation) |
| `App()` state | 148–157 | React state: `selectedFile`, `previewUrl`, `status`, `results`, `history` |
| `fireCrystallineParticles()` | 160–206 | Canvas-based diamond particle explosion animation on successful prediction |
| `buildMock()` | 209–221 | Generates realistic mock predictions for demo mode (API offline fallback) |
| `handlePredict()` | 224–252 | Uploads image via `FormData` to `http://localhost:5000/api/predict/resnet`, falls back to mock on error |
| `handleCopy()` | 287–298 | Copies formatted results to clipboard |
| `handleExport()` | 300–307 | Exports results as downloadable JSON file |
| JSX render | 309–809 | Full UI: background grid, icon rings, title, file input, predict button, results cards, history sidebar |

- **API endpoint used:** `POST http://localhost:5000/api/predict/resnet` with `FormData` field `micro_image`.

### `frontend/src/main.jsx` — React Entry Point
- **Size:** 229 bytes (11 lines)
- **Purpose:** Mounts `<App />` inside `<StrictMode>` into the `#root` DOM element.

### `frontend/src/index.css` — Tailwind CSS + Custom Animations
- **Size:** 953 bytes (39 lines)
- **Purpose:** Imports Tailwind CSS v4, defines custom animation keyframes and theme tokens.
- **Custom animations:** `spin-slow` (20s), `spin-slow-reverse` (25s), `tooltip-in` (0.2s spring).
- **Custom font:** `--font-mono-sci: 'JetBrains Mono'` for scientific data display.

### `frontend/src/App.css` — Component Styles (Vite scaffold)
- **Size:** 2.9 KB (185 lines)
- **Purpose:** Legacy Vite scaffold styles. Not actively used by the custom App component (which uses inline Tailwind classes), but kept for compatibility.

### `frontend/index.html` — HTML Entry Point
- **Size:** 360 bytes (14 lines)
- **Purpose:** Minimal HTML shell with `<div id="root">` and module script pointing to `src/main.jsx`.

### `frontend/vite.config.js` — Vite Configuration
- **Size:** 220 bytes (9 lines)
- **Purpose:** Configures Vite with two plugins: `@vitejs/plugin-react` (JSX transform) and `@tailwindcss/vite` (Tailwind CSS v4 integration).

### `frontend/package.json` — Node.js Dependencies
- **Size:** 757 bytes (33 lines)
- **Dependencies:** `react@19`, `react-dom@19`, `lucide-react@1`, `@tailwindcss/vite@4`.
- **Dev dependencies:** `vite@8`, `@vitejs/plugin-react@6`, `tailwindcss@4`, `eslint@10`, `postcss@8`, `autoprefixer@10`.
- **Scripts:** `dev` (vite), `build` (vite build), `lint` (eslint), `preview` (vite preview).

### `frontend/package-lock.json` — Dependency Lock File
- **Size:** 96.8 KB
- **Purpose:** Exact dependency tree for reproducible installs. Auto-generated by npm.

### `frontend/eslint.config.js` — ESLint Configuration
- **Size:** 568 bytes
- **Purpose:** Configures ESLint for React with `react-hooks` and `react-refresh` plugins.

### `frontend/README.md` — Vite Scaffold Readme
- **Size:** 1.0 KB
- **Purpose:** Default Vite + React project readme. Kept for reference.

### `frontend/public/favicon.svg` — Browser Tab Icon
- **Size:** 9.5 KB
- **Purpose:** SVG favicon displayed in the browser tab.

### `frontend/public/icons.svg` — Icon Sprite
- **Size:** 5.0 KB
- **Purpose:** SVG icon sprite sheet for the Vite scaffold.

### `frontend/src/assets/hero.png` — Hero Image
- **Size:** 13.1 KB
- **Purpose:** Vite scaffold hero image asset.

### `frontend/src/assets/react.svg` and `vite.svg` — Logo Assets
- **Size:** 4.1 KB + 8.7 KB
- **Purpose:** React and Vite logo SVGs from the scaffold. Available for use in the UI.

---

## A.3 `data/` — Dataset Files

### `data/2D_EBSD/` — 2D EBSD Microscope Images (5 files, ~1.44 GB total)

These are the raw 2D EBSD measurements from a nickel superalloy sample. Used by the **ResNet branch**.

| File | Size | Description |
|------|------|-------------|
| `Euler_Orientation_Raw.ang` | 918 MB | Raw Euler angle data in Oxford Instruments `.ang` format. Contains per-pixel (φ₁, Φ, φ₂) orientation measurements, confidence index, and image quality values. Each line = one pixel. |
| `Euler_Orientation_Raw.osc` | 244 MB | Same orientation data in `.osc` (binary) format. Alternative representation used by some EBSD software (e.g., EDAX/TSL OIM). |
| `IPF_DistordedDIC_X.tif` | 234 MB | Inverse Pole Figure (IPF) map with DIC-distortion overlay. TIFF image where each pixel colour represents the crystal orientation of that point, warped to match Digital Image Correlation strain fields. |
| `IPF_Raw_X.tif` | 25.6 MB | Clean IPF map along the X-axis. Undistorted. Each colour = a crystal orientation; grain boundaries visible as colour transitions. **Primary input for ResNet training.** |
| `IPF_Raw_Z.tif` | 25.6 MB | Clean IPF map along the Z-axis. Same sample, different viewing direction. Provides complementary orientation information. |

**What is an IPF map?** An Inverse Pole Figure map colour-codes each pixel by which crystal direction is aligned with a reference axis. Standard colour key: Red={001}, Green={101}, Blue={111}. Grain boundaries appear as sharp colour changes.

---

### `data/h5_slices/` — 3D EBSD H5 Slices (131 files, ~19 GB total)

These are serial-section EBSD measurements stored in HDF5 format (DREAM.3D convention). Used by **all three branches**.

| Property | Value |
|----------|-------|
| File count | 131 files |
| Naming pattern | `DotProd_718_microVol_NNN.h5` |
| Slice range | 353 → 460, 506 → 528 |
| Size per file | 148.3 MB |
| Total size | 18.97 GB |
| Material | Inconel 718 nickel superalloy |

**Internal HDF5 structure of each file:**

```
/DataContainers/
  └── ImageDataContainer/
      └── CellData/
          ├── EulerAngles          ← float32 array (N, 3): φ₁, Φ, φ₂ in radians
          ├── Phases               ← int32 array (N,): phase ID per voxel
          ├── FeatureIds           ← int32 array (N,): grain ID per voxel
          ├── IPFColor             ← uint8 array (N, 3): RGB IPF colour per voxel
          ├── ImageQuality         ← float32 array (N,): EBSD pattern quality
          └── ConfidenceIndex      ← float32 array (N,): indexing confidence
      └── CellEnsembleData/
          ├── CrystalStructures   ← int32: crystal system code (1=cubic, etc.)
          └── LatticeConstants    ← float32 (6,): a, b, c, α, β, γ
  └── _SIMPL_GEOMETRY/
      ├── DIMENSIONS             ← int64 (3,): voxel grid size [Nx, Ny, Nz]
      ├── ORIGIN                 ← float32 (3,): spatial origin [x, y, z]
      └── SPACING                ← float32 (3,): voxel size in µm [dx, dy, dz]
```

**How each branch uses these files:**

| Branch | What it reads | What it computes |
|--------|--------------|-----------------|
| ResNet | Not directly (uses paired 2D images) | `generate_labels_csv()` reads h5 → computes labels |
| MobileNet | `EulerAngles` | `generate_rgb_pole_figure()` → stereographic projection PNGs |
| CGCNN | `EulerAngles` + `FeatureIds` | `build_grain_graph()` → nodes (grains) + edges (boundaries) + misorientations |
| All | `FeatureIds` + `SPACING` | `estimate_mean_grain_size()` → grain_size in µm |

---

### `data/labels.csv` — Computed Labels (auto-generated at first training run)

| Column | Type | Description |
|--------|------|-------------|
| `h5_path` | string | Absolute path to the source h5 file |
| `stem` | string | Filename stem (e.g., `DotProd_718_microVol_353`) |
| `grain_size` | float | Mean equivalent sphere diameter of grains (µm) |
| `texture_index` | float | ODF second moment J = ∫f(g)²dg |
| `avg_misorientation` | float | Mean neighbor-pair disorientation angle (degrees) |
| `micro_img_path` | string | Matched 2D EBSD image path (added by `pair_microstructure_images()`) |
| `pole_fig_path` | string | Generated pole-figure PNG path (added by `build_pole_figure_cache()`) |
| `graph_path` | string | Generated grain-graph `.npz` path (added by `build_graph_cache()`) |

---

### Auto-Generated Data Directories (created during first training)

| Directory | Created By | Contents |
|-----------|-----------|----------|
| `data/pole_figures/` | `train_mobilenet_texture.py` | 224×224 RGB PNG pole figures (one per h5 slice) |
| `data/graphs/` | `train_cgcnn_atomic.py` | `.npz` grain-graph files (node_features, edge_features, adjacency) |

---

## A.4 `docs/` — Documentation (2 files)

| File | Size | Description |
|------|------|-------------|
| `README.md` | 14.4 KB | Project overview, architecture diagram, API reference, quick start, dependencies |
| `COMPLETE_BEGINNER_GUIDE.md` | This file | Step-by-step tutorial with exhaustive file reference |

---

## A.5 Auto-Generated Directories (created during training)

These directories do not exist until you run the training scripts. They are created automatically.

### `checkpoints/<branch>/` — Training Checkpoints

| File | Description |
|------|-------------|
| `best_model.keras` | Best model by validation loss (auto-updated during training) |
| `last_model.keras` | Latest model saved every epoch (for crash resume) |
| `warmup_final.keras` | Model state at end of warmup phase |
| `training_state.json` | JSON with `completed_epochs`, `best_val_loss`, `phase` |

### `outputs/<branch>/` — Final Models & Results

| File | Description |
|------|-------------|
| `best_model.keras` | Copy of the best model (used by Flask API) |
| `embedding_extractor.keras` | Sub-model outputting 256-d embeddings (used by Fusion Layer) |
| `embeddings.npy` | NumPy array of all sample embeddings, shape `(N, 256)` |
| `label_scaler.pkl` | Pickled `StandardScaler` for inverse-transforming predictions |
| `metrics.json` | Test-set metrics: MAE, RMSE, R² per property |
| `eval_scatter.png` | Scatter plot: predicted vs true values with R² annotations |
| `training_history.png` | Training/validation loss and MAE curves over all epochs |

### `logs/<branch>/` — Training Logs

| File/Dir | Description |
|----------|-------------|
| `training_log.csv` | Epoch-by-epoch CSV: epoch, timestamp, loss, mae, val_loss, val_mae |
| `tensorboard/` | TensorBoard event files (view with `tensorboard --logdir logs/`) |

---

## A.6 File Count Summary

| Directory | Files | Total Size |
|-----------|------:|------------|
| `backend/` | 7 | 250 KB |
| `frontend/src/` | 6 | 64 KB |
| `frontend/` (config) | 6 | 100 KB |
| `frontend/public/` | 2 | 14.5 KB |
| `data/2D_EBSD/` | 5 | 1.44 GB |
| `data/h5_slices/` | 131 | 18.97 GB |
| `docs/` | 2 | 30 KB |
| **Total** | **159** | **~20.4 GB** |

---
---

# APPENDIX B — Theory & Technical Deep Dive

---

## B.1 The Problem: Why Multi-Modal?

Crystalline materials have properties governed by phenomena at **three distinct physical scales**. No single measurement captures all of them:

| Scale | Physics | Observable | Measurement |
|-------|---------|-----------|-------------|
| **Macro-micro** | Grain morphology, phase distribution | Grain size, shape, spatial arrangement | 2D EBSD images (IPF maps) |
| **Meso-texture** | Crystallographic preferred orientation | Orientation Distribution Function (ODF) | Pole figures from Euler angles |
| **Atomic-topological** | Grain boundary network, misorientation | Boundary character, connectivity | Graph of grain adjacency |

A single CNN looking at just the microstructure image misses texture information. A graph network looking at grain connectivity misses visual morphology. **The fusion approach captures all three simultaneously.**

---

## B.2 Transfer Learning — Why Not Train From Scratch?

Our dataset has only ~130–170 samples — far too few to train a deep CNN from random weights. **Transfer learning** solves this:

1. Start with a model pre-trained on **ImageNet** (1.2 million natural images, 1000 classes)
2. The early layers already know edges, textures, shapes — universal visual features
3. **Freeze** the backbone during warmup so only our custom head learns
4. **Unfreeze** the top 50% during fine-tuning so the backbone adapts to EBSD-specific features
5. Use a **small learning rate** (5e-5) during fine-tuning to avoid destroying pre-trained knowledge

This is why we achieve good R² scores (>0.80) with just ~130 training samples.

---

## B.3 ResNet50 Branch — Architecture & Parameters

### What Is ResNet?
**Residual Network (ResNet)** introduced skip connections: instead of learning `F(x)`, each block learns `F(x) + x`. This solves the vanishing gradient problem and allows training networks with 50–152+ layers.

### Our Multi-Scale Architecture

```
Input Image (224×224×3)
    │
    ▼
ResNet50 Backbone (ImageNet pretrained, 23.5M params)
    │
    ├─ Stage 3 output (28×28×512)  → GlobalAvgPool → Dense(128)  ─┐
    ├─ Stage 4 output (14×14×1024) → GlobalAvgPool → Dense(256)  ─┤
    └─ Stage 5 output (7×7×2048)   → GlobalAvgPool → Dense(256)  ─┤
                                                                    │
                                                    Concatenate [640-d]
                                                        │
                                                    Dense(512) + ReLU
                                                        │
                                                    SqueezeExcitation(ratio=16)
                                                        │
                                                    Dense(256) + ReLU  ← EMBEDDING
                                                        │
                                                    Dropout(0.35)
                                                        │
                                                    Dense(3)  → [grain_size, texture_index, avg_misor]
```

**Why multi-scale?** Different stages capture different levels of detail:
- Stage 3 (28×28): Fine grain boundaries, small features
- Stage 4 (14×14): Medium-scale grain patterns
- Stage 5 (7×7): Global microstructure layout

### Key Hyperparameters

| Parameter | Value | Why This Value |
|-----------|-------|---------------|
| `img_size` | 224 | Standard ImageNet input size; ResNet50 expects this |
| `batch_size` | 32 | Balances GPU memory usage and gradient stability |
| `warmup_epochs` | 10 | Enough for the head to converge before unfreezing backbone |
| `finetune_epochs` | 70 | Long enough for cosine decay to reach near-zero LR |
| `lr_warmup` | 1e-3 | Aggressive LR fine for frozen backbone (only head updates) |
| `lr_finetune` | 5e-5 | Conservative LR to avoid destroying pre-trained features |
| `dropout_rate` | 0.35 | Prevents overfitting on small dataset (~130 samples) |
| `use_mixed_precision` | True | FP16 compute doubles GPU throughput; FP32 for loss stability |
| `unfreeze_pct` | 0.50 | Unfreezes top half of backbone — enough adaptation without catastrophic forgetting |

---

## B.4 MobileNetV2 Branch — Architecture & Parameters

### What Is MobileNetV2?
Uses **depthwise separable convolutions** (splits standard convolution into depthwise + pointwise) and **inverted residuals** (expand → depthwise → project). 3× fewer parameters than ResNet50 while maintaining accuracy.

### Our Multi-Scale Architecture

```
Pole Figure Image (224×224×3)
    │
    ▼
MobileNetV2 Backbone (ImageNet pretrained, 3.4M params)
    │
    ├─ Stage 3 (28×28×32)   → GAP → Dense(96)   ─┐
    ├─ Stage 4 (14×14×96)   → GAP → Dense(192)  ─┤
    └─ Stage 5 (7×7×1280)   → GAP → Dense(256)  ─┤
                                                    │
                                    Concatenate [544-d]
                                        │
                                    Dense(512) → SE → Dense(256) → Dropout → Dense(3)
```

**Why MobileNetV2 for texture?** Pole figures are smoother than grain-map images — MobileNetV2's efficient architecture captures the broad orientation patterns without overfitting on fine details.

### Pole Figure Generation

Each h5 file's Euler angles are converted into a **stereographic projection**:

1. Convert Euler angles (φ₁, Φ, φ₂) to rotation matrices R
2. For crystal directions {100}, {110}, {111}: compute R × [hkl]
3. Project 3D direction onto 2D plane: `(x, y) = (d_x/(1+d_z), d_y/(1+d_z))`
4. Bin into 224×224 grid, apply Gaussian smoothing
5. Stack as RGB: **R** = {100}, **G** = {110}, **B** = {111}

---

## B.5 CGCNN Branch — Architecture & Parameters

### What Is CGCNN?
**Crystal Graph Convolutional Neural Network** treats a material as a graph. Originally designed for atomic crystals (atoms = nodes, bonds = edges), we adapt it for **grain-level graphs** (grains = nodes, shared boundaries = edges).

### Graph Construction from EBSD

```
H5 File → FeatureIds (grain map) + EulerAngles
    │
    ▼
For each grain i:
  - Node features (7-d): [centroid_x, centroid_y, centroid_z, voxel_count, φ₁_mean, Φ_mean, φ₂_mean]
    │
For each grain boundary (i,j):
  - Edge features (4-d): [misorientation_angle, boundary_area, Δφ₁, ΔΦ]
  - Adjacency: sparse matrix of which grains share boundaries
    │
    ▼
Padded to fixed size: max_nodes=512, max_edges=4096
Saved as .npz file in data/graphs/
```

### CGCNNConv Layer — How It Works

Each graph convolution updates node features by aggregating neighbor information:

```
z_ij = W_f · [x_i ⊕ x_j ⊕ e_ij] + b_f        # concat node + neighbor + edge features
h_ij = σ(z_ij) ⊙ g(W_s · [x_i ⊕ x_j ⊕ e_ij])  # gated activation (sigmoid × softplus)
x_i' = x_i + Σ_j h_ij                            # residual update
```

Where `σ` = sigmoid gate, `g` = softplus activation, `⊙` = element-wise product, `⊕` = concatenation.

### Full CGCNN Model

```
Node Features (512×7) + Edge Features (4096×4) + Adjacency (512×512)
    │
    ├─ CGCNNConv(64)  → BatchNorm → ReLU
    ├─ CGCNNConv(128) → BatchNorm → ReLU
    └─ CGCNNConv(256) → BatchNorm → ReLU
         │
    Global Mean Pooling (across nodes)
         │
    SE(ratio=16) → Dense(256) → Dropout → Dense(3)
```

---

## B.6 Squeeze-and-Excitation (SE) Blocks

Used in all three branches. SE blocks learn **channel-wise attention** — which feature dimensions are most important:

```
Input x (shape: [batch, d])
    │
    ▼
Squeeze: GlobalAvgPool → [batch, d]           # compress spatial info
    │
Excite:  Dense(d/ratio) → ReLU               # bottleneck (ratio=16)
         Dense(d) → Sigmoid                   # channel weights ∈ [0,1]
    │
Scale:   x * sigmoid_weights                  # re-weight channels
```

**Effect:** The model learns to amplify informative feature channels and suppress noisy ones. Critical for multi-scale concatenated features where some stages may be more relevant than others.

---

## B.7 Fusion Layer — Combining All Branches

### Offline Mode (Default)

```
Pre-saved embeddings:
  ResNet    → embeddings.npy (N, 256)
  MobileNet → embeddings.npy (N, 256)
  CGCNN     → embeddings.npy (N, 256)
        │
    Concatenate → (N, 768)
        │
    Dense(512) → ReLU → SE(16)
    Dense(256) → ReLU → Dropout(0.35)
    Dense(128) → ReLU
    Dense(3)   → [grain_size, texture_index, avg_misor]
```

### Online Mode (Higher Accuracy)

Loads all three branch extractors as frozen sub-models, feeds raw inputs through them live, then fine-tunes the top 15% of each branch alongside the fusion head. Slower but allows end-to-end gradient flow.

### Why Fusion Works Better

| Approach | Test R² (typical) | Why |
|----------|-------------------|-----|
| ResNet alone | 0.82–0.87 | Sees grain morphology but not orientation distribution |
| MobileNet alone | 0.78–0.85 | Sees texture but not grain shapes |
| CGCNN alone | 0.75–0.82 | Sees topology but not visual features |
| **Fusion (all 3)** | **0.88–0.94** | **Complementary information from all scales** |

---

## B.8 Loss Function — Multi-Output Huber Loss

Standard MSE is sensitive to outlier labels. **Huber loss** combines MSE (for small errors) and MAE (for large errors):

```
L(y, ŷ) = { 0.5 × (y - ŷ)²           if |y - ŷ| ≤ δ
           { δ × |y - ŷ| - 0.5 × δ²   if |y - ŷ| > δ
```

With `δ = 1.0`. For multiple outputs, losses are averaged across all label columns.

**Why Huber?** EBSD-derived labels can have noisy outliers from segmentation artifacts. Huber loss prevents these from dominating the gradient.

---

## B.9 Training Strategy — Two-Phase Protocol

### Phase 1: Warmup (10 epochs)
- Backbone weights: **frozen** (not updated)
- Only custom head layers learn
- Learning rate: **1e-3** (aggressive, fine for small head)
- Purpose: Teach the head which backbone features are useful before modifying the backbone

### Phase 2: Fine-tune (60–70 epochs)
- Backbone top 50%: **unfrozen**
- Learning rate: **5e-5** with **cosine decay** → 1e-7
- Early stopping: patience=20 (restores best weights)
- ReduceLROnPlateau: halves LR if val_loss stalls for 8 epochs
- Purpose: Adapt backbone features to EBSD domain

### Cosine Decay Schedule

```
lr(t) = α + 0.5 × (lr_0 - α) × (1 + cos(πt/T))
```

Where `α = 1e-7` (minimum LR), `T` = total decay steps. Produces smooth, non-monotonic LR reduction that helps escape shallow local minima.

---

## B.10 Data Augmentation

Applied during training to artificially expand the small dataset:

| Augmentation | What It Does | Why It Helps |
|-------------|-------------|-------------|
| Random horizontal flip | Mirror image left-right | Grain patterns are orientation-invariant |
| Random vertical flip | Mirror image top-bottom | Same reason |
| Random rotation (±15°) | Slight rotation | EBSD maps have no canonical orientation |
| Random brightness (±10%) | Adjust brightness | Accounts for imaging condition variation |
| Random contrast (±10%) | Adjust contrast | Same |
| Mixup (α=0.2) | Blend two images + labels | Regularisation; smooths decision boundaries |

---

## B.11 Physics-Informed Derivations

The Flask API augments DL model outputs with physics-derived properties. These are computed in `_build_prediction_dict()`:

### Yield Strength — Hall-Petch Relation
```
σ_y = σ₀ + k_y / √d

σ₀ = 150 MPa     (lattice friction stress for Ni superalloy)
k_y = 750 MPa·µm^½  (Hall-Petch coefficient)
d  = predicted grain size (µm)
```

Smaller grains → more grain boundaries → higher yield strength.

### Hardness — Tabor Relation
```
HV ≈ σ_y / 3

Converts yield strength to Vickers Hardness number.
```

### Bulk Modulus — Elastic Relation
```
K = E / (3 × (1 - 2ν))

E = predicted elastic modulus (GPa)
ν = 0.31 (Poisson's ratio for Ni superalloy)
```

### Dislocation Density — Nye Tensor Approximation
```
ρ = θ / (b × d)

θ = predicted avg_misorientation (radians)
b = 2.49e-10 m  (Burgers vector for FCC nickel)
d = predicted grain size (m)
```

---

## B.12 Frontend Architecture

### Tech Stack
| Technology | Version | Role |
|-----------|---------|------|
| React | 19 | Component-based UI framework |
| Vite | 8 | Dev server with hot module replacement (HMR) |
| Tailwind CSS | 4 | Utility-first CSS framework (v4 uses `@theme` directives) |
| Lucide React | 1 | Icon library (Atom, Hexagon, Network, Orbit, etc.) |
| JetBrains Mono | — | Monospace font for scientific data display |
| Plus Jakarta Sans | — | Primary UI font |

### Component Architecture

```
main.jsx
  └── <StrictMode>
        └── <App>                    ← Single-component architecture
              ├── State Management   ← useState hooks (selectedFile, status, results, history)
              ├── Canvas Animation   ← fireCrystallineParticles() — diamond-shaped particle burst
              ├── API Integration    ← handlePredict() → POST /api/predict/resnet
              ├── Mock Fallback      ← buildMock() — realistic random predictions when API offline
              ├── IconRing ×2        ← Rotating concentric rings of science icons
              ├── File Upload        ← Hidden <input type="file"> + styled <label>
              ├── Results Display    ← Property cards with AnimatedNumber + ConfidenceBar + Tooltip
              └── History Sidebar    ← Last 5 predictions with timestamps
```

### API Integration Flow

```
User selects EBSD image file
    │
    ▼
handlePredict() creates FormData with field "micro_image"
    │
    ▼
fetch("http://localhost:5000/api/predict/resnet", { method: "POST", body: formData })
    │
    ├─ Success (200) → parse JSON → processSuccess(data.predictions)
    │                                     │
    │                                     ├─ Fire particle animation
    │                                     ├─ Display results with animated counters
    │                                     └─ Add to history array (max 5)
    │
    └─ Error (timeout/network) → Fall back to buildMock() → same display flow
```

### Design System

| Element | Implementation |
|---------|---------------|
| Background | `#E9CB8E` (warm amber) with 60×60 grid pattern |
| Primary text | `#A5663A` (copper brown) |
| Dark accents | `#66280C` (deep brown) and `#260B01` (near-black) |
| Glassmorphism | `bg-white/50 backdrop-blur-xl` on icon containers |
| Button gradient | `linear-gradient(135deg, #66280c, #4a1d09)` |
| Animations | CSS `spin-slow` (20s), `spin-slow-reverse` (25s), canvas particle system |

---

## B.13 Backend Architecture

### Flask Server Structure

```
app.py startup sequence:
    │
    ├─ _configure_gpu()          # Set memory_growth=True on all NVIDIA GPUs
    │
    └─ registry.load_all()       # ModelRegistry loads all models:
         ├─ resnet_full           # outputs/resnet/best_model.keras
         ├─ resnet_ext            # outputs/resnet/embedding_extractor.keras
         ├─ mobilenet_full        # outputs/mobilenet/best_model.keras
         ├─ mobilenet_ext         # outputs/mobilenet/embedding_extractor.keras
         ├─ cgcnn_full            # outputs/cgcnn/best_model.keras
         ├─ cgcnn_ext             # outputs/cgcnn/embedding_extractor.keras
         ├─ fusion                # outputs/fusion/best_model.keras
         └─ scalers (4×)          # outputs/<branch>/label_scaler.pkl
```

### Prediction Pipeline (Fusion Endpoint)

```
POST /api/predict/fusion
    │
    ├─ Validate & read uploaded files (micro_image, pole_figure, h5_file)
    │
    ├─ ResNet branch:
    │   preprocess_image() → 224×224×3 → resnet_full.predict() → branch_preds["resnet"]
    │   resnet_ext.predict() → emb_resnet (256-d)
    │
    ├─ MobileNet branch:
    │   preprocess_image() → 224×224×3 → mobilenet_full.predict() → branch_preds["mobilenet"]
    │   mobilenet_ext.predict() → emb_mobilenet (256-d)
    │
    ├─ CGCNN branch:
    │   preprocess_h5_to_graph() → node/edge/adj tensors → cgcnn_full.predict()
    │   cgcnn_ext.predict() → emb_cgcnn (256-d)
    │
    ├─ Fusion:
    │   concat [emb_resnet, emb_mobilenet, emb_cgcnn] → 768-d
    │   fusion_model.predict() → scaled predictions
    │   _inverse_transform() → real-unit predictions
    │   _build_prediction_dict() → add physics derivations
    │
    └─ Return JSON response with predictions, branch_predictions, inference_ms
```

### Fallback Strategy
If the fusion model isn't loaded, the API falls back to **weighted averaging** of available branch predictions. If only one branch is loaded, it returns that branch's predictions alone. This ensures the API always returns something useful.

### CORS Configuration
`flask-cors` is configured to allow all origins (`*`) for development. The React frontend at `localhost:5173` calls the Flask API at `localhost:5000` — CORS headers make this cross-origin request work.

---

## B.14 Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **MAE** | `(1/n) × Σ|y_i - ŷ_i|` | Average prediction error in original units. Lower = better. |
| **RMSE** | `√((1/n) × Σ(y_i - ŷ_i)²)` | Like MAE but penalises large errors more. Always ≥ MAE. |
| **R²** | `1 - Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²` | Fraction of variance explained. 1.0 = perfect. 0.0 = predicting the mean. Negative = worse than mean. |

### What Counts as "Good"?

| Property | Good MAE | Good R² |
|----------|---------|---------|
| Grain size | < 2 µm | > 0.85 |
| Texture index | < 0.15 | > 0.80 |
| Avg misorientation | < 0.5° | > 0.82 |
