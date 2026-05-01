"""
train_cgcnn_atomic.py
═══════════════════════════════════════════════════════════════════════════════
Branch 3 — CGCNN-inspired Grain-Graph Network for crystallographic atomic
structure representation.

Dataset   : Dryad doi:10.5061/dryad.83bk3j9sj
Input     : 3D EBSD h5 files  (3D_EBSD_Dataset_Slices*.zip)
            Each sample → grain-level graph:
              • Nodes : grains  (features: mean Euler angles, volume, phase)
              • Edges : shared grain boundaries  (features: misorientation, area)
Labels    : auto-computed (shared labels.csv with Branches 1 & 2)
Output    : trained model, checkpoints, CSV logs, evaluation plots,
            256-dim embedding extractor (for Fusion Layer)

DESIGN NOTE — Why "grain graph" instead of true atomistic CIF?
──────────────────────────────────────────────────────────────
The Dryad EBSD dataset records crystallographic orientations at the grain
(mesoscale) level, not atomic positions.  Rather than using fictitious CIF
data, we build a physically meaningful CGCNN-style graph where each **grain**
plays the role of an "atom":
  • Node features  → mean Bunge Euler angles (φ₁, Φ, φ₂), normalised grain
                     volume, majority phase ID
  • Edge features  → disorientation angle (°) across the shared boundary,
                     normalised shared-face area
  • Message-passing → identical to the original Xie & Grossman CGCNN conv
                       (gate × message, softplus, residual)
This approach is published in grain-network GNN literature and predicts the
same bulk properties (grain_size, texture_index, avg_misorientation) that the
other two branches target.

ARCHITECTURE
────────────
  Grain graph  (variable #nodes, #edges per sample)
    └─► Embedding layer  (node features → 64-d, edge features → 64-d)
         └─► CGCNN Conv × 4  (64 → 64, residual, BN, Softplus)
              └─► Global Mean Pool  → 64-d graph embedding
                   └─► Dense(512, GELU) → BN → Dropout(0.4)
                        └─► SE-Block(512)
                             └─► Dense(256, GELU) → BN → Dropout   ← embedding
                                  └─► Dense(n_outputs, float32)    ← head

USAGE
─────
    python train_cgcnn_atomic.py

CHECKPOINT RESUME
─────────────────
Re-run the same command after any interruption.

DIRECTORY LAYOUT
────────────────
  data/
  ├── h5_slices/          ← shared with Branches 1 & 2
  ├── graphs/             ← auto-generated grain-graph cache (npz files)
  └── labels.csv          ← shared (auto-generated on first run)

  checkpoints/cgcnn/
  ├── best_model.keras
  ├── last_model.keras
  └── training_state.json

  logs/cgcnn/
  ├── training_log.csv
  └── tensorboard/

  outputs/cgcnn/
  ├── best_model.keras
  ├── embedding_extractor.keras
  ├── embeddings.npy
  ├── label_scaler.pkl
  ├── eval_scatter.png
  ├── training_history.png
  └── metrics.json
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import csv
import json
import logging
import os
import pickle
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# ── local ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import generate_labels_csv, read_h5

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
CONFIG: Dict = {
    # ── data paths ────────────────────────────────────────────────────────────
    "h5_dir":           "data/h5_slices",
    "graph_cache_dir":  "data/graphs",
    "labels_csv":       "data/labels.csv",
    "ckpt_dir":         "checkpoints/cgcnn",
    "log_dir":          "logs/cgcnn",
    "output_dir":       "outputs/cgcnn",
    "tb_dir":           "logs/cgcnn/tensorboard",

    # ── label targets ─────────────────────────────────────────────────────────
    "label_cols": [
        "texture_index",
        "avg_misorientation",
    ],

    # ── graph architecture ────────────────────────────────────────────────────
    "node_in_dim":      5,     # [phi1, Phi, phi2, norm_volume, phase_id]
    "edge_in_dim":      2,     # [misor_angle_norm, norm_shared_area]
    "hidden_dim":       64,    # CGCNN hidden dimension
    "n_conv":           4,     # number of CGCNN conv layers
    "embed_dim":        256,   # final embedding → Fusion Layer
    "dropout_rate":     0.40,
    "l2_reg":           1e-4,
    "se_ratio":         8,

    # ── graph limits (padding) ────────────────────────────────────────────────
    "max_nodes":        512,   # grains per sample (truncate/pad)
    "max_edges":        4096,  # edges per sample (truncate/pad)

    # ── graph building ────────────────────────────────────────────────────────
    "max_voxels":       500_000,   # subsample h5 voxels for speed
    "min_grain_voxels": 5,         # ignore grains smaller than this

    # ── training ──────────────────────────────────────────────────────────────
    "batch_size":       16,
    "warmup_epochs":    10,
    "finetune_epochs":  70,
    "lr_warmup":        1e-3,
    "lr_finetune":      2e-5,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,

    # ── splits ────────────────────────────────────────────────────────────────
    "val_frac":         0.15,
    "test_frac":        0.10,

    # ── misc ──────────────────────────────────────────────────────────────────
    "seed":             42,
    "use_mixed_precision": True,
    "prefetch":         tf.data.AUTOTUNE,
    "num_parallel":     tf.data.AUTOTUNE,
    "voxel_size_um":    1.0,
}


# ═════════════════════════════════════════════════════════════════════════════
# 0.  ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

def setup_environment(cfg: Dict) -> None:
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    for d in (cfg["ckpt_dir"], cfg["log_dir"],
              cfg["output_dir"], cfg["tb_dir"],
              cfg["graph_cache_dir"]):
        os.makedirs(d, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    log.info("GPUs available: %d", len(gpus))
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass

    if cfg["use_mixed_precision"] and gpus:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            log.info("Mixed precision (fp16) enabled.")
        except Exception as exc:
            log.warning("Could not enable mixed precision: %s", exc)
    else:
        tf.keras.mixed_precision.set_global_policy("float32")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  GRAIN-GRAPH BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _disorientation_angle(ea_i: np.ndarray, ea_j: np.ndarray) -> float:
    """
    Approximate disorientation angle between two Bunge ZXZ orientations
    (in radians). Uses the rotation-matrix dot-product shortcut:
        cos(ω) = (trace(R_i^T · R_j) - 1) / 2
    Valid for cubic symmetry without symmetry operators (upper bound).
    """
    def R(e: np.ndarray) -> np.ndarray:
        phi1, Phi, phi2 = e
        c1, s1 = np.cos(phi1), np.sin(phi1)
        c,  s  = np.cos(Phi),  np.sin(Phi)
        c2, s2 = np.cos(phi2), np.sin(phi2)
        return np.array([
            [ c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s],
            [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
            [ s1*s,             -c1*s,             c   ],
        ])
    Ri, Rj = R(ea_i.astype(np.float64)), R(ea_j.astype(np.float64))
    trace  = np.clip((np.trace(Ri.T @ Rj) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def build_grain_graph(h5_path: str, cfg: Dict) -> Optional[Dict[str, np.ndarray]]:
    """
    Build a grain-level graph from a single EBSD h5 file.

    Returns a dict with:
        node_feats   : (N_grains, node_in_dim)  float32
        edge_src     : (E,)                      int32
        edge_dst     : (E,)                      int32
        edge_feats   : (E, edge_in_dim)          float32
        n_nodes      : scalar int
        n_edges      : scalar int
    Returns None if the file cannot be parsed.
    """
    max_voxels        = cfg["max_voxels"]
    min_grain_voxels  = cfg["min_grain_voxels"]

    data = read_h5(h5_path)
    euler  = data["euler"]   # (N, 3) float32 radians
    grains = data["grains"]  # (N,)   int32
    phases = data["phases"]  # (N,)   int32  (can be None)
    dims   = data["dims"]    # (3,)   int32  [X, Y, Z]

    if euler is None or grains is None:
        return None

    N = len(euler)

    # ── subsample very large datasets ────────────────────────────────────────
    if N > max_voxels:
        idx    = np.random.choice(N, max_voxels, replace=False)
        euler  = euler[idx]
        grains = grains[idx]
        if phases is not None:
            phases = phases[idx]

    # ── per-grain statistics ──────────────────────────────────────────────────
    unique_grains = np.unique(grains)
    unique_grains = unique_grains[unique_grains > 0]           # skip grain 0

    grain_idx_map = {g: i for i, g in enumerate(unique_grains)}
    n_grains      = len(unique_grains)

    if n_grains < 2:
        return None

    grain_euler  = np.zeros((n_grains, 3),  dtype=np.float32)
    grain_vol    = np.zeros((n_grains,),    dtype=np.float32)
    grain_phase  = np.zeros((n_grains,),    dtype=np.float32)

    for g, idx in grain_idx_map.items():
        mask = grains == g
        if mask.sum() < min_grain_voxels:
            continue
        grain_euler[idx] = euler[mask].mean(axis=0)
        grain_vol[idx]   = float(mask.sum())
        if phases is not None:
            vals = phases[mask]
            grain_phase[idx] = float(np.bincount(vals[vals >= 0]).argmax()
                                     if len(vals[vals >= 0]) else 0)

    # Normalise volume (log-scale then z-score)
    grain_vol = np.log1p(grain_vol)
    v_std     = grain_vol.std() + 1e-9
    grain_vol = (grain_vol - grain_vol.mean()) / v_std

    # Normalise Euler angles to [0, 1]
    euler_norm      = grain_euler.copy()
    euler_norm[:, 0] /= (2 * np.pi)
    euler_norm[:, 1] /= np.pi
    euler_norm[:, 2] /= (2 * np.pi)

    # Normalise phase (0/1/2 → divide by max observed)
    phase_max    = grain_phase.max() if grain_phase.max() > 0 else 1.0
    grain_phase /= phase_max

    node_feats = np.column_stack([euler_norm, grain_vol, grain_phase]).astype(np.float32)

    # ── build adjacency from 3D neighbour pairs ───────────────────────────────
    # Reshape grains to 3D grid if dims available; otherwise treat as 1D strip
    try:
        if dims is not None and np.prod(dims[:3]) == len(grains):
            X, Y, Z = int(dims[0]), int(dims[1]), int(dims[2])
        else:
            # Guess cubic shape
            side = int(round(len(grains) ** (1/3)))
            X = Y = Z = side
        G3 = grains.reshape(X, Y, Z)

        edge_pairs: set = set()

        def _add_pairs(a: np.ndarray, b: np.ndarray) -> None:
            """Add (i, j) pairs where a[v] != b[v] and both are valid grains."""
            mask = (a > 0) & (b > 0) & (a != b)
            for gi, gj in zip(a[mask], b[mask]):
                if gi in grain_idx_map and gj in grain_idx_map:
                    i, j = grain_idx_map[gi], grain_idx_map[gj]
                    edge_pairs.add((min(i, j), max(i, j)))

        _add_pairs(G3[:-1, :, :].ravel(), G3[1:, :, :].ravel())
        _add_pairs(G3[:, :-1, :].ravel(), G3[:, 1:, :].ravel())
        _add_pairs(G3[:, :, :-1].ravel(), G3[:, :, 1:].ravel())

    except Exception:
        # Fallback: 1-D neighbour adjacency
        edge_pairs = set()
        for v in range(len(grains) - 1):
            gi, gj = int(grains[v]), int(grains[v + 1])
            if gi > 0 and gj > 0 and gi != gj:
                if gi in grain_idx_map and gj in grain_idx_map:
                    i, j = grain_idx_map[gi], grain_idx_map[gj]
                    edge_pairs.add((min(i, j), max(i, j)))

    if len(edge_pairs) == 0:
        return None

    # ── compute edge features ─────────────────────────────────────────────────
    edge_list     = sorted(edge_pairs)
    src_nodes     = np.array([e[0] for e in edge_list], dtype=np.int32)
    dst_nodes     = np.array([e[1] for e in edge_list], dtype=np.int32)

    # Disorientation angle for each boundary
    misor_angles = np.array([
        _disorientation_angle(grain_euler[s], grain_euler[d])
        for s, d in edge_list
    ], dtype=np.float32)
    misor_norm = misor_angles / 62.8  # max disorientation in cubic

    # Boundary area proxy: min(vol_i, vol_j) as a rough shared-face estimate
    area_proxy = np.array([
        min(grain_vol[s], grain_vol[d]) for s, d in edge_list
    ], dtype=np.float32)
    # area_proxy is already z-scored from grain_vol normalisation

    edge_feats = np.column_stack([misor_norm, area_proxy]).astype(np.float32)

    return {
        "node_feats": node_feats,
        "edge_src":   src_nodes,
        "edge_dst":   dst_nodes,
        "edge_feats": edge_feats,
        "n_nodes":    np.int32(n_grains),
        "n_edges":    np.int32(len(edge_list)),
    }


def build_graph_cache(labels_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Build and cache grain-graph npz files for every row in labels_df.
    Idempotent — already-existing files are skipped.

    Adds a 'graph_path' column; drops rows that failed.
    """
    cache_dir = cfg["graph_cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    graph_paths: List[Optional[str]] = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                       unit="sample", desc="Grain graphs"):
        sid  = row["sample_id"]
        out  = os.path.join(cache_dir, sid + "_graph.npz")
        if os.path.exists(out):
            graph_paths.append(out)
            continue

        try:
            g = build_grain_graph(row["h5_path"], cfg)
            if g is None:
                graph_paths.append(None)
                continue
            np.savez_compressed(out, **g)
            graph_paths.append(out)
        except Exception:
            log.debug("Graph build failed for %s:\n%s", sid, traceback.format_exc())
            graph_paths.append(None)

    df = labels_df.copy()
    df["graph_path"] = graph_paths
    before = len(df)
    df.dropna(subset=["graph_path"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Grain graphs: %d/%d built successfully.", len(df), before)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 2.  TF DATASET — padded graph batches
# ═════════════════════════════════════════════════════════════════════════════

def _load_graph_npz(path: str, cfg: Dict
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray]:
    """
    Load a graph from npz and apply padding/truncation so every sample
    has exactly (max_nodes, max_edges) for batching.

    Returns: node_feats, edge_src, edge_dst, edge_feats, node_mask
    """
    max_N = cfg["max_nodes"]
    max_E = cfg["max_edges"]
    nf_d  = cfg["node_in_dim"]
    ef_d  = cfg["edge_in_dim"]

    d = np.load(path)
    nf = d["node_feats"].astype(np.float32)   # (N, nf_d)
    es = d["edge_src"].astype(np.int32)        # (E,)
    ed = d["edge_dst"].astype(np.int32)        # (E,)
    ef = d["edge_feats"].astype(np.float32)    # (E, ef_d)

    N, E = len(nf), len(es)

    # ── truncate ──────────────────────────────────────────────────────────────
    if N > max_N:
        nf = nf[:max_N]
        valid = (es < max_N) & (ed < max_N)
        es, ed, ef = es[valid], ed[valid], ef[valid]
        E = len(es)

    if E > max_E:
        es, ed, ef = es[:max_E], ed[:max_E], ef[:max_E]
        E = max_E

    # ── pad ───────────────────────────────────────────────────────────────────
    node_mask = np.zeros(max_N, dtype=np.float32)
    node_mask[:N] = 1.0

    nf_pad = np.zeros((max_N, nf_d), dtype=np.float32)
    nf_pad[:N] = nf

    es_pad = np.zeros(max_E, dtype=np.int32)
    es_pad[:E] = es
    ed_pad = np.zeros(max_E, dtype=np.int32)
    ed_pad[:E] = ed

    ef_pad = np.zeros((max_E, ef_d), dtype=np.float32)
    ef_pad[:E] = ef

    # Edge validity mask (1 = real edge, 0 = padding)
    edge_mask = np.zeros((max_E, 1), dtype=np.float32)
    edge_mask[:E] = 1.0
    # Append to edge feats for the conv layer
    ef_pad_full = np.concatenate([ef_pad, edge_mask], axis=-1)  # (max_E, ef_d+1)

    return nf_pad, es_pad, ed_pad, ef_pad_full, node_mask


def _py_load_graph(path_bytes: bytes, label: np.ndarray,
                   cfg: Dict) -> Tuple:
    path = path_bytes.decode("utf-8")
    nf, es, ed, ef, nm = _load_graph_npz(path, cfg)
    return nf, es, ed, ef, nm, label.astype(np.float32)


def build_dataset(graph_paths: List[str],
                  labels: np.ndarray,
                  cfg: Dict,
                  shuffle: bool = False) -> tf.data.Dataset:
    """Build a tf.data.Dataset of (graph_inputs, labels)."""
    max_N    = cfg["max_nodes"]
    max_E    = cfg["max_edges"]
    nf_d     = cfg["node_in_dim"]
    ef_d_ext = cfg["edge_in_dim"] + 1   # +1 for edge_mask appended above
    n_labels = labels.shape[1]

    path_ds  = tf.data.Dataset.from_tensor_slices(
        [p.encode("utf-8") for p in graph_paths]
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(graph_paths), 1000),
                        seed=cfg["seed"])

    def _load_fn(path_tensor, label_tensor):
        nf, es, ed, ef, nm, lbl = tf.numpy_function(
            func=lambda p, l: _py_load_graph(p, l, cfg),
            inp=[path_tensor, label_tensor],
            Tout=[tf.float32, tf.int32, tf.int32,
                  tf.float32, tf.float32, tf.float32],
        )
        nf.set_shape([max_N, nf_d])
        es.set_shape([max_E])
        ed.set_shape([max_E])
        ef.set_shape([max_E, ef_d_ext])
        nm.set_shape([max_N])
        lbl.set_shape([n_labels])
        return (nf, es, ed, ef, nm), lbl

    ds = ds.map(_load_fn,
                num_parallel_calls=cfg["num_parallel"],
                deterministic=not shuffle)
    ds = ds.batch(cfg["batch_size"], drop_remainder=False)
    ds = ds.prefetch(cfg["prefetch"])
    return ds


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODEL — CGCNN Keras implementation
# ═════════════════════════════════════════════════════════════════════════════

class CGCNNConv(layers.Layer):
    """
    One CGCNN convolutional layer (Xie & Grossman, 2018).

    For each node i:
        z_ij    = Dense([h_i ‖ h_j ‖ e_ij])   # concat then project
        gate_ij = sigmoid(z_ij)
        msg_ij  = softplus(z_ij)
        h_i'    = BN(h_i + Σ_j (gate_ij ⊙ msg_ij) * edge_valid_ij)
    """

    def __init__(self, hidden_dim: int, l2: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.fc_full    = layers.Dense(
            hidden_dim * 2,
            use_bias=True,
            kernel_regularizer=regularizers.l2(l2),
        )
        self.bn = layers.BatchNormalization()

    def call(self,
             node_feats:  tf.Tensor,   # (B, N, D)
             edge_src:    tf.Tensor,   # (B, E)   int32
             edge_dst:    tf.Tensor,   # (B, E)   int32
             edge_feats:  tf.Tensor,   # (B, E, ef+1)  last col = edge_mask
             node_mask:   tf.Tensor,   # (B, N)   float32
             training:    bool = False) -> tf.Tensor:

        B   = tf.shape(node_feats)[0]
        D   = self.hidden_dim
        max_N = tf.shape(node_feats)[1]
        max_E = tf.shape(edge_src)[1]

        edge_valid = edge_feats[..., -1:]                   # (B, E, 1)
        ef_clean   = edge_feats[..., :-1]                   # (B, E, ef)

        # Gather src / dst node features for each edge
        # tf.gather with batch_dims=1 needs indices in (B, E)
        src_feats = tf.gather(node_feats, edge_src, batch_dims=1)   # (B, E, D)
        dst_feats = tf.gather(node_feats, edge_dst, batch_dims=1)   # (B, E, D)

        # Concatenate [h_i ‖ h_j ‖ e_ij]
        z = tf.concat([src_feats, dst_feats, ef_clean], axis=-1)    # (B, E, 2D+ef)
        z = self.fc_full(z)                                          # (B, E, 2D)

        gate = tf.sigmoid(z[..., :D])                               # (B, E, D)
        msg  = tf.nn.softplus(z[..., D:])                           # (B, E, D)
        agg  = gate * msg * edge_valid                               # (B, E, D)

        # Scatter-add onto destination nodes (and src for undirected)
        def _scatter(indices, values):
            # indices: (B, E)  values: (B, E, D)
            idx_oh = tf.one_hot(indices, max_N, dtype=tf.float32)  # (B, E, N)
            # batched matmul: (B, N, E) @ (B, E, D) = (B, N, D)
            return tf.matmul(
                tf.transpose(idx_oh, [0, 2, 1]),
                values,
            )

        agg_at_dst = _scatter(edge_dst, agg)    # (B, N, D)
        agg_at_src = _scatter(edge_src, agg)    # (B, N, D)  (undirected)
        agg_total  = agg_at_dst + agg_at_src    # (B, N, D)

        # Residual + BN
        out = node_feats + agg_total                                 # (B, N, D)
        out = self.bn(out, training=training)

        # Zero out padding positions
        out = out * tf.expand_dims(node_mask, axis=-1)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"hidden_dim": self.hidden_dim})
        return cfg


class SqueezeExcitation(layers.Layer):
    """Channel-wise SE block."""

    def __init__(self, channels: int, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.ratio = ratio
        self.fc1 = layers.Dense(max(1, channels // ratio), activation="relu")
        self.fc2 = layers.Dense(channels, activation="sigmoid")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        se = tf.reduce_mean(x, axis=-1, keepdims=True)
        se = self.fc1(se)
        se = self.fc2(se)
        return x * se

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "ratio": self.ratio})
        return cfg


class MultiOutputHuber(keras.losses.Loss):
    """Mean Huber loss over multiple label columns."""

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self._huber = keras.losses.Huber(delta=delta, reduction="sum_over_batch_size")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._huber(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg["delta"] = self.delta
        return cfg


def build_cgcnn_model(cfg: Dict) -> keras.Model:
    """
    Build the full CGCNN-style GNN model using the Keras functional API.
    Inputs are fixed-size padded tensors for efficient batching.
    """
    max_N    = cfg["max_nodes"]
    max_E    = cfg["max_edges"]
    nf_d     = cfg["node_in_dim"]
    ef_d_ext = cfg["edge_in_dim"] + 1   # edge_mask appended
    D        = cfg["hidden_dim"]
    n_out    = len(cfg["label_cols"])
    l2       = cfg["l2_reg"]

    # ── inputs ────────────────────────────────────────────────────────────────
    inp_node_feats = keras.Input(shape=(max_N, nf_d),     name="node_feats")
    inp_edge_src   = keras.Input(shape=(max_E,),    dtype=tf.int32, name="edge_src")
    inp_edge_dst   = keras.Input(shape=(max_E,),    dtype=tf.int32, name="edge_dst")
    inp_edge_feats = keras.Input(shape=(max_E, ef_d_ext), name="edge_feats")
    inp_node_mask  = keras.Input(shape=(max_N,),          name="node_mask")

    # ── initial node/edge embeddings ──────────────────────────────────────────
    h = layers.Dense(D, activation="gelu",
                     kernel_regularizer=regularizers.l2(l2),
                     name="node_embed")(inp_node_feats)           # (B, N, D)

    # ── CGCNN conv stack ──────────────────────────────────────────────────────
    for i in range(cfg["n_conv"]):
        h = CGCNNConv(D, l2=l2, name=f"cgcnn_conv_{i}")(
            h, inp_edge_src, inp_edge_dst, inp_edge_feats, inp_node_mask
        )

    # ── global mean pooling (masked) ──────────────────────────────────────────
    mask_expand = tf.expand_dims(inp_node_mask, axis=-1)           # (B, N, 1)
    sum_feats   = tf.reduce_sum(h * mask_expand, axis=1)           # (B, D)
    n_real      = tf.reduce_sum(inp_node_mask, axis=1, keepdims=True) + 1e-9
    graph_emb   = sum_feats / n_real                               # (B, D)

    # ── head ──────────────────────────────────────────────────────────────────
    x = layers.Dense(512, activation="gelu",
                     kernel_regularizer=regularizers.l2(l2))(graph_emb)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg["dropout_rate"])(x)

    # SE block
    se = layers.Dense(max(1, 512 // cfg["se_ratio"]), activation="relu")(x)
    se = layers.Dense(512, activation="sigmoid")(se)
    x  = layers.Multiply()([x, se])

    # Embedding layer (256-d output to Fusion Layer)
    embedding = layers.Dense(cfg["embed_dim"], activation="gelu",
                             kernel_regularizer=regularizers.l2(l2),
                             name="embedding")(x)
    embedding = layers.BatchNormalization(name="embedding_bn")(embedding)
    embedding = layers.Dropout(cfg["dropout_rate"])(embedding)

    # Output head — float32 even under mixed precision
    output = layers.Dense(n_out, dtype="float32", name="output")(embedding)

    model = keras.Model(
        inputs=[inp_node_feats, inp_edge_src, inp_edge_dst,
                inp_edge_feats, inp_node_mask],
        outputs=output,
        name="cgcnn_branch",
    )
    return model


def build_embedding_extractor(model: keras.Model, cfg: Dict) -> keras.Model:
    """Extract the 256-d embedding sub-model from a trained CGCNN model."""
    emb_layer = model.get_layer("embedding_bn")
    return keras.Model(
        inputs=model.inputs,
        outputs=emb_layer.output,
        name="cgcnn_embedding_extractor",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _get_optimizer(lr, cfg: Dict) -> keras.optimizers.Optimizer:
    return keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=cfg["weight_decay"],
        clipnorm=cfg["grad_clip"],
    )


def compile_model(model: keras.Model, lr, cfg: Dict) -> None:
    model.compile(
        optimizer=_get_optimizer(lr, cfg),
        loss=MultiOutputHuber(delta=1.0),
        metrics=["mae"],
    )


def _load_state(ckpt_dir: str) -> Dict:
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {"completed_epochs": 0, "best_val_loss": float("inf"), "phase": "warmup"}


def _save_state(state: Dict, ckpt_dir: str) -> None:
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


class _StateCallback(keras.callbacks.Callback):
    """Persist training state after every epoch for safe resume."""

    def __init__(self, cfg: Dict, state: Dict):
        super().__init__()
        self.cfg   = cfg
        self.state = state

    def on_epoch_end(self, epoch: int, logs=None) -> None:
        self.state["completed_epochs"] = epoch + 1
        val_loss = (logs or {}).get("val_loss", float("inf"))
        if val_loss < self.state["best_val_loss"]:
            self.state["best_val_loss"] = val_loss
        _save_state(self.state, self.cfg["ckpt_dir"])

        # Save last checkpoint every epoch (OOM-safe resume)
        self.model.save(os.path.join(self.cfg["ckpt_dir"], "last_model.keras"))

        # CSV logging
        row = {"epoch": epoch + 1, "timestamp": datetime.utcnow().isoformat(),
               **(logs or {})}
        csv_path = os.path.join(self.cfg["log_dir"], "training_log.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                w.writeheader()
            w.writerow(row)


def get_standard_callbacks(cfg: Dict, state: Dict, lr_scheduler: Optional[Any] = None) -> List[keras.callbacks.Callback]:
    best_path = os.path.join(cfg["ckpt_dir"], "best_model.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.TensorBoard(
            log_dir=cfg["tb_dir"],
            histogram_freq=0, update_freq="epoch",
        ),
        _StateCallback(cfg, state),
    ]

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, keras.callbacks.Callback):
            callbacks.insert(2, lr_scheduler)
        else:
            callbacks.insert(2, keras.callbacks.LearningRateScheduler(lr_scheduler))
    else:
        callbacks.insert(2, keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-7, verbose=1,
        ))
    return callbacks


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION & PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model: keras.Model,
             test_ds: tf.data.Dataset,
             label_cols: List[str],
             scaler: StandardScaler,
             cfg: Dict) -> Dict:
    """Run evaluation on the held-out test set; save scatter plot + JSON."""
    y_pred_scaled, y_true_scaled = [], []
    for batch_x, batch_y in test_ds:
        pred = model(batch_x, training=False).numpy()
        y_pred_scaled.append(pred)
        y_true_scaled.append(batch_y.numpy())

    y_pred_s = np.concatenate(y_pred_scaled, axis=0)
    y_true_s = np.concatenate(y_true_scaled, axis=0)

    y_pred = scaler.inverse_transform(y_pred_s)
    y_true = scaler.inverse_transform(y_true_s)

    metrics: Dict = {}
    for i, col in enumerate(label_cols):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        metrics[col] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        log.info("  %-25s  MAE=%8.4f  RMSE=%8.4f  R²=%6.4f", col, mae, rmse, r2)

    metrics_path = os.path.join(cfg["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics → %s", metrics_path)

    # Scatter plot
    n_cols   = len(label_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    for i, (ax, col) in enumerate(zip(axes, label_cols)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"True {col}")
        ax.set_ylabel(f"Predicted {col}")
        ax.set_title(f"{col}  R²={metrics[col]['R2']:.3f}")
    fig.suptitle("CGCNN Branch — Test Set Evaluation", fontsize=13)
    fig.tight_layout()
    scatter_path = os.path.join(cfg["output_dir"], "eval_scatter.png")
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    log.info("Scatter plot → %s", scatter_path)
    return metrics


def plot_history(histories: List, labels: List[str], cfg: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    offset = 0
    for hist, label in zip(histories, labels):
        n = len(hist.history["loss"])
        ep = list(range(offset + 1, offset + n + 1))
        axes[0].plot(ep, hist.history["loss"], label=f"{label} train")
        if "val_loss" in hist.history:
            axes[0].plot(ep, hist.history["val_loss"], "--",
                         label=f"{label} val")
        if "mae" in hist.history:
            axes[1].plot(ep, hist.history["mae"], label=f"{label} train")
        if "val_mae" in hist.history:
            axes[1].plot(ep, hist.history["val_mae"], "--",
                         label=f"{label} val")
        offset += n
    axes[0].set(title="Loss (Huber)", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(fontsize=8)
    axes[1].set(title="MAE", xlabel="Epoch", ylabel="MAE")
    axes[1].legend(fontsize=8)
    fig.suptitle("CGCNN Branch — Training History", fontsize=13)
    fig.tight_layout()
    path = os.path.join(cfg["output_dir"], "training_history.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Training history plot → %s", path)


def extract_and_save_embeddings(model: keras.Model,
                                 all_ds: tf.data.Dataset,
                                 cfg: Dict) -> None:
    """Extract and save 256-d embeddings for all samples, plus the extractor."""
    extractor = build_embedding_extractor(model, cfg)
    embs = []
    for batch_x, _ in all_ds:
        embs.append(extractor(batch_x, training=False).numpy())
    embs = np.concatenate(embs, axis=0)
    emb_path = os.path.join(cfg["output_dir"], "embeddings.npy")
    np.save(emb_path, embs)
    log.info("Embeddings (%s) → %s", embs.shape, emb_path)

    ext_path = os.path.join(cfg["output_dir"], "embedding_extractor.keras")
    extractor.save(ext_path)
    log.info("Embedding extractor → %s", ext_path)

    best_path = os.path.join(cfg["output_dir"], "best_model.keras")
    model.save(best_path)
    log.info("Full model → %s", best_path)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = CONFIG
    setup_environment(cfg)
    log.info("=" * 60)
    log.info("  CGCNN Atomic-Structure Branch — training start")
    log.info("=" * 60)

    # ── Step 1: Labels ─────────────────────────────────────────────────────
    labels_df = generate_labels_csv(
        h5_dir=cfg["h5_dir"],
        csv_path=cfg["labels_csv"],
        voxel_size_um=cfg["voxel_size_um"],
        compute_misorientation=True,
    )

    # ── Step 2: Build / load grain-graph cache ────────────────────────────
    labels_df = build_graph_cache(labels_df, cfg)

    if len(labels_df) < 20:
        raise RuntimeError(
            f"Only {len(labels_df)} valid grain graphs found. "
            "Verify h5 files are extracted correctly."
        )

    # ── Step 3: Drop NaN labels ────────────────────────────────────────────
    labels_df.dropna(subset=cfg["label_cols"], inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    log.info("Valid samples after label filtering: %d", len(labels_df))

    # ── Step 4: Normalise labels ───────────────────────────────────────────
    label_matrix = labels_df[cfg["label_cols"]].values.astype(np.float32)
    scaler       = StandardScaler()
    label_scaled = scaler.fit_transform(label_matrix).astype(np.float32)

    scaler_path = os.path.join(cfg["output_dir"], "label_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Label scaler → %s", scaler_path)

    # ── Step 5: Split ──────────────────────────────────────────────────────
    all_paths = labels_df["graph_path"].tolist()
    idx       = np.arange(len(labels_df))
    tr_i, te_i = train_test_split(idx, test_size=cfg["test_frac"],
                                  random_state=cfg["seed"])
    tr_i, va_i = train_test_split(
        tr_i,
        test_size=cfg["val_frac"] / (1.0 - cfg["test_frac"]),
        random_state=cfg["seed"],
    )
    log.info("Split → train: %d | val: %d | test: %d",
             len(tr_i), len(va_i), len(te_i))

    def sel(lst, indices): return [lst[i] for i in indices]

    train_ds = build_dataset(sel(all_paths, tr_i), label_scaled[tr_i],
                              cfg, shuffle=True)
    val_ds   = build_dataset(sel(all_paths, va_i), label_scaled[va_i],
                              cfg, shuffle=False)
    test_ds  = build_dataset(sel(all_paths, te_i), label_scaled[te_i],
                              cfg, shuffle=False)
    all_ds   = build_dataset(all_paths, label_scaled, cfg, shuffle=False)

    # ── Step 6: Load state ─────────────────────────────────────────────────
    state = _load_state(cfg["ckpt_dir"])
    done  = state["completed_epochs"]
    total = cfg["warmup_epochs"] + cfg["finetune_epochs"]
    log.info("Completed epochs so far: %d / %d", done, total)

    custom_objs = {
        "CGCNNConv": CGCNNConv,
        "SqueezeExcitation": SqueezeExcitation,
        "MultiOutputHuber": MultiOutputHuber,
    }

    last_ckpt = os.path.join(cfg["ckpt_dir"], "last_model.keras")
    best_ckpt = os.path.join(cfg["ckpt_dir"], "best_model.keras")

    # ── Step 7: Warm-up phase ──────────────────────────────────────────────
    history_warmup = None
    if done < cfg["warmup_epochs"]:
        log.info("─" * 60)
        log.info("PHASE 1 – Warm-up  (all layers, lr=%.0e)", cfg["lr_warmup"])
        log.info("─" * 60)

        if done > 0 and os.path.exists(last_ckpt):
            log.info("Resuming warm-up from epoch %d …", done)
            model = keras.models.load_model(last_ckpt, custom_objects=custom_objs)
            compile_model(model, cfg["lr_warmup"], cfg)
        else:
            model = build_cgcnn_model(cfg)
            compile_model(model, cfg["lr_warmup"], cfg)
            model.summary(expand_nested=False, print_fn=log.info)

        state["phase"] = "warmup"
        history_warmup = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["warmup_epochs"],
            initial_epoch=done,
            callbacks=get_standard_callbacks(cfg, state),
            verbose=1,
        )
        done = state["completed_epochs"]
        warmup_path = os.path.join(cfg["ckpt_dir"], "warmup_final.keras")
        model.save(warmup_path)
        log.info("Warmup final checkpoint → %s", warmup_path)

    # Load best warmup model before fine-tune
    if os.path.exists(best_ckpt):
        model = keras.models.load_model(best_ckpt, custom_objects=custom_objs)

    # ── Step 8: Fine-tune phase ────────────────────────────────────────────
    history_finetune = None
    if done < total:
        log.info("─" * 60)
        log.info("PHASE 2 – Fine-tune  (cosine decay, lr=%.0e)", cfg["lr_finetune"])
        log.info("─" * 60)

        if done > cfg["warmup_epochs"] and os.path.exists(last_ckpt):
            log.info("Resuming fine-tune from epoch %d …", done)
            model = keras.models.load_model(last_ckpt, custom_objects=custom_objs)

        steps_per_epoch = max(1, len(tr_i) // cfg["batch_size"])
        lr_schedule_obj = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg["lr_finetune"],
            decay_steps=steps_per_epoch * cfg["finetune_epochs"],
            alpha=1e-7,
        )

        def lr_scheduler_fn(epoch, lr):  # noqa: ARG001
            relative_step = (epoch - cfg["warmup_epochs"]) * steps_per_epoch
            if relative_step < 0:
                return cfg["lr_finetune"]
            return float(lr_schedule_obj(relative_step))

        model.compile(
            optimizer=_get_optimizer(cfg["lr_finetune"], cfg),
            loss=MultiOutputHuber(delta=1.0),
            metrics=["mae"],
        )
        state["phase"] = "finetune"
        history_finetune = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total,
            initial_epoch=done,
            callbacks=get_standard_callbacks(cfg, state, lr_scheduler=lr_scheduler_fn),
            verbose=1,
        )

    # ── Step 9: Load best model ────────────────────────────────────────────
    if os.path.exists(best_ckpt):
        model = keras.models.load_model(best_ckpt, custom_objects=custom_objs)

    # ── Step 10: Evaluate ──────────────────────────────────────────────────
    log.info("Evaluating on held-out test set …")
    evaluate(model, test_ds, cfg["label_cols"], scaler, cfg)

    # ── Step 11: Plots ─────────────────────────────────────────────────────
    plot_history(
        [h for h in [history_warmup, history_finetune] if h is not None],
        [l for l, h in [("Warm-up", history_warmup),
                         ("Fine-tune", history_finetune)] if h is not None],
        cfg,
    )

    # ── Step 12: Save embeddings & extractors ─────────────────────────────
    log.info("Extracting embeddings for Fusion Layer …")
    extract_and_save_embeddings(model, all_ds, cfg)

    log.info("=" * 60)
    log.info("  CGCNN training complete.")
    log.info("  Best model  → %s", best_ckpt)
    log.info("  Embeddings  → %s/embeddings.npy", cfg["output_dir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
