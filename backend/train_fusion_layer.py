"""
train_fusion_layer.py
═══════════════════════════════════════════════════════════════════════════════
Fusion Layer — combines embeddings from all three trained branches:
  • Branch 1 : ResNet50    outputs/resnet/embedding_extractor.keras    (256-d)
  • Branch 2 : MobileNetV2 outputs/mobilenet/embedding_extractor.keras (256-d)
  • Branch 3 : CGCNN       outputs/cgcnn/embedding_extractor.keras     (256-d)

Two operation modes:
  (A) OFFLINE  — branches are frozen; Fusion trains on pre-computed embeddings.
                 Fast (no image/graph forward passes during training).
  (B) ONLINE   — branches are fine-tuned jointly with the Fusion Layer.
                 Slower but higher accuracy.  Set FUSION_CONFIG["mode"] = "online".

ARCHITECTURE (Fusion Layer)
───────────────────────────
  [emb_resnet (256)] ─┐
  [emb_mobilenet(256)]─┼─► Concat (768) ─► Dense(768, GELU) ─► BN ─► Dropout
  [emb_cgcnn  (256)] ─┘                ─► SE-Block(768)
                                        ─► Dense(512, GELU) ─► BN ─► Dropout
                                        ─► Dense(256, GELU) ─► BN ─► Dropout
                                        ─► Dense(n_outputs, float32)

USAGE
─────
    # Make sure all three branch training scripts have completed first:
    python train_resnet_microstructure.py
    python train_mobilenet_texture.py
    python train_cgcnn_atomic.py

    # Then train the Fusion Layer:
    python train_fusion_layer.py

DIRECTORY LAYOUT
────────────────
  outputs/
  ├── resnet/embedding_extractor.keras    ← pre-trained branch extractors
  ├── mobilenet/embedding_extractor.keras
  ├── cgcnn/embedding_extractor.keras
  └── fusion/
      ├── best_model.keras                ← full end-to-end fused model
      ├── fusion_head.keras               ← head only (for inference)
      ├── label_scaler.pkl
      ├── eval_scatter.png
      ├── training_history.png
      └── metrics.json

  checkpoints/fusion/
  ├── best_model.keras
  ├── last_model.keras
  └── training_state.json

  logs/fusion/
  ├── training_log.csv
  └── tensorboard/
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
from typing import Any, Dict, List, Optional, Tuple

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
from data_utils import generate_labels_csv, pair_microstructure_images
try:
    from train_cgcnn_atomic import (
        CGCNNConv, SqueezeExcitation as _SE_CGCNN,
        MultiOutputHuber as _Huber_CGCNN,
        build_graph_cache, build_dataset as build_graph_dataset,
        _load_graph_npz,
    )
    _CGCNN_AVAILABLE = True
except ImportError:
    _CGCNN_AVAILABLE = False
    logging.warning("train_cgcnn_atomic not importable — CGCNN custom objects "
                    "unavailable; load from saved weights only.")
from training_utils import (
    setup_cuda,
    setup_train_logger,
    get_git_info,
    tag_experiment,
    save_version_file,
    get_enhanced_callbacks,
    build_gradient_accumulation_model,
)

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
FUSION_CONFIG: Dict = {
    # ── branch extractor paths ─────────────────────────────────────────────
    "resnet_extractor":    "outputs/resnet/embedding_extractor.keras",
    "mobilenet_extractor": "outputs/mobilenet/embedding_extractor.keras",
    "cgcnn_extractor":     "outputs/cgcnn/embedding_extractor.keras",

    # ── pre-computed embeddings (offline mode) ────────────────────────────
    "resnet_embeddings":   "outputs/resnet/embeddings.npy",
    "mobilenet_embeddings":"outputs/mobilenet/embeddings.npy",
    "cgcnn_embeddings":    "outputs/cgcnn/embeddings.npy",

    # ── branch scalers (to inverse-transform branch-specific outputs) ─────
    "resnet_scaler":       "outputs/resnet/label_scaler.pkl",
    "mobilenet_scaler":    "outputs/mobilenet/label_scaler.pkl",
    "cgcnn_scaler":        "outputs/cgcnn/label_scaler.pkl",

    # ── data paths ────────────────────────────────────────────────────────
    "h5_dir":              "data/h5_slices",
    "micro_dir":           "data/2D_EBSD",
    "pole_fig_dir":        "data/pole_figures",
    "graph_cache_dir":     "data/graphs",
    "labels_csv":          "data/labels.csv",

    # ── output directories ────────────────────────────────────────────────
    "ckpt_dir":            "checkpoints/fusion",
    "log_dir":             "logs/fusion",
    "output_dir":          "outputs/fusion",
    "tb_dir":              "logs/fusion/tensorboard",

    # ── label targets ─────────────────────────────────────────────────────
    "label_cols": [
        "grain_size",
        "texture_index",
        "avg_misorientation",
    ],

    # ── mode ──────────────────────────────────────────────────────────────
    # "offline" : use pre-computed .npy embeddings; branches not loaded.
    # "online"  : load branch extractors, fine-tune end-to-end.
    "mode":                "offline",

    # ── online-mode: branch fine-tune settings ────────────────────────────
    "online_branch_lr":    1e-6,   # very small lr for branch layers
    "online_unfreeze_pct": 0.20,   # % of branch layers to unfreeze

    # ── fusion head architecture ──────────────────────────────────────────
    "embed_dim_each":      256,    # per-branch embedding dimension
    "n_branches":          3,      # resnet + mobilenet + cgcnn
    "fusion_dims":         [768, 512, 256],  # Dense layer widths
    "dropout_rate":        0.35,
    "l2_reg":              1e-4,
    "se_ratio":            8,

    # ── training ──────────────────────────────────────────────────────────
    "batch_size":          32,
    "warmup_epochs":       10,     # head only, branches frozen
    "finetune_epochs":     60,     # cosine decay
    "lr_warmup":           1e-3,
    "lr_finetune":         5e-5,
    "weight_decay":        1e-4,
    "grad_clip":           1.0,

    # ── splits ────────────────────────────────────────────────────────────
    "val_frac":            0.15,
    "test_frac":           0.10,

    # ── misc ──────────────────────────────────────────────────────────────
    "seed":                42,
    "use_mixed_precision": True,
    "prefetch":            tf.data.AUTOTUNE,
    "num_parallel":        tf.data.AUTOTUNE,
    "voxel_size_um":       1.0,

    # ── gradient accumulation ───────────────────────────────────────────────
    "grad_accum_steps":    4,

    # ── checkpoints ─────────────────────────────────────────────────────────
    "ckpt_every_n_epochs": 10,
    "max_kept_ckpts":      5,

    # ── version control ───────────────────────────────────────────────────
    "git_tag_runs":        False,

    # CGCNN graph settings (needed for online mode / direct graph loading)
    "max_nodes":           512,
    "max_edges":           4096,
    "node_in_dim":         5,
    "edge_in_dim":         2,
}


# ═════════════════════════════════════════════════════════════════════════════
# 0.  ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

def setup_environment(cfg: Dict) -> Dict:
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    for d in (cfg["ckpt_dir"], cfg["log_dir"],
              cfg["output_dir"], cfg["tb_dir"]):
        os.makedirs(d, exist_ok=True)

    # ── CUDA diagnostics & device setup (replaces manual GPU config) ──────
    device_diag = setup_cuda(cfg, logger=log)
    return device_diag


# ═════════════════════════════════════════════════════════════════════════════
# 1.  OFFLINE MODE — pre-computed embedding datasets
# ═════════════════════════════════════════════════════════════════════════════

def load_offline_embeddings(cfg: Dict,
                             labels_df: pd.DataFrame
                             ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-computed embeddings from all three branch .npy files.
    Aligns them by sample_id order in labels_df.

    The three .npy files must have been saved in the SAME sample order as
    labels.csv (both branch scripts use the same CSV). Rows with any missing
    embedding file are dropped.

    Returns
    -------
    fused_emb : (N, 768)  float32  concatenated embeddings
    label_mat : (N, n_labels) float32
    sample_ids : list of sample_id strings (aligned)
    """
    required = [
        cfg["resnet_embeddings"],
        cfg["mobilenet_embeddings"],
        cfg["cgcnn_embeddings"],
    ]
    branch_names = ["resnet", "mobilenet", "cgcnn"]
    embs = {}
    for path, name in zip(required, branch_names):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file not found: {path}\n"
                f"Run the {name} branch training script first."
            )
        arr = np.load(path).astype(np.float32)
        log.info("Loaded %s embeddings: %s", name, arr.shape)
        embs[name] = arr

    # Validate shapes
    n_samples = len(labels_df)
    for name, arr in embs.items():
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"{name} embeddings have {arr.shape[0]} rows but "
                f"labels_df has {n_samples} rows. "
                "Re-run all branch scripts with the same labels.csv."
            )

    fused = np.concatenate(
        [embs["resnet"], embs["mobilenet"], embs["cgcnn"]], axis=1
    ).astype(np.float32)   # (N, 768)

    label_mat = labels_df[cfg["label_cols"]].values.astype(np.float32)
    sample_ids = labels_df["sample_id"].tolist()
    return fused, label_mat, sample_ids


def build_offline_dataset(embeddings: np.ndarray,
                           labels: np.ndarray,
                           cfg: Dict,
                           shuffle: bool = False) -> tf.data.Dataset:
    """Simple tf.data pipeline for the pre-computed embedding + label pairs."""
    ds = tf.data.Dataset.from_tensor_slices(
        (embeddings.astype(np.float32), labels.astype(np.float32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(embeddings), 2000),
                        seed=cfg["seed"])
    ds = ds.batch(cfg["batch_size"], drop_remainder=False)
    ds = ds.prefetch(cfg["prefetch"])
    return ds


# ═════════════════════════════════════════════════════════════════════════════
# 2.  ONLINE MODE — per-sample multi-modal dataset
# ═════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _load_image(path: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False,
                                dtype=tf.uint8)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img.set_shape([224, 224, 3])
    return img


def _py_load_online_sample(micro_path: bytes,
                            pole_path: bytes,
                            graph_path: bytes,
                            label: np.ndarray,
                            cfg: Dict) -> Tuple:
    """Python-side loader for one online-mode sample."""
    from train_cgcnn_atomic import _load_graph_npz
    micro_img = _load_image(tf.constant(micro_path.decode("utf-8"))).numpy()
    pole_img  = _load_image(tf.constant(pole_path.decode("utf-8"))).numpy()
    nf, es, ed, ef, nm = _load_graph_npz(graph_path.decode("utf-8"), cfg)
    return (micro_img, pole_img, nf, es, ed, ef, nm, label.astype(np.float32))


def build_online_dataset(micro_paths: List[str],
                          pole_paths: List[str],
                          graph_paths: List[str],
                          labels: np.ndarray,
                          cfg: Dict,
                          shuffle: bool = False) -> tf.data.Dataset:
    """Build a dataset that returns (micro_img, pole_img, graph_tuple), label."""
    max_N    = cfg["max_nodes"]
    max_E    = cfg["max_edges"]
    nf_d     = cfg["node_in_dim"]
    ef_d_ext = cfg["edge_in_dim"] + 1
    n_labels = labels.shape[1]

    ds = tf.data.Dataset.from_tensor_slices((
        [p.encode("utf-8") for p in micro_paths],
        [p.encode("utf-8") for p in pole_paths],
        [p.encode("utf-8") for p in graph_paths],
        labels.astype(np.float32),
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(micro_paths), 1000),
                        seed=cfg["seed"])

    def _load_fn(mp, pp, gp, lbl):
        outputs = tf.numpy_function(
            func=lambda a, b, c, d: _py_load_online_sample(a, b, c, d, cfg),
            inp=[mp, pp, gp, lbl],
            Tout=[tf.float32, tf.float32, tf.float32,
                  tf.int32, tf.int32, tf.float32, tf.float32, tf.float32],
        )
        micro_img, pole_img, nf, es, ed, ef, nm, lbl_out = outputs
        micro_img.set_shape([224, 224, 3])
        pole_img.set_shape([224, 224, 3])
        nf.set_shape([max_N, nf_d])
        es.set_shape([max_E])
        ed.set_shape([max_E])
        ef.set_shape([max_E, ef_d_ext])
        nm.set_shape([max_N])
        lbl_out.set_shape([n_labels])
        return (micro_img, pole_img, nf, es, ed, ef, nm), lbl_out

    ds = ds.map(_load_fn,
                num_parallel_calls=cfg["num_parallel"],
                deterministic=not shuffle)
    ds = ds.batch(cfg["batch_size"], drop_remainder=False)
    ds = ds.prefetch(cfg["prefetch"])
    return ds


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MODEL — Fusion Head + optional branch extractors
# ═════════════════════════════════════════════════════════════════════════════

class MultiOutputHuber(keras.losses.Loss):
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self._huber = keras.losses.Huber(delta=delta, reduction="sum_over_batch_size")

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._huber(y_true, y_pred)

    def get_config(self):
        cfg = super().get_config()
        cfg["delta"] = self.delta
        return cfg


def build_fusion_head(cfg: Dict) -> keras.Model:
    """
    Fusion head that operates on concatenated (768-d) embeddings.
    Used in OFFLINE mode, or as the top of the ONLINE end-to-end model.
    """
    total_dim = cfg["embed_dim_each"] * cfg["n_branches"]   # 768
    n_out     = len(cfg["label_cols"])
    l2        = cfg["l2_reg"]
    dims      = cfg["fusion_dims"]   # [768, 512, 256]

    inp = keras.Input(shape=(total_dim,), name="fused_embedding")
    x   = inp

    for i, d in enumerate(dims):
        x = layers.Dense(d, activation="gelu",
                          kernel_regularizer=regularizers.l2(l2),
                          name=f"fusion_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"fusion_bn_{i}")(x)
        x = layers.Dropout(cfg["dropout_rate"],
                            name=f"fusion_drop_{i}")(x)

        if i == 0:
            # SE block after the first fusion layer
            se = layers.Dense(max(1, d // cfg["se_ratio"]),
                               activation="relu", name="se_fc1")(x)
            se = layers.Dense(d, activation="sigmoid", name="se_fc2")(se)
            x  = layers.Multiply(name="se_scale")([x, se])

    output = layers.Dense(n_out, dtype="float32", name="output")(x)
    return keras.Model(inputs=inp, outputs=output, name="fusion_head")


def build_online_model(cfg: Dict,
                       resnet_ext: keras.Model,
                       mobilenet_ext: keras.Model,
                       cgcnn_ext: keras.Model) -> keras.Model:
    """
    End-to-end model for ONLINE mode.
    Inputs: (micro_img, pole_img, nf, es, ed, ef, nm)
    """
    max_N    = cfg["max_nodes"]
    max_E    = cfg["max_edges"]
    nf_d     = cfg["node_in_dim"]
    ef_d_ext = cfg["edge_in_dim"] + 1

    inp_micro = keras.Input(shape=(224, 224, 3), name="micro_img")
    inp_pole  = keras.Input(shape=(224, 224, 3), name="pole_img")
    inp_nf    = keras.Input(shape=(max_N, nf_d), name="node_feats")
    inp_es    = keras.Input(shape=(max_E,), dtype=tf.int32, name="edge_src")
    inp_ed    = keras.Input(shape=(max_E,), dtype=tf.int32, name="edge_dst")
    inp_ef    = keras.Input(shape=(max_E, ef_d_ext), name="edge_feats")
    inp_nm    = keras.Input(shape=(max_N,), name="node_mask")

    emb_resnet    = resnet_ext(inp_micro, training=False)
    emb_mobilenet = mobilenet_ext(inp_pole, training=False)
    emb_cgcnn     = cgcnn_ext(
        [inp_nf, inp_es, inp_ed, inp_ef, inp_nm], training=False
    )

    fused = layers.Concatenate(name="concat_embeddings")(
        [emb_resnet, emb_mobilenet, emb_cgcnn]
    )

    head  = build_fusion_head(cfg)
    output = head(fused)

    return keras.Model(
        inputs=[inp_micro, inp_pole, inp_nf, inp_es, inp_ed, inp_ef, inp_nm],
        outputs=output,
        name="fusion_online",
    )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES  (same pattern as branch scripts)
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
    path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_epochs": 0, "best_val_loss": float("inf"), "phase": "warmup"}


def _save_state(state: Dict, ckpt_dir: str) -> None:
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION & PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model: keras.Model,
             test_ds: tf.data.Dataset,
             label_cols: List[str],
             scaler: StandardScaler,
             cfg: Dict) -> Dict:
    y_pred_s, y_true_s = [], []
    for batch_x, batch_y in test_ds:
        pred = model(batch_x, training=False).numpy()
        y_pred_s.append(pred)
        y_true_s.append(batch_y.numpy())

    y_pred_s = np.concatenate(y_pred_s, axis=0)
    y_true_s = np.concatenate(y_true_s, axis=0)
    y_pred   = scaler.inverse_transform(y_pred_s)
    y_true   = scaler.inverse_transform(y_true_s)

    metrics: Dict = {}
    for i, col in enumerate(label_cols):
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        metrics[col] = {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
        log.info("  %-25s  MAE=%8.4f  RMSE=%8.4f  R²=%6.4f", col, mae, rmse, r2)

    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics → %s/metrics.json", cfg["output_dir"])

    n_cols = len(label_cols)
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
    fig.suptitle("Fusion Layer — Test Set Evaluation", fontsize=13)
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
        n  = len(hist.history["loss"])
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
    fig.suptitle("Fusion Layer — Training History", fontsize=13)
    fig.tight_layout()
    path = os.path.join(cfg["output_dir"], "training_history.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Training history → %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def _get_custom_objects() -> Dict:
    """Collect all custom Keras objects needed to load saved models."""
    objs = {"MultiOutputHuber": MultiOutputHuber}
    if _CGCNN_AVAILABLE:
        objs.update({
            "CGCNNConv": CGCNNConv,
            "SqueezeExcitation": _SE_CGCNN,
            "MultiOutputHuber": _Huber_CGCNN,
        })
    return objs


def main() -> None:
    cfg = FUSION_CONFIG
    device_diag = setup_environment(cfg)

    # ── Structured file logger & run ID ───────────────────────────────────
    run_id, _ = setup_train_logger("fusion", cfg["log_dir"], logger=log)

    # ── Git version info ──────────────────────────────────────────────────
    git_info = get_git_info()
    log.info("Git commit : %s  (dirty=%s)", git_info["commit_hash"][:8],
             git_info["is_dirty"])

    log.info("=" * 60)
    log.info("  Fusion Layer — mode: %s", cfg["mode"].upper())
    log.info("  Run ID   : %s", run_id)
    log.info("  Grad Accum Steps: %d", cfg["grad_accum_steps"])
    log.info("=" * 60)

    # ── Step 1: Labels ─────────────────────────────────────────────────────
    labels_df = generate_labels_csv(
        h5_dir=cfg["h5_dir"],
        csv_path=cfg["labels_csv"],
        voxel_size_um=cfg["voxel_size_um"],
        compute_misorientation=True,
    )
    labels_df.dropna(subset=cfg["label_cols"], inplace=True)
    labels_df.reset_index(drop=True, inplace=True)
    n_samples = len(labels_df)
    log.info("Total valid samples: %d", n_samples)

    # ── Step 2: Scaler ─────────────────────────────────────────────────────
    label_matrix = labels_df[cfg["label_cols"]].values.astype(np.float32)
    scaler       = StandardScaler()
    label_scaled = scaler.fit_transform(label_matrix).astype(np.float32)

    scaler_path = os.path.join(cfg["output_dir"], "label_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Fusion label scaler → %s", scaler_path)

    # ── Step 3: Split indices ──────────────────────────────────────────────
    idx = np.arange(n_samples)
    tr_i, te_i = train_test_split(idx, test_size=cfg["test_frac"],
                                  random_state=cfg["seed"])
    tr_i, va_i = train_test_split(
        tr_i,
        test_size=cfg["val_frac"] / (1.0 - cfg["test_frac"]),
        random_state=cfg["seed"],
    )
    log.info("Split → train: %d | val: %d | test: %d",
             len(tr_i), len(va_i), len(te_i))

    # ── Step 4: Build datasets ─────────────────────────────────────────────
    custom_objs = _get_custom_objects()
    last_ckpt   = os.path.join(cfg["ckpt_dir"], "last_model.keras")
    best_ckpt   = os.path.join(cfg["ckpt_dir"], "best_model.keras")

    if cfg["mode"] == "offline":
        log.info("Loading pre-computed embeddings (offline mode) …")
        fused_emb, _, sample_ids = load_offline_embeddings(cfg, labels_df)

        train_ds = build_offline_dataset(fused_emb[tr_i], label_scaled[tr_i],
                                          cfg, shuffle=True)
        val_ds   = build_offline_dataset(fused_emb[va_i], label_scaled[va_i],
                                          cfg, shuffle=False)
        test_ds  = build_offline_dataset(fused_emb[te_i], label_scaled[te_i],
                                          cfg, shuffle=False)
        all_ds   = build_offline_dataset(fused_emb, label_scaled,
                                          cfg, shuffle=False)

        state = _load_state(cfg["ckpt_dir"])
        done  = state["completed_epochs"]
        total = cfg["warmup_epochs"] + cfg["finetune_epochs"]

        # Build or load the fusion head
        if done > 0 and os.path.exists(last_ckpt):
            log.info("Resuming from checkpoint (epoch %d) …", done)
            model = keras.models.load_model(last_ckpt, custom_objects=custom_objs)
            compile_model(model, cfg["lr_warmup"], cfg)
        else:
            model = build_fusion_head(cfg)
            compile_model(model, cfg["lr_warmup"], cfg)
            model.summary(print_fn=log.info)

        history_warmup = None
        if done < cfg["warmup_epochs"]:
            log.info("─" * 60)
            log.info("PHASE 1 – Warm-up  (lr=%.0e)", cfg["lr_warmup"])
            log.info("─" * 60)
            state["phase"] = "warmup"
            history_warmup = model.fit(
                train_ds, validation_data=val_ds,
                epochs=cfg["warmup_epochs"],
                initial_epoch=done,
                callbacks=get_enhanced_callbacks(
                    cfg, state, run_id, logger=log),
                verbose=1,
            )
            done = state["completed_epochs"]
            model.save(os.path.join(cfg["ckpt_dir"], "warmup_final.keras"))

        if os.path.exists(best_ckpt):
            model = keras.models.load_model(best_ckpt, custom_objects=custom_objs)

        history_finetune = None
        if done < total:
            log.info("─" * 60)
            log.info("PHASE 2 – Fine-tune  (cosine decay, lr=%.0e)",
                     cfg["lr_finetune"])
            log.info("─" * 60)
            if done > cfg["warmup_epochs"] and os.path.exists(last_ckpt):
                model = keras.models.load_model(last_ckpt,
                                                custom_objects=custom_objs)
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
                train_ds, validation_data=val_ds,
                epochs=total,
                initial_epoch=done,
                callbacks=get_enhanced_callbacks(
                    cfg, state, run_id, lr_scheduler=lr_scheduler_fn,
                    logger=log),
                verbose=1,
            )

    else:
        # ── ONLINE MODE ────────────────────────────────────────────────────
        log.info("Loading branch extractors for online mode …")

        resnet_ext    = keras.models.load_model(
            cfg["resnet_extractor"], custom_objects=custom_objs)
        mobilenet_ext = keras.models.load_model(
            cfg["mobilenet_extractor"], custom_objects=custom_objs)
        cgcnn_ext     = keras.models.load_model(
            cfg["cgcnn_extractor"], custom_objects=custom_objs)

        # Freeze by default; unfreeze top layers after warmup
        for ext in (resnet_ext, mobilenet_ext, cgcnn_ext):
            ext.trainable = False

        # Pair with image / graph files
        labels_df = pair_microstructure_images(labels_df, cfg["micro_dir"])
        if "pole_fig_path" not in labels_df.columns:
            from data_utils import build_pole_figure_cache
            labels_df = build_pole_figure_cache(
                labels_df, cfg["pole_fig_dir"])
        cgcnn_cfg = {k: cfg[k] for k in
                     ("max_nodes", "max_edges", "node_in_dim", "edge_in_dim",
                      "min_grain_voxels", "max_voxels") if k in cfg}
        cgcnn_cfg.setdefault("min_grain_voxels", 5)
        cgcnn_cfg.setdefault("max_voxels", 500_000)
        labels_df = build_graph_cache(labels_df, {**cfg, **cgcnn_cfg})
        labels_df.dropna(subset=["micro_img_path", "pole_fig_path",
                                  "graph_path"], inplace=True)
        labels_df.reset_index(drop=True, inplace=True)

        micro_paths  = labels_df["micro_img_path"].tolist()
        pole_paths   = labels_df["pole_fig_path"].tolist()
        graph_paths  = labels_df["graph_path"].tolist()
        label_scaled = scaler.transform(
            labels_df[cfg["label_cols"]].values.astype(np.float32)
        )
        idx  = np.arange(len(labels_df))
        tr_i, te_i = train_test_split(idx, test_size=cfg["test_frac"],
                                      random_state=cfg["seed"])
        tr_i, va_i = train_test_split(
            tr_i,
            test_size=cfg["val_frac"] / (1.0 - cfg["test_frac"]),
            random_state=cfg["seed"],
        )

        def sel(lst, ii): return [lst[i] for i in ii]

        train_ds = build_online_dataset(
            sel(micro_paths, tr_i), sel(pole_paths, tr_i),
            sel(graph_paths, tr_i), label_scaled[tr_i], cfg, shuffle=True)
        val_ds   = build_online_dataset(
            sel(micro_paths, va_i), sel(pole_paths, va_i),
            sel(graph_paths, va_i), label_scaled[va_i], cfg, shuffle=False)
        test_ds  = build_online_dataset(
            sel(micro_paths, te_i), sel(pole_paths, te_i),
            sel(graph_paths, te_i), label_scaled[te_i], cfg, shuffle=False)
        all_ds   = build_online_dataset(
            micro_paths, pole_paths, graph_paths,
            label_scaled, cfg, shuffle=False)

        state = _load_state(cfg["ckpt_dir"])
        done  = state["completed_epochs"]
        total = cfg["warmup_epochs"] + cfg["finetune_epochs"]

        if done > 0 and os.path.exists(last_ckpt):
            model = keras.models.load_model(last_ckpt, custom_objects=custom_objs)
            compile_model(model, cfg["lr_warmup"], cfg)
        else:
            model = build_online_model(cfg, resnet_ext, mobilenet_ext, cgcnn_ext)
            compile_model(model, cfg["lr_warmup"], cfg)
            model.summary(expand_nested=False, print_fn=log.info)

        history_warmup = None
        if done < cfg["warmup_epochs"]:
            log.info("─" * 60)
            log.info("PHASE 1 – Online warm-up  (branches frozen, lr=%.0e)",
                     cfg["lr_warmup"])
            log.info("─" * 60)
            state["phase"] = "warmup"
            history_warmup = model.fit(
                train_ds, validation_data=val_ds,
                epochs=cfg["warmup_epochs"],
                initial_epoch=done,
                callbacks=get_enhanced_callbacks(
                    cfg, state, run_id, logger=log),
                verbose=1,
            )
            done = state["completed_epochs"]

        if os.path.exists(best_ckpt):
            model = keras.models.load_model(best_ckpt, custom_objects=custom_objs)

        history_finetune = None
        if done < total:
            log.info("─" * 60)
            log.info("PHASE 2 – Online fine-tune  (top %d%% branches unfrozen)",
                     int(cfg["online_unfreeze_pct"] * 100))
            log.info("─" * 60)
            # Unfreeze top fraction of each branch extractor
            for ext_name in ("resnet50_extractor", "mobilenet_extractor",
                             "cgcnn_embedding_extractor"):
                try:
                    sub = model.get_layer(ext_name)
                    n   = len(sub.layers)
                    cut = int(n * (1.0 - cfg["online_unfreeze_pct"]))
                    for layer in sub.layers[:cut]:
                        layer.trainable = False
                    for layer in sub.layers[cut:]:
                        layer.trainable = True
                    log.info("Unfroze top %d/%d layers of %s", n - cut, n, ext_name)
                except ValueError:
                    pass

            steps_per_epoch = max(1, len(tr_i) // cfg["batch_size"])
            # Use a mixed LR: lower for branch layers, higher for fusion head
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=cfg["lr_finetune"],
                decay_steps=steps_per_epoch * cfg["finetune_epochs"],
                alpha=1e-7,
            )
            model.compile(
                optimizer=_get_optimizer(lr_schedule, cfg),
                loss=MultiOutputHuber(delta=1.0),
                metrics=["mae"],
            )
            state["phase"] = "finetune"
            history_finetune = model.fit(
                train_ds, validation_data=val_ds,
                epochs=total,
                initial_epoch=done,
                callbacks=get_enhanced_callbacks(
                    cfg, state, run_id, logger=log),
                verbose=1,
            )

    # ── Step 5: Load best model ────────────────────────────────────────────
    if os.path.exists(best_ckpt):
        model = keras.models.load_model(best_ckpt, custom_objects=custom_objs)

    # ── Step 6: Evaluate ───────────────────────────────────────────────────
    log.info("Evaluating on held-out test set …")
    evaluate(model, test_ds, cfg["label_cols"], scaler, cfg)

    # ── Step 7: Plots ──────────────────────────────────────────────────────
    plot_history(
        [h for h in [history_warmup, history_finetune] if h is not None],
        [l for l, h in [("Warm-up",  history_warmup),
                         ("Fine-tune", history_finetune)] if h is not None],
        cfg,
    )

    # ── Step 8: Save final model ───────────────────────────────────────────
    final_path = os.path.join(cfg["output_dir"], "best_model.keras")
    model.save(final_path)
    log.info("Fusion model → %s", final_path)

    # ── Git experiment tag (opt-in) ────────────────────────────────────────
    if cfg.get("git_tag_runs", False):
        tag_experiment("fusion", run_id, logger=log)

    log.info("=" * 60)
    log.info("  Fusion Layer training complete.")
    log.info("  Run ID   → %s", run_id)
    log.info("  Model    → %s", final_path)
    log.info("  Metrics  → %s/metrics.json", cfg["output_dir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
