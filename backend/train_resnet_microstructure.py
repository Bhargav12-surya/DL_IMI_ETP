"""
train_resnet_microstructure.py
═══════════════════════════════════════════════════════════════════════════════
Production-ready ResNet50 training pipeline for EBSD microstructure images.

Dataset   : Dryad doi:10.5061/dryad.83bk3j9sj
Input     : 2D EBSD IPF / grain-map images  (2D_EBSD_Dataset.zip)
Labels    : auto-computed from h5 orientation files
            (3D_EBSD_Dataset_Slices1to176_h5.zip)
Output    : trained model, checkpoints, CSV logs, evaluation plots,
            256-dim embedding extractor (for future fusion)

USAGE
─────
    python train_resnet_microstructure.py

CHECKPOINT RESUME
─────────────────
If training is interrupted (OOM, crash, etc.) just re-run the script.
It detects the last saved checkpoint and resumes from that epoch automatically.

DIRECTORY LAYOUT (set in CONFIG below)
───────────────────────────────────────
  data/
  ├── 2D_EBSD/          ← extracted from 2D_EBSD_Dataset.zip
  ├── h5_slices/        ← extracted from 3D_EBSD_Dataset_Slices*.zip
  └── labels.csv        ← auto-generated on first run

  checkpoints/resnet/
  ├── best_model.keras  ← saved every time val_loss improves
  ├── last_model.keras  ← overwritten every epoch (for resume)
  └── training_state.json

  logs/resnet/
  ├── training_log.csv
  └── tensorboard/

  outputs/resnet/
  ├── eval_scatter.png
  ├── training_history.png
  ├── metrics.json
  └── embeddings.npy
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
from tensorflow.keras.applications import ResNet50

# ── local ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    generate_labels_csv,
    pair_microstructure_images,
    validate_dataset,
    build_pole_figure_cache,
)
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
# CONFIGURATION  ← edit these paths and hyperparameters
# ═════════════════════════════════════════════════════════════════════════════
CONFIG: Dict = {
    # ── data paths ────────────────────────────────────────────────────────────
    "micro_dir":        "data/2D_EBSD",       # extracted 2D_EBSD_Dataset.zip
    "h5_dir":           "data/h5_slices",     # extracted h5 slice zips
    "labels_csv":       "data/labels.csv",
    "ckpt_dir":         "checkpoints/resnet",
    "log_dir":          "logs/resnet",
    "output_dir":       "outputs/resnet",
    "tb_dir":           "logs/resnet/tensorboard",

    # ── label targets (columns from labels.csv) ───────────────────────────────
    "label_cols": [
        "texture_index",
        "avg_misorientation",
    ],

    # ── image ─────────────────────────────────────────────────────────────────
    "img_size":         (224, 224),           # (H, W) fed to ResNet50
    "img_channels":     3,

    # ── architecture ──────────────────────────────────────────────────────────
    "embed_dim":        256,                  # final embedding dimension
    "dropout_rate":     0.40,
    "l2_reg":           1e-4,
    "se_ratio":         8,                    # squeeze-excitation reduction

    # ── training ──────────────────────────────────────────────────────────────
    "batch_size":       32,
    "warmup_epochs":    10,                   # heads-only (backbone frozen)
    "finetune_epochs":  70,                   # backbone top-50% unfrozen
    "lr_warmup":        1e-3,
    "lr_finetune":      5e-5,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,
    "unfreeze_from":    0.50,                 # fraction of backbone to unfreeze

    # ── gradient accumulation ─────────────────────────────────────────────────
    "grad_accum_steps": 4,                    # effective batch = 32 × 4 = 128

    # ── checkpoints ───────────────────────────────────────────────────────────
    "ckpt_every_n_epochs": 10,                # save epoch-N snapshot every 10
    "max_kept_ckpts":     5,                  # prune oldest snapshots past 5

    # ── version control ───────────────────────────────────────────────────────
    "git_tag_runs":     False,                # create git tag after training

    # ── augmentation ──────────────────────────────────────────────────────────
    "use_mixup":        True,
    "mixup_alpha":      0.2,
    "aug_brightness":   0.15,
    "aug_contrast":     0.20,
    "aug_saturation":   0.20,
    "aug_hue":          0.05,

    # ── train / val / test splits ─────────────────────────────────────────────
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
# 0.  ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_environment(cfg: Dict) -> Dict:
    """Configure GPU, seeds, directories, and mixed precision."""
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    for d in (cfg["ckpt_dir"], cfg["log_dir"],
              cfg["output_dir"], cfg["tb_dir"]):
        os.makedirs(d, exist_ok=True)

    # ── CUDA diagnostics & device setup (replaces manual GPU config) ──────
    device_diag = setup_cuda(cfg, logger=log)
    return device_diag


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _load_and_preprocess_image(path: tf.Tensor,
                                img_size: Tuple[int, int],
                                augment: bool) -> tf.Tensor:
    """
    Load an image from disk, resize, and apply optional augmentation.
    All ops are TF-native (no py_function) for graph compatibility.
    """
    raw = tf.io.read_file(path)

    # decode_image handles PNG / TIFF / BMP / JPEG automatically
    img = tf.image.decode_image(raw, channels=3, expand_animations=False,
                                 dtype=tf.uint8)
    img.set_shape([None, None, 3])  # required after decode_image
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, img_size, method="bilinear")

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        # Random 90° rotation
        k   = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.80, 1.20)
        img = tf.image.random_saturation(img, 0.80, 1.20)
        img = tf.image.random_hue(img, 0.05)

    img = tf.clip_by_value(img, 0.0, 1.0)
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return img


def _make_map_fn(cfg: Dict, augment: bool):
    img_size = tuple(cfg["img_size"])

    def _fn(path: tf.Tensor, label: tf.Tensor):
        img = _load_and_preprocess_image(path, img_size, augment)
        return img, label

    return _fn


def _apply_mixup(images: tf.Tensor, labels: tf.Tensor,
                  alpha: float = 0.2) -> Tuple[tf.Tensor, tf.Tensor]:
    """In-batch MixUp augmentation using Beta(alpha, alpha) distribution."""
    batch_size = tf.shape(images)[0]
    # Beta(a, a) via two independent Gamma samples: lam = g1 / (g1 + g2)
    g1 = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    g2 = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    lam = g1 / (g1 + g2 + 1e-9)
    lam = tf.maximum(lam, 1.0 - lam)

    indices = tf.random.shuffle(tf.range(batch_size))

    lam_img = tf.reshape(lam, [-1, 1, 1, 1])
    mixed_images = lam_img * images + (1.0 - lam_img) * tf.gather(images, indices)

    lam_lab = tf.reshape(lam, [-1, 1])
    mixed_labels = lam_lab * labels + (1.0 - lam_lab) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def build_dataset(paths: List[str],
                  labels: np.ndarray,
                  cfg: Dict,
                  augment: bool = False,
                  shuffle: bool = True) -> tf.data.Dataset:
    """Build a tf.data.Dataset from image paths and float labels."""
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(
        labels.astype(np.float32))

    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 2000),
            seed=cfg["seed"],
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_make_map_fn(cfg, augment),
                num_parallel_calls=cfg["num_parallel"])

    ds = ds.batch(cfg["batch_size"], drop_remainder=False)

    if augment and cfg["use_mixup"]:
        ds = ds.map(
            lambda imgs, labs: _apply_mixup(imgs, labs, cfg["mixup_alpha"]),
            num_parallel_calls=cfg["num_parallel"],
        )

    return ds.prefetch(cfg["prefetch"])


# ═════════════════════════════════════════════════════════════════════════════
# 2.  MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════════════

class SqueezeExcitation(layers.Layer):
    """Channel-wise Squeeze-and-Excitation for 1-D feature vectors."""

    def __init__(self, units: int, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.ratio = ratio
        self.fc1 = layers.Dense(max(units // ratio, 1), activation="relu",
                                 use_bias=True)
        self.fc2 = layers.Dense(units, activation="sigmoid", use_bias=True)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:  # noqa: ARG002
        attn = self.fc1(x)
        attn = self.fc2(attn)
        return x * attn

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"units": self.units, "ratio": self.ratio})
        return cfg


def _get_resnet_stage_outputs(backbone: keras.Model):
    """
    Extract multi-scale feature outputs from ResNet50.
    Returns (s3, s4, s5) tensors with shapes:
        s3 : (B, 28, 28,  512)  – end of conv3
        s4 : (B, 14, 14, 1024)  – end of conv4
        s5 : (B,  7,  7, 2048)  – end of conv5
    Layer names are stable across TF 2.x for ResNet50.
    """
    name_map = {
        "s3": ["conv3_block4_out"],
        "s4": ["conv4_block6_out"],
        "s5": ["conv5_block3_out"],
    }
    outputs = {}
    for key, names in name_map.items():
        for name in names:
            try:
                outputs[key] = backbone.get_layer(name).output
                break
            except ValueError:
                continue
        if key not in outputs:
            raise ValueError(
                f"Could not find ResNet50 layer for stage '{key}'. "
                "Verify your TensorFlow version supports ResNet50."
            )
    return outputs["s3"], outputs["s4"], outputs["s5"]


def build_resnet_model(cfg: Dict,
                        trainable_backbone: bool = False) -> keras.Model:
    """
    Build the Microstructure ResNet50 model.

    Architecture
    ────────────
    Input (224×224×3)
      └─► ResNet50 (ImageNet pretrained)
            ├─ Stage-3 features (28×28×512)  → GAP → Dense(128) → GELU
            ├─ Stage-4 features (14×14×1024) → GAP → Dense(256) → GELU
            └─ Stage-5 features (7×7×2048)   → GAP → Dense(256) → GELU
            └─ Concat  [640-dim]
               → Dense(512, GELU) → BN → Dropout
               → SE-Block(512)
               → Dense(256, GELU) → BN → Dropout       ← embedding
               → Dense(n_outputs)                       ← prediction head
    """
    reg = regularizers.l2(cfg["l2_reg"])
    inp = keras.Input(shape=(*cfg["img_size"], cfg["img_channels"]),
                      name="micro_input")

    backbone_base = ResNet50(
        include_top=False,
        weights="imagenet",
        pooling=None,
    )

    # Create a multi-output model to extract intermediate stages
    name_map = {
        "s3": ["conv3_block4_out"],
        "s4": ["conv4_block6_out"],
        "s5": ["conv5_block3_out"],
    }
    stage_outputs = []
    for stage in ["s3", "s4", "s5"]:
        found = False
        for name in name_map[stage]:
            try:
                stage_outputs.append(backbone_base.get_layer(name).output)
                found = True
                break
            except ValueError:
                continue
        if not found:
            raise ValueError(f"Could not find ResNet layers for stage {stage}")

    backbone = keras.Model(
        inputs=backbone_base.input,
        outputs=stage_outputs,
        name="resnet50"
    )
    backbone.trainable = trainable_backbone

    # Connect backbone to input and get multi-scale features
    s3, s4, s5 = backbone(inp)

    def gap_project(feat, units, name):
        x = layers.GlobalAveragePooling2D(name=f"gap_{name}")(feat)
        x = layers.Dense(units, kernel_regularizer=reg,
                          name=f"proj_{name}")(x)
        x = layers.BatchNormalization(name=f"bn_{name}")(x)
        x = layers.Activation("gelu", name=f"act_{name}")(x)
        return x

    f3 = gap_project(s3, 128, "s3")
    f4 = gap_project(s4, 256, "s4")
    f5 = gap_project(s5, 256, "s5")

    x = layers.Concatenate(name="scale_concat")([f3, f4, f5])  # 640-dim

    x = layers.Dense(512, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("gelu", name="act1")(x)
    x = layers.Dropout(cfg["dropout_rate"], name="drop1")(x)

    x = SqueezeExcitation(512, ratio=cfg["se_ratio"], name="se")(x)

    x = layers.Dense(cfg["embed_dim"], kernel_regularizer=reg,
                      name="embedding")(x)
    x = layers.BatchNormalization(name="bn_embed")(x)
    x = layers.Activation("gelu", name="act_embed")(x)
    x = layers.Dropout(cfg["dropout_rate"] * 0.5, name="drop2")(x)

    # Output head – cast to float32 for mixed-precision stability
    n_out = len(cfg["label_cols"])
    out = layers.Dense(n_out, dtype="float32", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out,
                         name="ResNet50_Microstructure")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3.  LOSS, METRICS, COMPILE
# ═════════════════════════════════════════════════════════════════════════════

class MultiOutputHuber(keras.losses.Loss):
    """Huber loss averaged over all output dimensions."""

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self._huber = keras.losses.Huber(delta=delta, reduction="sum_over_batch_size")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._huber(y_true, y_pred)

    def get_config(self) -> dict:
        return {**super().get_config(), "delta": self.delta}


def _get_optimizer(lr: float, cfg: Dict) -> keras.optimizers.Optimizer:
    """Return AdamW with correct import for TF ≥ 2.11."""
    # TF 2.11–2.13 : tf.keras.optimizers.experimental.AdamW
    # TF 2.14+     : tf.keras.optimizers.AdamW
    for cls_path in (
        "tensorflow.keras.optimizers.AdamW",
        "tensorflow.keras.optimizers.experimental.AdamW",
    ):
        try:
            parts = cls_path.split(".")
            obj = __import__(parts[0])
            for p in parts[1:]:
                obj = getattr(obj, p)
            return obj(
                learning_rate=lr,
                weight_decay=cfg["weight_decay"],
                clipnorm=cfg["grad_clip"],
            )
        except (AttributeError, ImportError):
            continue

    # Final fallback: Adam (no weight decay)
    log.warning("AdamW not available – using Adam without weight decay.")
    return keras.optimizers.Adam(learning_rate=lr, clipnorm=cfg["grad_clip"])


def compile_model(model: keras.Model, lr: float, cfg: Dict) -> keras.Model:
    model.compile(
        optimizer=_get_optimizer(lr, cfg),
        loss=MultiOutputHuber(delta=1.0),
        metrics=["mae"],
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4.  CHECKPOINT & LOG CALLBACK  (via shared training_utils)
# ═════════════════════════════════════════════════════════════════════════════

def _load_state(ckpt_dir: str) -> Dict:
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        log.info("Resuming from state: %s", state)
        return state
    return {
        "completed_epochs": 0,
        "best_val_loss":    float("inf"),
        "phase":            "warmup",
    }


def _save_state(state: Dict, ckpt_dir: str) -> None:
    path = os.path.join(ckpt_dir, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model: keras.Model,
              test_ds: tf.data.Dataset,
              label_cols: List[str],
              scaler: StandardScaler,
              cfg: Dict) -> Dict:
    """Run inference on the test set and compute per-label metrics."""
    y_pred_list, y_true_list = [], []

    for imgs, y in test_ds:
        preds = model(imgs, training=False)
        y_pred_list.append(preds.numpy())
        y_true_list.append(y.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0).astype(np.float32)
    y_true = np.concatenate(y_true_list, axis=0).astype(np.float32)

    # Inverse-transform to original scale
    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)

    metrics: Dict[str, Dict] = {}
    header = f"{'Property':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}"
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for i, col in enumerate(label_cols):
        yt = y_true_orig[:, i]
        yp = y_pred_orig[:, i]
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp))
        mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
        metrics[col] = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
        print(f"  {col:<23} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f} {mape:>7.2f}%")

    print(sep + "\n")

    # ── scatter plots ─────────────────────────────────────────────────────────
    n_cols = len(label_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, label_cols)):
        yt = y_true_orig[:, i]
        yp = y_pred_orig[:, i]
        ax.scatter(yt, yp, alpha=0.35, s=12, color="#2563EB")
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
        ax.set_xlabel(f"True {col}")
        ax.set_ylabel(f"Predicted {col}")
        ax.set_title(f"{col}\nMAE={metrics[col]['mae']:.3f}  "
                     f"R²={metrics[col]['r2']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("ResNet50 – Microstructure Branch  |  Test Set", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(cfg["output_dir"], "eval_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Scatter plots → %s", fig_path)

    # Save metrics JSON
    metrics_path = os.path.join(cfg["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics JSON → %s", metrics_path)

    return metrics


def plot_history(histories: List, labels: List[str], cfg: Dict) -> None:
    """Plot and save training / validation loss curves."""
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4))
    if len(histories) == 1:
        axes = [axes]

    for ax, hist, label in zip(axes, histories, labels):
        if hist is None:
            continue
        ax.plot(hist.history["loss"],     label="Train", linewidth=1.5)
        ax.plot(hist.history["val_loss"], label="Val",   linewidth=1.5)
        ax.set_title(f"{label}  –  Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("ResNet50 Microstructure – Training History")
    plt.tight_layout()
    path = os.path.join(cfg["output_dir"], "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Training history → %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  EMBEDDING EXTRACTION  (for future fusion layer)
# ═════════════════════════════════════════════════════════════════════════════

def extract_and_save_embeddings(model: keras.Model,
                                 all_ds: tf.data.Dataset,
                                 cfg: Dict) -> np.ndarray:
    """
    Extract 256-dim embeddings from the 'embedding' Dense layer.
    Saved to outputs/embeddings.npy for use in the Fusion pipeline.
    """
    try:
        embed_layer_output = model.get_layer("embedding").output
    except ValueError:
        log.warning("Layer 'embedding' not found – using final layer output.")
        embed_layer_output = model.layers[-2].output

    embed_model = keras.Model(
        inputs=model.input,
        outputs=embed_layer_output,
        name="ResNet_EmbeddingExtractor",
    )
    embed_model_path = os.path.join(cfg["output_dir"], "embedding_extractor.keras")
    embed_model.save(embed_model_path)
    log.info("Embedding extractor saved → %s", embed_model_path)

    embs = []
    for imgs, _ in all_ds:
        embs.append(embed_model(imgs, training=False).numpy())
    embeddings = np.concatenate(embs, axis=0)

    npy_path = os.path.join(cfg["output_dir"], "embeddings.npy")
    np.save(npy_path, embeddings)
    log.info("Embeddings %s → %s", embeddings.shape, npy_path)
    return embeddings


# ═════════════════════════════════════════════════════════════════════════════
# 7.  MAIN  TRAINING  PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg = CONFIG
    device_diag = setup_environment(cfg)

    # ── Structured file logger & run ID ────────────────────────────────────
    run_id, _ = setup_train_logger("resnet", cfg["log_dir"], logger=log)

    # ── Git version info ──────────────────────────────────────────────────
    git_info = get_git_info()
    log.info("Git commit : %s  (dirty=%s)", git_info["commit_hash"][:8],
             git_info["is_dirty"])

    log.info("=" * 60)
    log.info("  ResNet50 – Microstructure Branch  Training")
    log.info("  Run ID   : %s", run_id)
    log.info("  Grad Accum Steps: %d  (effective batch = %d)",
             cfg["grad_accum_steps"],
             cfg["batch_size"] * cfg["grad_accum_steps"])
    log.info("=" * 60)

    # ── Step 1: Labels ─────────────────────────────────────────────────────
    labels_df = generate_labels_csv(
        h5_dir       = cfg["h5_dir"],
        csv_path     = cfg["labels_csv"],
        voxel_size_um= cfg["voxel_size_um"],
        compute_misorientation=True,
    )

    # ── Step 2: Pair with 2D microstructure images ─────────────────────────
    labels_df = pair_microstructure_images(labels_df, cfg["micro_dir"])

    if len(labels_df) < 20:
        log.warning("Insufficient matched 2D images (%d). Falling back to auto-generated Pole Figures.", len(labels_df))
        # Reuse labels_df but add generated pole figures as the image column
        labels_df_full = generate_labels_csv(cfg["h5_dir"], cfg["labels_csv"])
        labels_df = build_pole_figure_cache(
            labels_df_full,
            cache_dir = os.path.join(os.path.dirname(cfg["labels_csv"]), "pole_figures"),
        )
        labels_df.rename(columns={"pole_fig_path": "micro_img_path"}, inplace=True)
        log.info("Using %d generated pole figures as microstructure input.", len(labels_df))

    # ── Step 3: Validate ───────────────────────────────────────────────────
    labels_df = validate_dataset(
        labels_df,
        img_col     = "micro_img_path",
        label_cols  = cfg["label_cols"],
        min_samples = 20,
    )

    # ── Step 4: Normalise labels ───────────────────────────────────────────
    label_matrix = labels_df[cfg["label_cols"]].values.astype(np.float32)
    scaler = StandardScaler()
    label_scaled = scaler.fit_transform(label_matrix).astype(np.float32)

    # Save scaler for inference
    scaler_path = os.path.join(cfg["output_dir"], "label_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Label scaler → %s", scaler_path)

    # ── Step 5: Split ──────────────────────────────────────────────────────
    all_paths = labels_df["micro_img_path"].tolist()
    idx  = np.arange(len(labels_df))
    tr_i, te_i = train_test_split(idx, test_size=cfg["test_frac"],
                                   random_state=cfg["seed"])
    tr_i, va_i = train_test_split(
        tr_i,
        test_size=cfg["val_frac"] / (1.0 - cfg["test_frac"]),
        random_state=cfg["seed"],
    )
    log.info("Split → train: %d | val: %d | test: %d",
             len(tr_i), len(va_i), len(te_i))

    def sel(lst, idx_):
        return [lst[i] for i in idx_]

    train_ds = build_dataset(sel(all_paths, tr_i), label_scaled[tr_i],
                              cfg, augment=True,  shuffle=True)
    val_ds   = build_dataset(sel(all_paths, va_i), label_scaled[va_i],
                              cfg, augment=False, shuffle=False)
    test_ds  = build_dataset(sel(all_paths, te_i), label_scaled[te_i],
                              cfg, augment=False, shuffle=False)
    all_ds   = build_dataset(all_paths, label_scaled,
                              cfg, augment=False, shuffle=False)

    # ── Step 6: Load training state (checkpoint resume) ────────────────────
    state = _load_state(cfg["ckpt_dir"])
    done  = state["completed_epochs"]

    total_epochs = cfg["warmup_epochs"] + cfg["finetune_epochs"]
    if done >= total_epochs:
        log.info("Training already complete (%d/%d epochs done).", done, total_epochs)
    else:
        log.info("Completed epochs so far: %d / %d", done, total_epochs)

    # ── Step 7: Warmup phase ───────────────────────────────────────────────
    history_warmup = None
    last_ckpt = os.path.join(cfg["ckpt_dir"], "last_model.keras")

    if done < cfg["warmup_epochs"]:
        log.info("─" * 60)
        log.info("PHASE 1 – Warm-up  (backbone frozen, lr=%.0e)", cfg["lr_warmup"])
        log.info("─" * 60)

        if done > 0 and os.path.exists(last_ckpt):
            log.info("Resuming warm-up from epoch %d …", done)
            model = keras.models.load_model(
                last_ckpt,
                custom_objects={"SqueezeExcitation": SqueezeExcitation,
                                "MultiOutputHuber": MultiOutputHuber},
            )
            # Recompile with current LR (optimizer state restored from model)
            compile_model(model, cfg["lr_warmup"], cfg)
        else:
            model = build_resnet_model(cfg, trainable_backbone=False)
            compile_model(model, cfg["lr_warmup"], cfg)

        # ── Gradient accumulation wrapper ──
        model = build_gradient_accumulation_model(
            model, cfg.get("grad_accum_steps", 1))

        model.summary(expand_nested=False, print_fn=log.info)
        state["phase"] = "warmup"

        history_warmup = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = cfg["warmup_epochs"],
            initial_epoch   = done,
            callbacks       = get_enhanced_callbacks(
                cfg, state, run_id, logger=log),
            verbose         = 1,
        )
        done = state["completed_epochs"]

        # Always save a named warmup checkpoint for reference
        warmup_path = os.path.join(cfg["ckpt_dir"], "warmup_final.keras")
        model.save(warmup_path)
        log.info("Warmup final checkpoint → %s", warmup_path)

    elif done == cfg["warmup_epochs"]:
        # Warmup done, fine-tune not started yet → load best warmup weights
        best_path = os.path.join(cfg["ckpt_dir"], "best_model.keras")
        if os.path.exists(best_path):
            model = keras.models.load_model(
                best_path,
                custom_objects={"SqueezeExcitation": SqueezeExcitation,
                                "MultiOutputHuber": MultiOutputHuber},
            )
            log.info("Loaded best warmup model for fine-tuning.")
        else:
            model = build_resnet_model(cfg, trainable_backbone=False)
            compile_model(model, cfg["lr_warmup"], cfg)
    else:
        # Fine-tune was already in progress; model will be loaded below
        pass

    # ── Step 8: Fine-tune phase ────────────────────────────────────────────
    history_finetune = None
    if done < total_epochs:
        log.info("─" * 60)
        log.info("PHASE 2 – Fine-tune  (backbone top-%.0f%% unfrozen, lr=%.0e)",
                 (1.0 - cfg["unfreeze_from"]) * 100, cfg["lr_finetune"])
        log.info("─" * 60)

        if done > cfg["warmup_epochs"] and os.path.exists(last_ckpt):
            log.info("Resuming fine-tune from epoch %d …", done)
            model = keras.models.load_model(
                last_ckpt,
                custom_objects={"SqueezeExcitation": SqueezeExcitation,
                                "MultiOutputHuber": MultiOutputHuber},
            )
        elif done == cfg["warmup_epochs"]:
            # model already loaded above; now unfreeze backbone layers
            pass

        # Unfreeze top fraction of backbone
        backbone = model.get_layer("resnet50")
        n_layers  = len(backbone.layers)
        unfreeze_from_idx = int(n_layers * cfg["unfreeze_from"])
        for layer in backbone.layers[:unfreeze_from_idx]:
            layer.trainable = False
        for layer in backbone.layers[unfreeze_from_idx:]:
            layer.trainable = True

        trainable_count = sum(
            1 for l in backbone.layers if l.trainable
        )
        log.info("Unfrozen %d / %d backbone layers.", trainable_count, n_layers)

        state["phase"] = "finetune"

        # Cosine-decay schedule for fine-tuning via Callback (prevents Keras 3 TypeError)
        steps_per_epoch = max(1, len(tr_i) // cfg["batch_size"])
        lr_schedule_obj = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg["lr_finetune"],
            decay_steps=steps_per_epoch * cfg["finetune_epochs"],
            alpha=1e-7,
        )

        def lr_scheduler_fn(epoch, lr):  # noqa: ARG001
            # decay is relative to the start of fine-tuning
            relative_step = (epoch - cfg["warmup_epochs"]) * steps_per_epoch
            if relative_step < 0:
                return cfg["lr_finetune"]
            return float(lr_schedule_obj(relative_step))

        model.compile(
            optimizer=_get_optimizer(cfg["lr_finetune"], cfg),
            loss=MultiOutputHuber(delta=1.0),
            metrics=["mae"],
        )

        # ── Gradient accumulation wrapper ──
        model = build_gradient_accumulation_model(
            model, cfg.get("grad_accum_steps", 1))

        history_finetune = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = total_epochs,
            initial_epoch   = done,
            callbacks       = get_enhanced_callbacks(
                cfg, state, run_id, lr_scheduler=lr_scheduler_fn,
                logger=log),
            verbose         = 1,
        )

    # ── Step 9: Load best model for evaluation ─────────────────────────────
    best_path = os.path.join(cfg["ckpt_dir"], "best_model.keras")
    if os.path.exists(best_path):
        log.info("Loading best model for evaluation …")
        model = keras.models.load_model(
            best_path,
            custom_objects={"SqueezeExcitation": SqueezeExcitation,
                            "MultiOutputHuber": MultiOutputHuber},
        )

    # ── Step 10: Evaluate ──────────────────────────────────────────────────
    log.info("Evaluating on held-out test set …")
    metrics = evaluate(model, test_ds, cfg["label_cols"], scaler, cfg)

    # ── Step 11: Plots ─────────────────────────────────────────────────────
    plot_history(
        [h for h in [history_warmup, history_finetune] if h is not None],
        [l for l, h in [("Warm-up", history_warmup),
                         ("Fine-tune", history_finetune)] if h is not None],
        cfg,
    )

    # ── Step 12: Extract & save embeddings ────────────────────────────────
    log.info("Extracting embeddings for future Fusion Layer …")
    extract_and_save_embeddings(model, all_ds, cfg)

    # ── Git experiment tag (opt-in) ────────────────────────────────────────
    if cfg.get("git_tag_runs", False):
        tag_experiment("resnet", run_id, logger=log)

    log.info("=" * 60)
    log.info("  Training complete.")
    log.info("  Run ID      → %s", run_id)
    log.info("  Best model  → %s", best_path)
    log.info("  Metrics     → %s/metrics.json", cfg["output_dir"])
    log.info("  Embeddings  → %s/embeddings.npy", cfg["output_dir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
