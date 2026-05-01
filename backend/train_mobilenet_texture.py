"""
train_mobilenet_texture.py
═══════════════════════════════════════════════════════════════════════════════
Production-ready MobileNetV2 training pipeline for EBSD texture descriptors.

Dataset   : Dryad doi:10.5061/dryad.83bk3j9sj
Input     : 3-channel pole-figure images  {100} | {110} | {111}
            (auto-generated from h5 Euler-angle data by data_utils)
Labels    : auto-computed from h5 orientation files  (shared with ResNet)
Output    : trained model, checkpoints, CSV logs, evaluation plots,
            256-dim embedding extractor (for future fusion)

ARCHITECTURE
────────────
  Input  (224 × 224 × 3)
    └─► MobileNetV2  (ImageNet pretrained)
          ├─ Stage-3 features  (28×28 × 192)  → GAP → Dense(96)  → GELU
          ├─ Stage-4 features  (14×14 × 576)  → GAP → Dense(192) → GELU
          └─ Stage-5 features  ( 7× 7 ×1280)  → GAP → Dense(256) → GELU
          └─ Concat  [544-dim]
             → Dense(512, GELU) → BN → Dropout
             → SE-Block(512)
             → Dense(256, GELU) → BN → Dropout      ← embedding
             → Dense(n_outputs, float32)             ← prediction head

USAGE
─────
    python train_mobilenet_texture.py

CHECKPOINT RESUME
─────────────────
If training is interrupted (OOM, crash) re-run the same command.
The script detects last_model.keras and resumes from the saved epoch.

DIRECTORY LAYOUT
────────────────
  data/
  ├── h5_slices/          ← extracted from 3D_EBSD_Dataset_Slices*.zip
  ├── pole_figures/       ← auto-generated PNG cache (created on first run)
  └── labels.csv          ← shared with ResNet branch (auto-generated)

  checkpoints/mobilenet/
  ├── best_model.keras    ← saved whenever val_loss improves
  ├── last_model.keras    ← overwritten every epoch  (OOM-safe resume)
  └── training_state.json

  logs/mobilenet/
  ├── training_log.csv
  └── tensorboard/

  outputs/mobilenet/
  ├── eval_scatter.png
  ├── training_history.png
  ├── metrics.json
  ├── label_scaler.pkl
  ├── embedding_extractor.keras
  └── embeddings.npy

FILES TO DOWNLOAD FROM DRYAD
─────────────────────────────
  REQUIRED (minimum):
    • 3D_EBSD_Dataset_Slices1to176_h5.zip   (14.18 GB)  → data/h5_slices/
      (contains the .h5 Euler-angle files used to generate pole figures and labels)

  OPTIONAL (more data = better model):
    • 3D_EBSD_Dataset_Slices177to352_h5.zip (14.11 GB)  → data/h5_slices/
    • 3D_EBSD_Dataset_Slices353to528_h5.zip (14.16 GB)  → data/h5_slices/

  NOT NEEDED for this branch:
    • 2D_EBSD_Dataset.zip              (ResNet branch only)
    • 3D_EBSD_Dataset_Images.zip       (3D renders, optional reference)
    • 3D_EBSD_Dataset_Meshed_*.zip     (FEM meshes, not used)
    • 3D_EBSD_Dataset_Voxelized.zip    (voxel data, not used here)
    • HRDIC_Results_Full_Fields.zip    (DIC strain fields, not used)
    • 3D_EBSD_Dataset_Up2_2slices.zip  (upsampled slices, optional)
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
from tensorflow.keras.applications import MobileNetV2

# ── local ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import (
    build_pole_figure_cache,
    generate_labels_csv,
    validate_dataset,
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
# CONFIGURATION  ← edit paths and hyperparameters here
# ═════════════════════════════════════════════════════════════════════════════
CONFIG: Dict = {
    # ── data paths ────────────────────────────────────────────────────────────
    # h5 slice files (Euler angles & grain IDs)
    "h5_dir":           "data/h5_slices",
    # auto-generated pole-figure PNG cache
    "pole_fig_dir":     "data/pole_figures",
    # shared labels CSV (created once, reused by both branches)
    "labels_csv":       "data/labels.csv",
    # checkpoint, log, output directories
    "ckpt_dir":         "checkpoints/mobilenet",
    "log_dir":          "logs/mobilenet",
    "output_dir":       "outputs/mobilenet",
    "tb_dir":           "logs/mobilenet/tensorboard",

    # ── label targets ─────────────────────────────────────────────────────────
    "label_cols": [
        "grain_size",           # mean equiv-sphere-diameter  (µm)
        "texture_index",        # ODF second moment  J  (dimensionless)
        "avg_misorientation",   # mean intra-grain misorientation  (°)
    ],

    # ── image ─────────────────────────────────────────────────────────────────
    "img_size":         (224, 224),
    "img_channels":     3,                  # 3-channel RGB pole figure

    # ── pole-figure generation ────────────────────────────────────────────────
    "pole_fig_n_grid":  224,                # match img_size
    "pole_fig_subsample": 8000,             # orientations per sample

    # ── architecture ──────────────────────────────────────────────────────────
    "embed_dim":        256,                # final embedding dimension
    "dropout_rate":     0.35,
    "l2_reg":           1e-4,
    "se_ratio":         8,                  # squeeze-excitation reduction

    # ── training schedule ─────────────────────────────────────────────────────
    "batch_size":       32,
    "warmup_epochs":    10,                 # heads-only, backbone frozen
    "finetune_epochs":  60,                 # backbone top-50% unfrozen
    "lr_warmup":        1e-3,
    "lr_finetune":      2e-5,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,
    "unfreeze_from":    0.50,               # fraction of backbone to unfreeze

    # ── augmentation ──────────────────────────────────────────────────────────
    "use_mixup":        True,
    "mixup_alpha":      0.2,
    "aug_brightness":   0.12,
    "aug_contrast":     0.15,
    "aug_saturation":   0.15,
    "aug_hue":          0.04,

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
# 0.  ENVIRONMENT SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_environment(cfg: Dict) -> None:
    """Configure GPU memory growth, global seeds, directories, mixed precision."""
    tf.random.set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    for d in (cfg["ckpt_dir"], cfg["log_dir"],
              cfg["output_dir"], cfg["tb_dir"],
              cfg["pole_fig_dir"]):
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
# 1.  DATA PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _load_and_preprocess_image(path: tf.Tensor,
                                img_size: Tuple[int, int],
                                augment: bool) -> tf.Tensor:
    """
    Load a pole-figure PNG from disk, resize, and apply augmentation.
    Uses TF-native ops only (graph-mode compatible).
    """
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False,
                                 dtype=tf.uint8)
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, img_size, method="bilinear")

    if augment:
        # Spatial
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        k   = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        # Colour – mild for pole figures (avoid destroying physical features)
        img = tf.image.random_brightness(img, 0.12)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        img = tf.image.random_saturation(img, 0.85, 1.15)
        img = tf.image.random_hue(img, 0.04)

    img = tf.clip_by_value(img, 0.0, 1.0)
    # ImageNet normalisation (MobileNetV2 was trained this way)
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
    """In-batch MixUp for regression targets using Beta(alpha, alpha) distribution."""
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
    """Build a tf.data pipeline from pole-figure paths and float labels."""
    path_ds  = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))
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
    """
    Squeeze-and-Excitation block for 1-D feature vectors.
    Performs channel-wise recalibration via learned attention weights.
    """

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


# ── MobileNetV2 multi-scale layer names (verified for TF 2.x, 224×224 input)
# Stage-3 : 28×28 spatial resolution, 192 channels after expand
# Stage-4 : 14×14 spatial resolution, 576 channels after expand
# Stage-5 : 7×7 spatial resolution, 1280 channels (final conv)
_MOB_STAGE_LAYERS: Dict[str, List[str]] = {
    "s3": [
        "block_6_expand_relu",   # primary   (TF 2.4 – 2.15)
        "block_5_project_BN",    # fallback
        "block_4_project_BN",    # fallback-2
    ],
    "s4": [
        "block_13_expand_relu",  # primary
        "block_12_project_BN",   # fallback
        "block_10_project_BN",   # fallback-2
    ],
    "s5": [
        "out_relu",              # primary   (post-conv1 ReLU6, 7×7×1280)
        "Conv_1_bn",             # fallback
        "block_16_project_BN",   # fallback-2
    ],
}


def _get_mobilenet_stage_outputs(backbone: keras.Model):
    """
    Extract multi-scale intermediate tensors from MobileNetV2.

    Returns (s3, s4, s5) corresponding to 28×28, 14×14, 7×7 spatial scales.
    Tries each candidate name in order; raises ValueError if all fail.
    """
    outputs: Dict[str, tf.Tensor] = {}
    for key, candidates in _MOB_STAGE_LAYERS.items():
        for name in candidates:
            try:
                outputs[key] = backbone.get_layer(name).output
                log.info("  MobileNetV2 %s → layer '%s'", key, name)
                break
            except ValueError:
                continue
        if key not in outputs:
            raise ValueError(
                f"Could not locate MobileNetV2 layer for stage '{key}'. "
                f"Tried: {candidates}. "
                "If you are using a custom TF version, inspect "
                "[l.name for l in backbone.layers] and update "
                "_MOB_STAGE_LAYERS in this script."
            )
    return outputs["s3"], outputs["s4"], outputs["s5"]


def build_mobilenet_model(cfg: Dict,
                           trainable_backbone: bool = False) -> keras.Model:
    """
    Build the Texture MobileNetV2 model.

    Architecture Summary
    ────────────────────
    Input  (224 × 224 × 3)
      └─► MobileNetV2  (ImageNet pretrained)
            ├─ Stage-3 (28×28×192)  → GAP → Dense(96,  GELU) → BN
            ├─ Stage-4 (14×14×576)  → GAP → Dense(192, GELU) → BN
            └─ Stage-5 ( 7× 7×1280) → GAP → Dense(256, GELU) → BN
            └─ Concat  [544-dim]
               → Dense(512, GELU) → BN → Dropout(0.35)
               → SE-Block(512, r=8)
               → Dense(256, GELU) → BN → Dropout(0.17)    ← embedding
               → Dense(n_outputs, float32)                 ← predictions
    """
    reg = regularizers.l2(cfg["l2_reg"])
    inp = keras.Input(shape=(*cfg["img_size"], cfg["img_channels"]),
                      name="texture_input")

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inp,
        pooling=None,        # we do our own pooling
        alpha=1.0,           # full-width MobileNetV2
    )
    backbone.trainable = trainable_backbone

    # ── Multi-scale feature extraction ────────────────────────────────────────
    s3, s4, s5 = _get_mobilenet_stage_outputs(backbone)

    def gap_project(feat, units, name):
        """Global-average-pool → Dense → BN → GELU."""
        x = layers.GlobalAveragePooling2D(name=f"gap_{name}")(feat)
        x = layers.Dense(units, kernel_regularizer=reg,
                          name=f"proj_{name}")(x)
        x = layers.BatchNormalization(name=f"bn_{name}")(x)
        x = layers.Activation("gelu", name=f"gelu_{name}")(x)
        return x

    f3 = gap_project(s3, 96,  "s3")   # 96
    f4 = gap_project(s4, 192, "s4")   # 192
    f5 = gap_project(s5, 256, "s5")   # 256

    x = layers.Concatenate(name="scale_concat")([f3, f4, f5])  # 544-dim

    # ── Fusion head ───────────────────────────────────────────────────────────
    x = layers.Dense(512, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="bn_fc1")(x)
    x = layers.Activation("gelu", name="gelu_fc1")(x)
    x = layers.Dropout(cfg["dropout_rate"], name="drop1")(x)

    # Channel-attention (SE)
    x = SqueezeExcitation(512, ratio=cfg["se_ratio"], name="se_block")(x)

    # Embedding layer
    x = layers.Dense(cfg["embed_dim"], kernel_regularizer=reg,
                      name="embedding")(x)
    x = layers.BatchNormalization(name="bn_embed")(x)
    x = layers.Activation("gelu", name="gelu_embed")(x)
    x = layers.Dropout(cfg["dropout_rate"] * 0.50, name="drop2")(x)

    # Output: cast to float32 for numerical stability under mixed precision
    n_out = len(cfg["label_cols"])
    out = layers.Dense(n_out, dtype="float32", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out,
                         name="MobileNetV2_Texture")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 3.  LOSS, OPTIMIZER, COMPILE
# ═════════════════════════════════════════════════════════════════════════════

class MultiOutputHuber(keras.losses.Loss):
    """
    Huber loss averaged over all output dimensions.
    More robust to outlier labels than MSE.
    """

    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._delta = delta
        self._huber = keras.losses.Huber(delta=delta, reduction="auto")

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return self._huber(y_true, y_pred)

    def get_config(self) -> dict:
        return {**super().get_config(), "delta": self._delta}


def _get_optimizer(lr, cfg: Dict) -> keras.optimizers.Optimizer:
    """
    Return AdamW with gradient clipping.
    Tries keras 2.14+ path first, falls back to experimental, then Adam.
    ``lr`` may be a float or a LR-schedule object.
    """
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

    log.warning("AdamW unavailable – falling back to Adam (no weight decay).")
    return keras.optimizers.Adam(learning_rate=lr,
                                  clipnorm=cfg["grad_clip"])


def compile_model(model: keras.Model, lr, cfg: Dict) -> keras.Model:
    """Compile the model with AdamW + Huber loss."""
    model.compile(
        optimizer=_get_optimizer(lr, cfg),
        loss=MultiOutputHuber(delta=1.0),
        metrics=["mae"],
    )
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 4.  CHECKPOINT STATE  (JSON-based resume logic)
# ═════════════════════════════════════════════════════════════════════════════

def _load_state(ckpt_dir: str) -> Dict:
    """Load persisted training state or return a fresh initial state."""
    state_path = os.path.join(ckpt_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        log.info("Checkpoint state loaded: %s", state)
        return state
    return {
        "completed_epochs": 0,
        "best_val_loss":    float("inf"),
        "phase":            "warmup",
    }


def _save_state(state: Dict, ckpt_dir: str) -> None:
    """Persist training state to JSON."""
    with open(os.path.join(ckpt_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  CALLBACKS
# ═════════════════════════════════════════════════════════════════════════════

class CheckpointAndLogCallback(keras.callbacks.Callback):
    """
    Per-epoch callback that:
      1. Saves  last_model.keras  (unconditional – ensures OOM-safe resume)
      2. Saves  best_model.keras  (only if val_loss improved)
      3. Appends a row to the CSV training log
      4. Writes training_state.json for next-run resume logic
    """

    def __init__(self, ckpt_dir: str, csv_path: str, state: Dict):
        super().__init__()
        self.ckpt_dir          = ckpt_dir
        self.csv_path          = csv_path
        self.state             = state
        self.best_val_loss     = float(state.get("best_val_loss", float("inf")))
        self._csv_header_written = os.path.exists(csv_path)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        logs = logs or {}
        ts   = datetime.now().isoformat(timespec="seconds")

        # ── 1. Always save last checkpoint ──
        last_path = os.path.join(self.ckpt_dir, "last_model.keras")
        try:
            self.model.save(last_path, overwrite=True)
        except Exception as exc:
            log.warning("Could not save last checkpoint: %s", exc)

        # ── 2. Save best checkpoint ──
        val_loss = float(logs.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.ckpt_dir, "best_model.keras")
            try:
                self.model.save(best_path, overwrite=True)
                log.info("  ✓ Best model updated  (val_loss = %.5f)", val_loss)
            except Exception as exc:
                log.warning("Could not save best checkpoint: %s", exc)

        # ── 3. Persist training state ──
        self.state["completed_epochs"] = epoch + 1
        self.state["best_val_loss"]    = self.best_val_loss
        _save_state(self.state, self.ckpt_dir)

        # ── 4. CSV append ──
        row = {
            "epoch":     epoch + 1,
            "timestamp": ts,
            **{k: (f"{v:.6f}" if isinstance(v, float) else v)
               for k, v in logs.items()},
        }
        try:
            mode = "a" if self._csv_header_written else "w"
            with open(self.csv_path, mode, newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                writer.writerow(row)
        except Exception as exc:
            log.warning("CSV write error: %s", exc)


def get_standard_callbacks(cfg: Dict, state: Dict) -> List[keras.callbacks.Callback]:
    """Return the standard set of callbacks used in both training phases."""
    ckpt_dir = cfg["ckpt_dir"]
    csv_path = os.path.join(cfg["log_dir"], "training_log.csv")
    tb_dir   = cfg["tb_dir"]

    return [
        CheckpointAndLogCallback(ckpt_dir, csv_path, state),
        keras.callbacks.EarlyStopping(
            monitor            = "val_loss",
            patience           = 12,
            restore_best_weights = True,
            verbose            = 1,
            min_delta          = 1e-5,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,
            patience  = 5,
            min_lr    = 1e-8,
            verbose   = 1,
            min_delta = 1e-5,
        ),
        keras.callbacks.TensorBoard(
            log_dir        = tb_dir,
            histogram_freq = 0,
            update_freq    = "epoch",
        ),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(model: keras.Model,
             test_ds: tf.data.Dataset,
             label_cols: List[str],
             scaler: StandardScaler,
             cfg: Dict) -> Dict:
    """
    Run inference on the held-out test set and compute per-label metrics
    (MAE, RMSE, R², MAPE) in the original (un-scaled) label space.
    Saves a scatter-plot PNG and a metrics.json.
    """
    y_pred_list, y_true_list = [], []
    for imgs, y in test_ds:
        preds = model(imgs, training=False)
        y_pred_list.append(preds.numpy())
        y_true_list.append(y.numpy())

    y_pred = np.concatenate(y_pred_list, axis=0).astype(np.float32)
    y_true = np.concatenate(y_true_list, axis=0).astype(np.float32)

    y_pred_orig = scaler.inverse_transform(y_pred)
    y_true_orig = scaler.inverse_transform(y_true)

    metrics: Dict[str, Dict] = {}
    header = f"{'Property':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}"
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    for i, col in enumerate(label_cols):
        yt   = y_true_orig[:, i]
        yp   = y_pred_orig[:, i]
        mae  = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = float(r2_score(yt, yp))
        mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-9))) * 100)
        metrics[col] = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
        print(f"  {col:<23} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f} {mape:>7.2f}%")

    print(sep + "\n")

    # ── Scatter plots ─────────────────────────────────────────────────────────
    n_cols = len(label_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, label_cols)):
        yt, yp = y_true_orig[:, i], y_pred_orig[:, i]
        ax.scatter(yt, yp, alpha=0.35, s=14, color="#7C3AED")
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="ideal")
        ax.set_xlabel(f"True  {col}")
        ax.set_ylabel(f"Predicted  {col}")
        ax.set_title(f"{col}\nMAE={metrics[col]['mae']:.3f}  "
                     f"R²={metrics[col]['r2']:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("MobileNetV2 – Texture Branch  |  Test Set", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(cfg["output_dir"], "eval_scatter.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Scatter plots → %s", fig_path)

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(cfg["output_dir"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics JSON → %s", metrics_path)

    return metrics


def plot_history(histories: List, labels: List[str], cfg: Dict) -> None:
    """Plot and save training/validation loss curves for each training phase."""
    valid = [(h, l) for h, l in zip(histories, labels) if h is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, len(valid), figsize=(6 * len(valid), 4))
    if len(valid) == 1:
        axes = [axes]

    for ax, (hist, label) in zip(axes, valid):
        ax.plot(hist.history["loss"],     label="Train", linewidth=1.5)
        ax.plot(hist.history["val_loss"], label="Val",   linewidth=1.5)
        ax.set_title(f"{label}  –  Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Huber Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("MobileNetV2 Texture Branch – Training History")
    plt.tight_layout()
    path = os.path.join(cfg["output_dir"], "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Training history → %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  EMBEDDING EXTRACTION  (for future fusion layer)
# ═════════════════════════════════════════════════════════════════════════════

def extract_and_save_embeddings(model: keras.Model,
                                 all_ds: tf.data.Dataset,
                                 cfg: Dict) -> np.ndarray:
    """
    Create a sub-model that outputs the 256-dim 'embedding' layer activations.
    Saves:
      • outputs/embedding_extractor.keras   (loadable sub-model)
      • outputs/embeddings.npy              (all-sample embeddings array)

    These artefacts plug directly into the Fusion Layer in the next stage.
    """
    try:
        embed_out = model.get_layer("embedding").output
    except ValueError:
        log.warning("'embedding' layer not found – using second-to-last layer.")
        embed_out = model.layers[-2].output

    embed_model = keras.Model(
        inputs  = model.input,
        outputs = embed_out,
        name    = "MobileNet_EmbeddingExtractor",
    )
    extractor_path = os.path.join(cfg["output_dir"], "embedding_extractor.keras")
    embed_model.save(extractor_path)
    log.info("Embedding extractor → %s", extractor_path)

    embs = []
    for imgs, _ in all_ds:
        embs.append(embed_model(imgs, training=False).numpy())
    embeddings = np.concatenate(embs, axis=0)

    npy_path = os.path.join(cfg["output_dir"], "embeddings.npy")
    np.save(npy_path, embeddings)
    log.info("Embeddings  %s → %s", embeddings.shape, npy_path)
    return embeddings


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MAIN TRAINING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:  # noqa: C901  (complexity OK for orchestration function)
    cfg = CONFIG
    setup_environment(cfg)

    log.info("=" * 64)
    log.info("  MobileNetV2 – Texture Branch  Training")
    log.info("=" * 64)

    # ── Step 1: Generate / load labels CSV ───────────────────────────────────
    log.info("Step 1 / 7 – Generate labels …")
    labels_df = generate_labels_csv(
        h5_dir                = cfg["h5_dir"],
        csv_path              = cfg["labels_csv"],
        voxel_size_um         = cfg["voxel_size_um"],
        compute_misorientation= True,
    )

    # ── Step 2: Build pole-figure image cache ─────────────────────────────────
    log.info("Step 2 / 7 – Build pole-figure cache …")
    labels_df = build_pole_figure_cache(
        labels_df  = labels_df,
        cache_dir  = cfg["pole_fig_dir"],
        n_grid     = cfg["pole_fig_n_grid"],
        subsample  = cfg["pole_fig_subsample"],
    )

    # ── Step 3: Validate ──────────────────────────────────────────────────────
    log.info("Step 3 / 7 – Validate dataset …")
    labels_df = validate_dataset(
        labels_df,
        img_col    = "pole_fig_path",
        label_cols = cfg["label_cols"],
        min_samples= 20,
    )

    # ── Step 4: Scale labels → zero-mean / unit-variance ─────────────────────
    log.info("Step 4 / 7 – Scale labels …")
    label_matrix = labels_df[cfg["label_cols"]].values.astype(np.float32)
    scaler       = StandardScaler()
    label_scaled = scaler.fit_transform(label_matrix).astype(np.float32)

    # Persist scaler for inference time
    scaler_path = os.path.join(cfg["output_dir"], "label_scaler.pkl")
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    log.info("Label scaler → %s", scaler_path)

    # ── Step 5: Train / val / test split ─────────────────────────────────────
    log.info("Step 5 / 7 – Split data …")
    all_paths = labels_df["pole_fig_path"].tolist()
    idx = np.arange(len(labels_df))

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

    # ── Step 6: Load checkpoint state (handles OOM resume) ───────────────────
    log.info("Step 6 / 7 – Check checkpoints …")
    state = _load_state(cfg["ckpt_dir"])
    done  = state["completed_epochs"]

    total_epochs = cfg["warmup_epochs"] + cfg["finetune_epochs"]
    if done >= total_epochs:
        log.info("Training already complete (%d/%d epochs done).",
                 done, total_epochs)
    else:
        log.info("Completed epochs so far: %d / %d", done, total_epochs)

    last_ckpt = os.path.join(cfg["ckpt_dir"], "last_model.keras")
    best_ckpt = os.path.join(cfg["ckpt_dir"], "best_model.keras")

    _custom = {"SqueezeExcitation": SqueezeExcitation,
               "MultiOutputHuber":  MultiOutputHuber}

    # ── Phase 1: Warmup (backbone frozen) ────────────────────────────────────
    history_warmup = None
    model          = None

    if done < cfg["warmup_epochs"]:
        log.info("─" * 64)
        log.info("PHASE 1 – Warm-up  (backbone frozen,  lr = %.0e)",
                 cfg["lr_warmup"])
        log.info("─" * 64)

        if done > 0 and os.path.exists(last_ckpt):
            log.info("Resuming warm-up from epoch %d …", done)
            model = keras.models.load_model(last_ckpt, custom_objects=_custom)
            compile_model(model, cfg["lr_warmup"], cfg)
        else:
            model = build_mobilenet_model(cfg, trainable_backbone=False)
            compile_model(model, cfg["lr_warmup"], cfg)
            log.info("MobileNetV2 backbone layer count: %d",
                     len(model.get_layer("mobilenetv2_1.00_224").layers))

        model.summary(expand_nested=False, print_fn=log.info)
        state["phase"] = "warmup"

        history_warmup = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = cfg["warmup_epochs"],
            initial_epoch   = done,
            callbacks       = get_standard_callbacks(cfg, state),
            verbose         = 1,
        )
        done = state["completed_epochs"]

        # Named warmup checkpoint for reference
        warmup_path = os.path.join(cfg["ckpt_dir"], "warmup_final.keras")
        model.save(warmup_path)
        log.info("Warmup final checkpoint → %s", warmup_path)

    elif done == cfg["warmup_epochs"]:
        # Warmup finished, fine-tune not yet started → load best warmup weights
        if os.path.exists(best_ckpt):
            model = keras.models.load_model(best_ckpt, custom_objects=_custom)
            log.info("Loaded best warmup model for fine-tuning.")
        else:
            log.warning("best_model.keras not found; rebuilding.")
            model = build_mobilenet_model(cfg, trainable_backbone=False)
    # else: fine-tune was already in progress → model loaded below

    # ── Phase 2: Fine-tune (top 50% of backbone unfrozen) ────────────────────
    history_finetune = None
    if done < total_epochs:
        log.info("─" * 64)
        log.info(
            "PHASE 2 – Fine-tune  (backbone top-%.0f%% unfrozen,  lr = %.0e)",
            (1.0 - cfg["unfreeze_from"]) * 100,
            cfg["lr_finetune"],
        )
        log.info("─" * 64)

        if done > cfg["warmup_epochs"] and os.path.exists(last_ckpt):
            log.info("Resuming fine-tune from epoch %d …", done)
            model = keras.models.load_model(last_ckpt, custom_objects=_custom)

        # Unfreeze top ``unfreeze_from`` fraction of backbone
        try:
            backbone = model.get_layer("mobilenetv2_1.00_224")
        except ValueError:
            # Alternative name in some TF versions
            backbone = model.get_layer("mobilenetv2_1.00_224_1")

        n_layers         = len(backbone.layers)
        unfreeze_idx     = int(n_layers * cfg["unfreeze_from"])
        for layer in backbone.layers[:unfreeze_idx]:
            layer.trainable = False
        for layer in backbone.layers[unfreeze_idx:]:
            layer.trainable = True

        trainable_count = sum(1 for l in backbone.layers if l.trainable)
        log.info("Unfrozen %d / %d backbone layers.", trainable_count, n_layers)

        state["phase"] = "finetune"

        # Cosine-decay LR schedule over the entire fine-tune phase
        steps_per_epoch = max(1, len(tr_i) // cfg["batch_size"])
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = cfg["lr_finetune"],
            decay_steps           = steps_per_epoch * cfg["finetune_epochs"],
            alpha                 = 1e-7,
        )
        model.compile(
            optimizer = _get_optimizer(lr_schedule, cfg),
            loss      = MultiOutputHuber(delta=1.0),
            metrics   = ["mae"],
        )

        history_finetune = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs          = total_epochs,
            initial_epoch   = done,
            callbacks       = get_standard_callbacks(cfg, state),
            verbose         = 1,
        )

    # ── Step 7: Load best model → evaluate → save embeddings ─────────────────
    log.info("Step 7 / 7 – Evaluate & extract embeddings …")
    if os.path.exists(best_ckpt):
        log.info("Loading best model for evaluation …")
        model = keras.models.load_model(best_ckpt, custom_objects=_custom)

    metrics = evaluate(model, test_ds, cfg["label_cols"], scaler, cfg)

    plot_history(
        [history_warmup, history_finetune],
        ["Warm-up", "Fine-tune"],
        cfg,
    )

    extract_and_save_embeddings(model, all_ds, cfg)

    log.info("=" * 64)
    log.info("  ✓ Training complete.")
    log.info("  Best model      → %s", best_ckpt)
    log.info("  Metrics         → %s/metrics.json", cfg["output_dir"])
    log.info("  Embeddings      → %s/embeddings.npy", cfg["output_dir"])
    log.info("  Scaler          → %s", scaler_path)
    log.info("  Embedding model → %s/embedding_extractor.keras",
             cfg["output_dir"])
    log.info("=" * 64)


if __name__ == "__main__":
    main()
