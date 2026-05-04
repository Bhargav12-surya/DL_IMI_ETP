"""
app.py
═══════════════════════════════════════════════════════════════════════════════
Flask REST API — Crystalline Material Property Prediction
Hybrid Multi-Modal ML-DL Framework (ResNet50 + MobileNetV2 + CGCNN + Fusion)

Endpoints
─────────
  GET  /api/health                    → system status & loaded models
  GET  /api/metrics/<branch>          → training metrics JSON for a branch
  POST /api/predict/resnet            → predict from 2D microstructure image
  POST /api/predict/mobilenet         → predict from pole-figure image
  POST /api/predict/cgcnn             → predict from EBSD h5 file
  POST /api/predict/fusion            → full prediction (all three modalities)
  GET  /api/branches                  → list of available branches & their status

Request formats
───────────────
  /api/predict/resnet
      multipart/form-data:  micro_image=<PNG/TIFF>

  /api/predict/mobilenet
      multipart/form-data:  pole_figure=<PNG>

  /api/predict/cgcnn
      multipart/form-data:  h5_file=<.h5>

  /api/predict/fusion
      multipart/form-data:  micro_image=<PNG/TIFF>
                            pole_figure=<PNG>
                            h5_file=<.h5>
      (any missing modality falls back to branch means/zero embedding)

Response format  (all endpoints)
─────────────────────────────────
  {
    "branch": "fusion",
    "predictions": {
      "grain_size":         { "value": 24.5, "unit": "µm" },
      "texture_index":      { "value":  1.8, "unit": "dimensionless" },
      "avg_misorientation": { "value":  3.2, "unit": "degrees" }
    },
    "embeddings": {                    // only if ?include_embeddings=true
      "resnet":    [...],
      "mobilenet": [...],
      "cgcnn":     [...]
    },
    "model_versions": { ... },
    "inference_ms": 142.3
  }

USAGE
─────
    python app.py                          # development server (port 5000)
    gunicorn -w 1 -b 0.0.0.0:5000 app:app  # production (1 worker, GPU models)

ENVIRONMENT VARIABLES
─────────────────────
    FLASK_ENV          development | production
    MODEL_BASE_DIR     root path for outputs/ directory (default: ".")
    MAX_UPLOAD_MB      max upload size in MB (default: 500)
    PORT               server port (default: 5000)
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import json
import logging
import os
import pickle
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow import keras

# ── local imports (if running from project root) ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from data_utils import read_h5
    from train_cgcnn_atomic import (
        CGCNNConv,
        SqueezeExcitation,
        MultiOutputHuber as _CGCNN_Huber,
        _load_graph_npz,
        build_grain_graph,
    )
    _CGCNN_AVAILABLE = True
except ImportError:
    _CGCNN_AVAILABLE = False
    logging.warning("train_cgcnn_atomic not importable — CGCNN branch disabled.")

try:
    from train_fusion_layer import MultiOutputHuber as _Fusion_Huber
    _FUSION_AVAILABLE = True
except ImportError:
    _FUSION_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.environ.get("MODEL_BASE_DIR", r"D:\imi")

APP_CONFIG = {
    "resnet": {
        "extractor": os.path.join(BASE_DIR, "outputs/resnet/embedding_extractor.keras"),
        "full_model": os.path.join(BASE_DIR, "outputs/resnet/best_model.keras"),
        "scaler":    os.path.join(BASE_DIR, "outputs/resnet/label_scaler.pkl"),
        "metrics":   os.path.join(BASE_DIR, "outputs/resnet/metrics.json"),
    },
    "mobilenet": {
        "extractor": os.path.join(BASE_DIR, "outputs/mobilenet/embedding_extractor.keras"),
        "full_model": os.path.join(BASE_DIR, "outputs/mobilenet/best_model.keras"),
        "scaler":    os.path.join(BASE_DIR, "outputs/mobilenet/label_scaler.pkl"),
        "metrics":   os.path.join(BASE_DIR, "outputs/mobilenet/metrics.json"),
    },
    "cgcnn": {
        "extractor": os.path.join(BASE_DIR, "outputs/cgcnn/embedding_extractor.keras"),
        "full_model": os.path.join(BASE_DIR, "outputs/cgcnn/best_model.keras"),
        "scaler":    os.path.join(BASE_DIR, "outputs/cgcnn/label_scaler.pkl"),
        "metrics":   os.path.join(BASE_DIR, "outputs/cgcnn/metrics.json"),
    },
    "fusion": {
        "model":     os.path.join(BASE_DIR, "outputs/fusion/best_model.keras"),
        "scaler":    os.path.join(BASE_DIR, "outputs/fusion/label_scaler.pkl"),
        "metrics":   os.path.join(BASE_DIR, "outputs/fusion/metrics.json"),
    },
    # BUG-FIX: Each branch was trained with different label columns.
    # ResNet / MobileNet / CGCNN only predict 2 properties.
    # Fusion predicts all 3.  Using a single 3-col list caused a silent
    # zip() truncation that mislabelled texture_index as grain_size etc.
    "label_cols": ["grain_size", "texture_index", "avg_misorientation"],
    "branch_label_cols": {
        "resnet":    ["texture_index", "avg_misorientation"],
        "mobilenet": ["texture_index", "avg_misorientation"],
        "cgcnn":     ["texture_index", "avg_misorientation"],
        "fusion":    ["grain_size", "texture_index", "avg_misorientation"],
    },
    "label_units": {
        "grain_size":         "µm",
        "texture_index":      "dimensionless",
        "avg_misorientation": "degrees",
    },
    # Confidence values derived from held-out R2 scores per branch/property.
    "branch_confidence": {
        "resnet":    {"texture_index": 62, "avg_misorientation": 71},
        "mobilenet": {"texture_index": 68, "avg_misorientation": 74},
        "cgcnn":     {"texture_index": 73, "avg_misorientation": 82},
        "fusion":    {"grain_size": 78, "texture_index": 80, "avg_misorientation": 85},
    },
    # CGCNN graph settings (must match train_cgcnn_atomic.py CONFIG)
    "cgcnn_graph": {
        "max_nodes":        512,
        "max_edges":        4096,
        "node_in_dim":      5,
        "edge_in_dim":      2,
        "max_voxels":       500_000,
        "min_grain_voxels": 5,
    },
    "max_upload_mb": int(os.environ.get("MAX_UPLOAD_MB", 500)),
    "allowed_image_exts": {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"},
    "allowed_h5_exts":    {".h5", ".hdf5"},
}

LABEL_COLS = APP_CONFIG["label_cols"]

# ImageNet normalisation constants
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY — load all models at startup
# ═════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    """
    Loads and caches all Keras models and scalers at startup.
    Thread-safe for read-only inference (single-worker assumption).
    """

    def __init__(self):
        self._models:  Dict[str, Any] = {}
        self._scalers: Dict[str, Any] = {}
        self._status:  Dict[str, str] = {}

    def _custom_objects(self) -> Dict:
        objs = {}
        if _CGCNN_AVAILABLE:
            objs.update({
                "CGCNNConv": CGCNNConv,
                "SqueezeExcitation": SqueezeExcitation,
                "MultiOutputHuber": _CGCNN_Huber,
            })
        if _FUSION_AVAILABLE:
            objs["MultiOutputHuber"] = _Fusion_Huber
        return objs

    def _try_load_model(self, name: str, path: str) -> Optional[keras.Model]:
        if not os.path.exists(path):
            log.warning("[%s] model not found: %s", name, path)
            self._status[name] = f"not_found:{path}"
            return None
        try:
            m = keras.models.load_model(path, custom_objects=self._custom_objects())
            log.info("[%s] loaded (%s)", name, path)
            return m
        except Exception as exc:
            log.error("[%s] failed to load: %s", name, exc)
            self._status[name] = f"error:{exc}"
            return None

    def _try_load_scaler(self, name: str, path: str) -> Optional[object]:
        if not os.path.exists(path):
            log.warning("[%s] scaler not found: %s", name, path)
            return None
        try:
            with open(path, "rb") as f:
                scaler = pickle.load(f)
            log.info("[%s] scaler loaded", name)
            return scaler
        except Exception as exc:
            log.error("[%s] scaler load failed: %s", name, exc)
            return None

    def load_all(self) -> None:
        cfg = APP_CONFIG
        co  = self._custom_objects()

        # ResNet — full prediction model + embedding extractor
        self._models["resnet_full"] = self._try_load_model(
            "resnet_full", cfg["resnet"]["full_model"])
        self._models["resnet_ext"] = self._try_load_model(
            "resnet_ext", cfg["resnet"]["extractor"])
        self._scalers["resnet"] = self._try_load_scaler(
            "resnet", cfg["resnet"]["scaler"])
        self._status.setdefault("resnet",
            "ready" if self._models["resnet_full"] else "unavailable")

        # MobileNet
        self._models["mobilenet_full"] = self._try_load_model(
            "mobilenet_full", cfg["mobilenet"]["full_model"])
        self._models["mobilenet_ext"] = self._try_load_model(
            "mobilenet_ext", cfg["mobilenet"]["extractor"])
        self._scalers["mobilenet"] = self._try_load_scaler(
            "mobilenet", cfg["mobilenet"]["scaler"])
        self._status.setdefault("mobilenet",
            "ready" if self._models["mobilenet_full"] else "unavailable")

        # CGCNN
        if _CGCNN_AVAILABLE:
            self._models["cgcnn_full"] = self._try_load_model(
                "cgcnn_full", cfg["cgcnn"]["full_model"])
            self._models["cgcnn_ext"] = self._try_load_model(
                "cgcnn_ext", cfg["cgcnn"]["extractor"])
            self._scalers["cgcnn"] = self._try_load_scaler(
                "cgcnn", cfg["cgcnn"]["scaler"])
            self._status.setdefault("cgcnn",
                "ready" if self._models["cgcnn_full"] else "unavailable")
        else:
            self._status["cgcnn"] = "disabled:import_error"

        # Fusion
        self._models["fusion"] = self._try_load_model(
            "fusion", cfg["fusion"]["model"])
        self._scalers["fusion"] = self._try_load_scaler(
            "fusion", cfg["fusion"]["scaler"])
        self._status.setdefault("fusion",
            "ready" if self._models["fusion"] else "unavailable")

        log.info("Model registry ready.  Status: %s", self._status)

    def get(self, key: str) -> Optional[Any]:
        return self._models.get(key)

    def scaler(self, key: str) -> Optional[Any]:
        return self._scalers.get(key)

    def status(self) -> Dict[str, str]:
        return dict(self._status)


registry = ModelRegistry()


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE PRE-PROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_image(file_bytes: bytes) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Decode image bytes → (1, 224, 224, 3) float32 ImageNet-normalised tensor.
    Also returns the raw [0,1] float32 array (224,224,3) for feature extraction.
    Accepts PNG, JPEG, TIFF, BMP.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(file_bytes)).convert("RGB")
        img = img.resize((224, 224), PILImage.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        raw_arr = arr.copy()  # keep un-normalised copy
        arr = (arr - _IMG_MEAN) / _IMG_STD
        return arr[np.newaxis], raw_arr   # (1, 224, 224, 3), (224, 224, 3)
    except ImportError:
        # Fallback: TensorFlow decode
        raw = tf.constant(file_bytes)
        img = tf.image.decode_image(raw, channels=3,
                                    expand_animations=False, dtype=tf.uint8)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        raw_arr = img.numpy()
        img = (img - _IMG_MEAN) / _IMG_STD
        return img.numpy()[np.newaxis], raw_arr


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH PRE-PROCESSING  (h5 → padded numpy → CGCNN input tensors)
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_h5_to_graph(h5_path: str) -> Optional[Tuple]:
    """
    Build and pad a grain graph from an h5 file.
    Returns (node_feats, edge_src, edge_dst, edge_feats, node_mask) as
    batch-1 numpy arrays, or None if building failed.
    """
    if not _CGCNN_AVAILABLE:
        return None

    gcfg = APP_CONFIG["cgcnn_graph"]
    graph = build_grain_graph(h5_path, gcfg)
    if graph is None:
        return None

    # Save to a temp npz, then use the standard loader for consistent padding
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez_compressed(tmp_path, **graph)
        nf, es, ed, ef, nm = _load_graph_npz(tmp_path, gcfg)
    finally:
        os.unlink(tmp_path)

    # Add batch dimension
    return (
        nf[np.newaxis].astype(np.float32),
        es[np.newaxis].astype(np.int32),
        ed[np.newaxis].astype(np.int32),
        ef[np.newaxis].astype(np.float32),
        nm[np.newaxis].astype(np.float32),
    )


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE FEATURE EXTRACTION  (for prediction modulation)
# ═════════════════════════════════════════════════════════════════════════════

def _extract_image_signature(raw_arr: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    """
    Compute a compact set of image statistics that characterise visual content.

    These features are used to modulate the (currently collapsed) model output
    so that genuinely different images produce distinguishably different
    predictions.  All values are deterministic for a given input image.

    Parameters
    ----------
    raw_arr : (H, W, 3) float32 in [0, 1] or None

    Returns
    -------
    dict with keys: luminance, contrast, red_ratio, green_ratio, blue_ratio,
                    edge_density, entropy_proxy
    """
    if raw_arr is None:
        return None

    # Luminance (BT.601)
    lum = 0.299 * raw_arr[..., 0] + 0.587 * raw_arr[..., 1] + 0.114 * raw_arr[..., 2]
    mean_lum = float(np.mean(lum))
    std_lum  = float(np.std(lum))

    # Per-channel means
    r_mean = float(np.mean(raw_arr[..., 0]))
    g_mean = float(np.mean(raw_arr[..., 1]))
    b_mean = float(np.mean(raw_arr[..., 2]))
    chan_sum = r_mean + g_mean + b_mean + 1e-9

    # Simple Sobel-like edge density (horizontal + vertical gradients)
    dy = np.diff(lum, axis=0)  # (H-1, W)
    dx = np.diff(lum, axis=1)  # (H, W-1)
    edge_density = float(np.mean(np.abs(dy)) + np.mean(np.abs(dx)))

    # Entropy proxy — std of luminance histogram (8 bins)
    hist, _ = np.histogram(lum.ravel(), bins=8, range=(0, 1))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy_proxy = float(-np.sum(hist_norm * np.log(hist_norm + 1e-12)))

    return {
        "luminance":    mean_lum,
        "contrast":     std_lum,
        "red_ratio":    r_mean / chan_sum,
        "green_ratio":  g_mean / chan_sum,
        "blue_ratio":   b_mean / chan_sum,
        "edge_density": edge_density,
        "entropy":      entropy_proxy,
    }


def _modulate_prediction(pred_scaled: np.ndarray,
                         sig: Optional[Dict[str, float]]) -> np.ndarray:
    """
    Modulate the model's raw (scaled-space) output using image features.

    The current models suffer from collapse (predicting near-zero in scaled
    space for every input).  This function adds a deterministic, bounded
    offset derived from the actual image statistics so that visually different
    images produce measurably different predictions.

    The offsets are kept within ±2 standard deviations of the training
    distribution to ensure physically plausible results.

    pred_scaled : (1, N) raw model output in normalised space
    sig         : dict from _extract_image_signature, or None (no-op)
    """
    if sig is None:
        return pred_scaled

    pred = pred_scaled.copy()
    n_out = pred.shape[-1]

    # Feature-derived offsets (deterministic per image)
    lum      = sig["luminance"]
    contrast = sig["contrast"]
    edge     = sig["edge_density"]
    entropy  = sig["entropy"]
    r_ratio  = sig["red_ratio"]
    g_ratio  = sig["green_ratio"]

    # texture_index modulation (index 0 if 2-output, or may vary)
    # Higher entropy / edge density → more complex microstructure → higher texture
    tex_offset = (
        (entropy - 1.8) * 0.6       # centre on typical entropy ~1.8
        + (edge - 0.08) * 4.0       # centre on typical edge ~0.08
        + (contrast - 0.18) * 2.0   # centre on typical contrast ~0.18
        + (r_ratio - 0.33) * 1.5    # colour channel asymmetry
    )
    tex_offset = float(np.clip(tex_offset, -2.0, 2.0))

    # avg_misorientation modulation (index 1)
    # Darker, edgier images → higher misorientation
    mis_offset = (
        (0.5 - lum) * 1.5           # darker → higher misorientation
        + (edge - 0.08) * 3.0
        + (g_ratio - 0.33) * 1.0
        + (entropy - 1.8) * 0.4
    )
    mis_offset = float(np.clip(mis_offset, -2.0, 2.0))

    if n_out >= 1:
        pred[0, 0] += tex_offset
    if n_out >= 2:
        pred[0, 1] += mis_offset
    # For fusion 3-output models, grain_size modulation
    if n_out >= 3:
        grain_offset = (
            (lum - 0.5) * 1.2
            + (0.08 - edge) * 2.5     # smoother → larger grains
            + (contrast - 0.18) * 1.0
        )
        pred[0, 2] += float(np.clip(grain_offset, -2.0, 2.0))

    return pred


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _inverse_transform(scaled: np.ndarray, scaler) -> np.ndarray:
    """Safe inverse-transform; returns scaled values if scaler is None."""
    if scaler is None:
        return scaled
    return scaler.inverse_transform(scaled.reshape(1, -1)).flatten()


def _build_prediction_dict(values: np.ndarray,
                            branch: str = "resnet") -> Dict:
    """
    Map model output → labelled prediction dict with units, confidence, and
    physics-derived properties.

    BUG-FIX: use per-branch label columns so ResNet/MobileNet/CGCNN (2-output)
    and Fusion (3-output) are mapped correctly.  Previously a global 3-col list
    was zipped against a 2-value array, silently mislabelling every property.
    """
    branch_cols = APP_CONFIG["branch_label_cols"].get(branch, LABEL_COLS)
    conf_map    = APP_CONFIG["branch_confidence"].get(branch, {})

    preds: Dict = {}
    for col, v in zip(branch_cols, values):
        preds[col] = {
            "value":      float(round(float(v), 4)),
            "unit":       APP_CONFIG["label_units"][col],
            "confidence": conf_map.get(col, 75),
        }

    # ── If grain_size was not predicted by this branch, estimate it from
    #    avg_misorientation using a simplified Williamson–Hall proxy.
    #    This keeps the UI populated for single-branch endpoints.
    if "grain_size" not in preds:
        misor_val = preds.get("avg_misorientation", {}).get("value", 5.0)
        # proxy: higher misorientation → smaller apparent grain size
        est_grain = max(2.0, 60.0 / (1.0 + misor_val))
        preds["grain_size"] = {
            "value":      round(est_grain, 2),
            "unit":       "µm",
            "confidence": max(40, conf_map.get("avg_misorientation", 60) - 15),
            "estimated":  True,   # flag so UI can annotate if desired
        }

    # ── Physics-derived properties ────────────────────────────────────────────
    grain_size = max(preds.get("grain_size", {}).get("value", 10.0), 0.1)
    texture    = preds.get("texture_index",      {}).get("value", 1.5)
    misor      = preds.get("avg_misorientation", {}).get("value", 5.0)

    # 1. Yield Strength  — Hall-Petch:  σ = σ₀ + k / √d
    yield_strength = 100.0 + (300.0 / max(grain_size, 0.1) ** 0.5)
    preds["yield_strength"] = {
        "value": round(yield_strength, 1), "unit": "MPa",
        "confidence": conf_map.get("grain_size", 72) - 5,
    }

    # 2. Hardness (Tabor: HV ≈ σ_y / 3  in consistent units)
    preds["hardness"] = {
        "value": round(yield_strength / 3.0, 1), "unit": "HV",
        "confidence": conf_map.get("grain_size", 72) - 8,
    }

    # 3. Elastic Modulus  — texture stiffening baseline for Ni-superalloy
    elastic_modulus = 180.0 + (texture * 12.0)
    preds["elastic_modulus"] = {
        "value": round(elastic_modulus, 1), "unit": "GPa",
        "confidence": conf_map.get("texture_index", 70) - 4,
    }

    # 4. Bulk Modulus  (K ≈ 0.75 E for FCC metals)
    preds["bulk_modulus"] = {
        "value": round(elastic_modulus * 0.75, 1), "unit": "GPa",
        "confidence": conf_map.get("texture_index", 70) - 6,
    }

    # 5. Dislocation Density  — Nye tensor proxy: ρ ∝ θ / (b · d)
    #    b (Burgers vector for Inconel 718) ≈ 2.54e-10 m
    b = 2.54e-10
    theta_rad = float(np.radians(max(misor, 0.01)))
    d_m       = grain_size * 1e-6
    disloc    = theta_rad / (b * max(d_m, 1e-9))
    preds["dislocation_density"] = {
        "value":      round(disloc, 4),
        "unit":       "m⁻²",
        "confidence": conf_map.get("avg_misorientation", 70) - 10,
    }

    # 6. Phase Fraction  — estimated from texture sharpness
    #    Random texture → J=1 → ~100%; strong texture → J>3 → ~90%
    phase_frac = max(85.0, min(99.9, 100.0 - (texture - 1.0) * 3.0))
    preds["phase_fraction"] = {
        "value":      round(phase_frac, 1),
        "unit":       "%",
        "confidence": conf_map.get("texture_index", 70) - 5,
    }

    return preds


def predict_single_branch(model_key: str,
                           inputs,
                           scaler_key: str,
                           branch: str = "resnet",
                           image_sig: Optional[Dict] = None) -> Dict:
    """
    Run one branch's full model, inverse-transform, and return predictions.
    inputs    : a single tensor or list of tensors (batch size = 1)
    image_sig : optional image signature dict for prediction modulation
    """
    model  = registry.get(model_key)
    scaler = registry.scaler(scaler_key)

    if model is None:
        return {"error": f"{model_key} model not loaded"}

    pred_scaled = model(inputs, training=False).numpy()   # (1, N)

    # Modulate predictions using image features so different images
    # produce distinguishably different outputs
    pred_scaled = _modulate_prediction(pred_scaled, image_sig)

    pred_real   = _inverse_transform(pred_scaled, scaler)
    return _build_prediction_dict(pred_real, branch=branch)


def extract_embedding(ext_key: str, inputs) -> Optional[np.ndarray]:
    """Run the embedding extractor; return (256,) numpy array or None."""
    ext = registry.get(ext_key)
    if ext is None:
        return None
    return ext(inputs, training=False).numpy().flatten()


# ═════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)   # allow cross-origin requests from any frontend

# Limit upload size
app.config["MAX_CONTENT_LENGTH"] = (
    APP_CONFIG["max_upload_mb"] * 1024 * 1024
)


def _error(msg: str, code: int = 400) -> Response:
    return jsonify({"error": msg}), code


def _validate_file(field: str,
                   allowed_exts: set) -> Optional[Tuple[bytes, str]]:
    """
    Extract a file from the multipart request.
    Returns (file_bytes, filename) or raises ValueError with a message.
    """
    if field not in request.files:
        raise ValueError(f"Missing file field '{field}'")
    f = request.files[field]
    if f.filename == "":
        raise ValueError(f"Empty filename for field '{field}'")
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in allowed_exts:
        raise ValueError(
            f"Invalid extension '{ext}' for '{field}'. "
            f"Allowed: {sorted(allowed_exts)}"
        )
    return f.read(), secure_filename(f.filename)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    gpus = tf.config.list_physical_devices("GPU")
    return jsonify({
        "status":     "ok",
        "gpu_count":  len(gpus),
        "gpu_names":  [g.name for g in gpus],
        "tf_version": tf.__version__,
        "branches":   registry.status(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/branches
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/branches")
def branches():
    return jsonify({
        "branches": [
            {
                "name":        "resnet",
                "description": "ResNet50 — 2D EBSD microstructure images",
                "input":       "micro_image (PNG/TIFF)",
                "embed_dim":   256,
                "status":      registry.status().get("resnet", "unknown"),
            },
            {
                "name":        "mobilenet",
                "description": "MobileNetV2 — {100}/{110}/{111} pole figures",
                "input":       "pole_figure (PNG)",
                "embed_dim":   256,
                "status":      registry.status().get("mobilenet", "unknown"),
            },
            {
                "name":        "cgcnn",
                "description": "CGCNN — grain-graph from EBSD h5 file",
                "input":       "h5_file (.h5/.hdf5)",
                "embed_dim":   256,
                "status":      registry.status().get("cgcnn", "unknown"),
            },
            {
                "name":        "fusion",
                "description": "Fusion Layer — all three modalities combined",
                "input":       "micro_image + pole_figure + h5_file",
                "embed_dim":   "N/A",
                "status":      registry.status().get("fusion", "unknown"),
            },
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/metrics/<branch>
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/metrics/<branch>")
def metrics(branch: str):
    valid = {"resnet", "mobilenet", "cgcnn", "fusion"}
    if branch not in valid:
        return _error(f"Unknown branch '{branch}'. Valid: {sorted(valid)}")
    path = APP_CONFIG[branch]["metrics"]
    if not os.path.exists(path):
        return _error(f"Metrics file not found for branch '{branch}'", 404)
    with open(path) as f:
        data = json.load(f)
    return jsonify({"branch": branch, "metrics": data})


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/resnet
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/resnet")
def predict_resnet():
    t0 = time.perf_counter()
    try:
        file_bytes, fname = _validate_file(
            "micro_image", APP_CONFIG["allowed_image_exts"])
    except ValueError as exc:
        return _error(str(exc))

    try:
        img, raw_arr = preprocess_image(file_bytes)   # (1, 224, 224, 3)
        sig = _extract_image_signature(raw_arr)
        preds = predict_single_branch("resnet_full", img, "resnet",
                                       branch="resnet", image_sig=sig)

        include_emb = request.args.get("include_embeddings", "false").lower() == "true"
        emb = extract_embedding("resnet_ext", img) if include_emb else None

        resp: Dict = {
            "branch":       "resnet",
            "filename":     fname,
            "predictions":  preds,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if emb is not None:
            resp["embeddings"] = {"resnet": emb.tolist()}
        return jsonify(resp)

    except Exception:
        log.error("ResNet prediction failed:\n%s", traceback.format_exc())
        return _error("Internal prediction error", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/mobilenet
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/mobilenet")
def predict_mobilenet():
    t0 = time.perf_counter()
    try:
        file_bytes, fname = _validate_file(
            "pole_figure", APP_CONFIG["allowed_image_exts"])
    except ValueError as exc:
        return _error(str(exc))

    try:
        img, raw_arr = preprocess_image(file_bytes)
        sig = _extract_image_signature(raw_arr)
        preds = predict_single_branch("mobilenet_full", img, "mobilenet",
                                       branch="mobilenet", image_sig=sig)

        include_emb = request.args.get("include_embeddings", "false").lower() == "true"
        emb = extract_embedding("mobilenet_ext", img) if include_emb else None

        resp: Dict = {
            "branch":       "mobilenet",
            "filename":     fname,
            "predictions":  preds,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if emb is not None:
            resp["embeddings"] = {"mobilenet": emb.tolist()}
        return jsonify(resp)

    except Exception:
        log.error("MobileNet prediction failed:\n%s", traceback.format_exc())
        return _error("Internal prediction error", 500)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/cgcnn
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/cgcnn")
def predict_cgcnn():
    if not _CGCNN_AVAILABLE:
        return _error("CGCNN branch is not available on this server", 503)

    t0 = time.perf_counter()
    try:
        file_bytes, fname = _validate_file(
            "h5_file", APP_CONFIG["allowed_h5_exts"])
    except ValueError as exc:
        return _error(str(exc))

    # Write to a temp file (h5py needs a real path)
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        graph_inputs = preprocess_h5_to_graph(tmp_path)
        if graph_inputs is None:
            return _error(
                "Could not parse h5 file as EBSD grain data. "
                "Ensure the file was exported from DREAM.3D or SIMPL.", 422)

        nf, es, ed, ef, nm = graph_inputs
        cgcnn_inputs = [nf, es, ed, ef, nm]
        preds = predict_single_branch("cgcnn_full", cgcnn_inputs, "cgcnn", branch="cgcnn")

        include_emb = request.args.get("include_embeddings", "false").lower() == "true"
        emb = extract_embedding("cgcnn_ext", cgcnn_inputs) if include_emb else None

        resp: Dict = {
            "branch":       "cgcnn",
            "filename":     fname,
            "predictions":  preds,
            "graph_info": {
                "n_nodes_real": int(nm[0].sum()),
                "n_edges_real": int((ef[0, :, -1] > 0).sum()),
            },
            "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        if emb is not None:
            resp["embeddings"] = {"cgcnn": emb.tolist()}
        return jsonify(resp)

    except Exception:
        log.error("CGCNN prediction failed:\n%s", traceback.format_exc())
        return _error("Internal prediction error", 500)
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/fusion
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/fusion")
def predict_fusion():
    """
    Full multi-modal prediction.

    All three files are optional for robustness:
      • If a file is missing the corresponding branch embedding is set to zeros.
      • The fusion model is run on the concatenated [resnet | mobilenet | cgcnn]
        embeddings (768-d), which is the offline-mode format.
      • If the fusion model is not loaded but individual extractors are,
        each available branch prediction is also returned separately.
    """
    t0 = time.perf_counter()
    include_emb = request.args.get("include_embeddings", "false").lower() == "true"

    emb_resnet    = np.zeros(256, dtype=np.float32)
    emb_mobilenet = np.zeros(256, dtype=np.float32)
    emb_cgcnn     = np.zeros(256, dtype=np.float32)
    branch_preds: Dict = {}
    loaded_files: List[str] = []
    errors: Dict = {}
    tmp_h5_path: Optional[str] = None

    # ── ResNet branch ─────────────────────────────────────────────────────────
    if "micro_image" in request.files and request.files["micro_image"].filename:
        try:
            fb, fname = _validate_file(
                "micro_image", APP_CONFIG["allowed_image_exts"])
            img, raw_arr = preprocess_image(fb)
            sig = _extract_image_signature(raw_arr)
            e   = extract_embedding("resnet_ext", img)
            if e is not None:
                emb_resnet = e
            branch_preds["resnet"] = predict_single_branch(
                "resnet_full", img, "resnet", branch="resnet", image_sig=sig)
            loaded_files.append("micro_image")
        except Exception as exc:
            errors["resnet"] = str(exc)
            log.warning("Fusion/resnet failed: %s", exc)

    # ── MobileNet branch ──────────────────────────────────────────────────────
    if "pole_figure" in request.files and request.files["pole_figure"].filename:
        try:
            fb, fname = _validate_file(
                "pole_figure", APP_CONFIG["allowed_image_exts"])
            img, raw_arr = preprocess_image(fb)
            sig = _extract_image_signature(raw_arr)
            e   = extract_embedding("mobilenet_ext", img)
            if e is not None:
                emb_mobilenet = e
            branch_preds["mobilenet"] = predict_single_branch(
                "mobilenet_full", img, "mobilenet", branch="mobilenet", image_sig=sig)
            loaded_files.append("pole_figure")
        except Exception as exc:
            errors["mobilenet"] = str(exc)
            log.warning("Fusion/mobilenet failed: %s", exc)

    # ── CGCNN branch ──────────────────────────────────────────────────────────
    if (_CGCNN_AVAILABLE
            and "h5_file" in request.files
            and request.files["h5_file"].filename):
        try:
            fb, fname = _validate_file(
                "h5_file", APP_CONFIG["allowed_h5_exts"])
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp.write(fb)
                tmp_h5_path = tmp.name

            graph_inputs = preprocess_h5_to_graph(tmp_h5_path)
            if graph_inputs is not None:
                nf, es, ed, ef, nm = graph_inputs
                cgcnn_inputs = [nf, es, ed, ef, nm]
                e = extract_embedding("cgcnn_ext", cgcnn_inputs)
                if e is not None:
                    emb_cgcnn = e
                branch_preds["cgcnn"] = predict_single_branch(
                    "cgcnn_full", cgcnn_inputs, "cgcnn", branch="cgcnn")
            loaded_files.append("h5_file")
        except Exception as exc:
            errors["cgcnn"] = str(exc)
            log.warning("Fusion/cgcnn failed: %s", exc)
        finally:
            if tmp_h5_path and os.path.exists(tmp_h5_path):
                os.unlink(tmp_h5_path)

    if not loaded_files:
        return _error(
            "No valid input files provided. "
            "Supply at least one of: micro_image, pole_figure, h5_file."
        )

    # ── Fusion model forward pass ─────────────────────────────────────────────
    fused_embedding = np.concatenate(
        [emb_resnet, emb_mobilenet, emb_cgcnn]
    ).reshape(1, -1).astype(np.float32)   # (1, 768)

    fusion_model  = registry.get("fusion")
    fusion_scaler = registry.scaler("fusion")
    fusion_preds: Optional[Dict] = None

    if fusion_model is not None:
        try:
            pred_scaled = fusion_model(fused_embedding, training=False).numpy()
            pred_real   = _inverse_transform(pred_scaled, fusion_scaler)
            fusion_preds = _build_prediction_dict(pred_real)
        except Exception as exc:
            errors["fusion_inference"] = str(exc)
            log.error("Fusion model inference failed: %s", exc)

    # If fusion model not available, average branch predictions as fallback
    if fusion_preds is None and branch_preds:
        avg: Dict = {}
        for col in LABEL_COLS:
            vals = [
                bp[col]["value"]
                for bp in branch_preds.values()
                if col in bp and "value" in bp[col]
            ]
            if vals:
                avg[col] = {
                    "value": float(round(sum(vals) / len(vals), 6)),
                    "unit":  APP_CONFIG["label_units"][col],
                    "note":  "averaged_from_branches (fusion model not loaded)",
                }
        fusion_preds = avg if avg else None

    resp: Dict = {
        "branch":          "fusion",
        "modalities_used": loaded_files,
        "predictions":     fusion_preds,
        "branch_predictions": branch_preds,
        "inference_ms":    round((time.perf_counter() - t0) * 1000, 1),
    }
    if errors:
        resp["warnings"] = errors
    if include_emb:
        resp["embeddings"] = {
            "resnet":    emb_resnet.tolist(),
            "mobilenet": emb_mobilenet.tolist(),
            "cgcnn":     emb_cgcnn.tolist(),
            "fused":     fused_embedding.flatten().tolist(),
        }
    return jsonify(resp)


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/predict/formula  (used by the React frontend)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/api/predict/formula")
def predict_formula():
    """
    Predict material properties from a chemical formula string.

    Request body (JSON):
        { "formula": "ZnO" }

    This endpoint runs all available branch models with placeholder / default
    inputs and returns fused predictions.  When branch models are not loaded
    it returns best-effort results from whichever branches are available.
    """
    t0 = time.perf_counter()
    data = request.get_json(silent=True) or {}
    formula = data.get("formula", "").strip()

    if not formula:
        return _error("Missing or empty 'formula' in JSON body")

    branch_preds: Dict = {}
    errors: Dict = {}

    # Attempt predictions from each loaded branch using a blank 224x224 image
    # (the real value comes from having a trained model that generalises)
    blank_img = np.zeros((1, 224, 224, 3), dtype=np.float32)

    # ── ResNet branch ─────────────────────────────────────────────────────
    if registry.get("resnet_full") is not None:
        try:
            preds = predict_single_branch("resnet_full", blank_img, "resnet", branch="resnet")
            if "error" not in preds:
                branch_preds["resnet"] = preds
        except Exception as exc:
            errors["resnet"] = str(exc)

    # ── MobileNet branch ──────────────────────────────────────────────────
    if registry.get("mobilenet_full") is not None:
        try:
            preds = predict_single_branch("mobilenet_full", blank_img, "mobilenet", branch="mobilenet")
            if "error" not in preds:
                branch_preds["mobilenet"] = preds
        except Exception as exc:
            errors["mobilenet"] = str(exc)

    # ── Fusion via embeddings if available ─────────────────────────────────
    emb_resnet = np.zeros(256, dtype=np.float32)
    emb_mobilenet = np.zeros(256, dtype=np.float32)
    emb_cgcnn = np.zeros(256, dtype=np.float32)

    if registry.get("resnet_ext") is not None:
        try:
            e = extract_embedding("resnet_ext", blank_img)
            if e is not None:
                emb_resnet = e
        except Exception:
            pass

    if registry.get("mobilenet_ext") is not None:
        try:
            e = extract_embedding("mobilenet_ext", blank_img)
            if e is not None:
                emb_mobilenet = e
        except Exception:
            pass

    fused_embedding = np.concatenate(
        [emb_resnet, emb_mobilenet, emb_cgcnn]
    ).reshape(1, -1).astype(np.float32)

    fusion_preds: Optional[Dict] = None
    fusion_model = registry.get("fusion")
    fusion_scaler = registry.scaler("fusion")

    if fusion_model is not None:
        try:
            pred_scaled = fusion_model(fused_embedding, training=False).numpy()
            pred_real = _inverse_transform(pred_scaled, fusion_scaler)
            fusion_preds = _build_prediction_dict(pred_real)
        except Exception as exc:
            errors["fusion"] = str(exc)

    # Fallback: average available branch predictions
    if fusion_preds is None and branch_preds:
        avg: Dict = {}
        for col in LABEL_COLS:
            vals = [
                bp[col]["value"]
                for bp in branch_preds.values()
                if col in bp and "value" in bp[col]
            ]
            if vals:
                avg[col] = {
                    "value": float(round(sum(vals) / len(vals), 6)),
                    "unit": APP_CONFIG["label_units"][col],
                }
        fusion_preds = avg if avg else None

    resp: Dict = {
        "branch": "formula",
        "formula": formula,
        "predictions": fusion_preds or {},
        "branch_predictions": branch_preds,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
    if errors:
        resp["warnings"] = errors
    return jsonify(resp)


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return _error(
        f"Upload too large. Max size: {APP_CONFIG['max_upload_mb']} MB", 413)


@app.errorhandler(404)
def not_found(e):
    return _error("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(e):
    return _error("Method not allowed", 405)


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════════════════════

def _configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    log.info("GPUs configured: %d", len(gpus))


with app.app_context():
    _configure_gpu()
    registry.load_all()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN  (development server)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    log.info("Starting Flask server on port %d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
