"""
training_utils.py
═══════════════════════════════════════════════════════════════════════════════
Shared training utilities for all four pipeline branches.

Provides:
  • setup_cuda()                  — CUDA device selection & diagnostics
  • setup_train_logger()          — structured Python file-handler logging
  • get_git_info() / tag_experiment() — git-based experiment versioning
  • EnhancedCheckpointCallback    — epoch-N snapshots + VERSION metadata
  • GradientAccumulationModel     — gradient accumulation for large batches
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Delay TF import so CUDA env vars can be set first
_tf = None
_keras = None


def _ensure_tf():
    """Lazy-import TensorFlow so CUDA env vars are set before first import."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        _tf = tf
        _keras = keras
    return _tf, _keras


# ═════════════════════════════════════════════════════════════════════════════
# 1.  CUDA / DEVICE SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_cuda(cfg: Dict, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Configure CUDA devices and print a diagnostic banner.

    Reads optional config keys:
        cfg["cuda_visible_devices"]  — e.g. "0", "0,1", or "" for CPU-only
        cfg["use_mixed_precision"]   — enable fp16 (default True if GPU)

    Also honours the CUDA_VISIBLE_DEVICES environment variable (config
    takes priority if both are set).

    Returns a dict with device diagnostics for logging/serialization.
    """
    log = logger or logging.getLogger(__name__)

    # ── 1. Set CUDA_VISIBLE_DEVICES before TF init ────────────────────────
    if "cuda_visible_devices" in cfg:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["cuda_visible_devices"])
        log.info("CUDA_VISIBLE_DEVICES set to '%s' (from config)",
                 cfg["cuda_visible_devices"])
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        log.info("CUDA_VISIBLE_DEVICES = '%s' (from environment)",
                 os.environ["CUDA_VISIBLE_DEVICES"])

    tf, keras = _ensure_tf()

    # ── 2. GPU discovery & memory growth ──────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    diag: Dict[str, Any] = {
        "n_gpus": len(gpus),
        "gpu_names": [],
        "cuda_built": tf.test.is_built_with_cuda(),
        "tf_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
            # Extract GPU name
            details = tf.config.experimental.get_device_details(g)
            name = details.get("device_name", g.name)
            diag["gpu_names"].append(name)
            compute_cap = details.get("compute_capability", "unknown")
            diag.setdefault("compute_capabilities", []).append(
                f"{compute_cap[0]}.{compute_cap[1]}"
                if isinstance(compute_cap, tuple) else str(compute_cap)
            )
    else:
        log.warning("No GPU detected. Training will run on CPU (very slow).")

    # ── 3. Mixed precision ────────────────────────────────────────────────
    use_mp = cfg.get("use_mixed_precision", True) and len(gpus) > 0
    if use_mp:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            diag["mixed_precision"] = "float16"
            log.info("Mixed precision (fp16) enabled.")
        except Exception as exc:
            log.warning("Could not enable mixed precision: %s", exc)
            tf.keras.mixed_precision.set_global_policy("float32")
            diag["mixed_precision"] = "float32"
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
        diag["mixed_precision"] = "float32"

    # ── 4. CUDA / cuDNN version info ──────────────────────────────────────
    try:
        cuda_version = tf.sysconfig.get_build_info().get(
            "cuda_version", "unknown")
        cudnn_version = tf.sysconfig.get_build_info().get(
            "cudnn_version", "unknown")
    except Exception:
        cuda_version = "unknown"
        cudnn_version = "unknown"
    diag["cuda_version"] = cuda_version
    diag["cudnn_version"] = cudnn_version

    # ── 5. Print diagnostic banner ────────────────────────────────────────
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║            DEVICE  DIAGNOSTICS                  ║")
    log.info("╠══════════════════════════════════════════════════╣")
    log.info("║  TensorFlow   : %-31s ║", tf.__version__)
    log.info("║  Python       : %-31s ║", platform.python_version())
    log.info("║  CUDA built   : %-31s ║", diag["cuda_built"])
    log.info("║  CUDA version : %-31s ║", cuda_version)
    log.info("║  cuDNN version: %-31s ║", cudnn_version)
    log.info("║  GPUs found   : %-31s ║", len(gpus))
    for i, name in enumerate(diag["gpu_names"]):
        cc = diag.get("compute_capabilities", ["?"])[i]
        log.info("║    GPU %d      : %-31s ║", i, f"{name} (CC {cc})")
    log.info("║  Mixed prec.  : %-31s ║", diag["mixed_precision"])
    log.info("╚══════════════════════════════════════════════════╝")

    return diag


# ═════════════════════════════════════════════════════════════════════════════
# 2.  STRUCTURED TRAINING LOGGER
# ═════════════════════════════════════════════════════════════════════════════

def setup_train_logger(branch_name: str,
                       log_dir: str,
                       logger: Optional[logging.Logger] = None,
                       ) -> tuple[str, logging.Logger]:
    """
    Attach a FileHandler to the Python logger so all log.info() calls
    are also written to ``logs/<branch>/train_<run_id>.log``.

    Returns (run_id, logger).
    """
    os.makedirs(log_dir, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:6]
    log_file = os.path.join(log_dir, f"train_{run_id}.log")

    log = logger or logging.getLogger(branch_name)

    # Avoid adding duplicate handlers on re-entry
    if not any(isinstance(h, logging.FileHandler) and
               getattr(h, '_train_utils_marker', False)
               for h in log.handlers):
        fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        fh._train_utils_marker = True  # type: ignore[attr-defined]
        log.addHandler(fh)

    # Write header
    log.info("=" * 60)
    log.info("  Training Run: %s", run_id)
    log.info("  Branch      : %s", branch_name)
    log.info("  Log file    : %s", log_file)
    log.info("  Started at  : %s", datetime.now().isoformat(timespec="seconds"))
    log.info("  Platform    : %s", platform.platform())
    log.info("  Python      : %s", platform.python_version())
    log.info("=" * 60)

    return run_id, log


# ═════════════════════════════════════════════════════════════════════════════
# 3.  GIT VERSION CONTROL
# ═════════════════════════════════════════════════════════════════════════════

def get_git_info(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Query the git repository for the current commit hash, branch, and
    dirty status.  Returns a dict that can be serialized into JSON.

    If git is not available or repo_dir is not a repo, returns a dict
    with "unavailable" fields.
    """
    cwd = repo_dir or os.getcwd()
    info: Dict[str, Any] = {
        "commit_hash": "unknown",
        "branch": "unknown",
        "is_dirty": None,
        "commit_msg": "unknown",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        def _run(cmd: List[str]) -> str:
            return subprocess.check_output(
                cmd, cwd=cwd, stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()

        info["commit_hash"] = _run(["git", "rev-parse", "HEAD"])
        info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["is_dirty"] = len(_run(["git", "status", "--porcelain"])) > 0
        info["commit_msg"] = _run(["git", "log", "-1", "--pretty=%s"])
    except Exception:
        pass  # git not installed or not a repo
    return info


def tag_experiment(branch_name: str,
                   run_id: str,
                   repo_dir: Optional[str] = None,
                   logger: Optional[logging.Logger] = None) -> Optional[str]:
    """
    Create a lightweight git tag for the current experiment run.
    Tag format: ``run/<branch>/<run_id>``

    Only creates a tag if the working tree is clean.
    Returns the tag name on success, None on failure/skip.
    """
    log = logger or logging.getLogger(__name__)
    cwd = repo_dir or os.getcwd()
    tag_name = f"run/{branch_name}/{run_id}"

    try:
        # Check if tree is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        if status:
            log.info("Git working tree is dirty — skipping experiment tag.")
            return None

        subprocess.check_call(
            ["git", "tag", tag_name],
            cwd=cwd, stderr=subprocess.DEVNULL, timeout=5,
        )
        log.info("Created git tag: %s", tag_name)
        return tag_name
    except Exception as exc:
        log.debug("Could not create git tag: %s", exc)
        return None


def save_version_file(save_dir: str,
                      cfg: Dict,
                      run_id: str,
                      device_diag: Optional[Dict] = None,
                      repo_dir: Optional[str] = None) -> str:
    """
    Write a VERSION.json file alongside checkpoints containing
    full reproducibility metadata.
    """
    os.makedirs(save_dir, exist_ok=True)
    version_info = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": get_git_info(repo_dir),
        "config": {k: str(v) for k, v in cfg.items()},  # stringify for JSON safety
        "device": device_diag or {},
    }
    path = os.path.join(save_dir, "VERSION.json")
    with open(path, "w") as f:
        json.dump(version_info, f, indent=2, default=str)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ENHANCED CHECKPOINT CALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedCheckpointCallback:
    """
    Factory that returns a keras.callbacks.Callback.

    Must be called AFTER TensorFlow is imported.  Usage:
        callback = EnhancedCheckpointCallback.create(ckpt_dir, csv_path, ...)
    """

    @staticmethod
    def create(ckpt_dir: str,
               csv_path: str,
               state: Dict,
               run_id: str,
               cfg: Dict,
               repo_dir: Optional[str] = None,
               logger: Optional[logging.Logger] = None):
        """Build and return the callback instance."""
        tf, keras = _ensure_tf()
        log = logger or logging.getLogger(__name__)

        ckpt_every_n = cfg.get("ckpt_every_n_epochs", 10)
        max_kept_ckpts = cfg.get("max_kept_ckpts", 5)

        class _Callback(keras.callbacks.Callback):
            """
            Every epoch:
              • Saves 'last_model.keras'  (unconditional — for OOM resume)
              • Saves 'best_model.keras'  (only if val_loss improved)
              • Saves 'ckpt_epoch_NNN.keras' every N epochs
              • Appends a row to the CSV training log
              • Writes training_state.json with git info for resume logic
              • Writes VERSION.json for reproducibility
            """

            def __init__(self):
                super().__init__()
                self.ckpt_dir = ckpt_dir
                self.csv_path = csv_path
                self.state = state
                self.run_id = run_id
                self.best_val_loss = float(state.get("best_val_loss", float("inf")))
                self._csv_header_written = os.path.exists(csv_path)
                self._epoch_ckpts: List[str] = []  # track epoch-N snapshots

            def on_train_begin(self, logs=None):
                # Write VERSION.json at training start
                save_version_file(
                    self.ckpt_dir, cfg, self.run_id,
                    repo_dir=repo_dir,
                )
                log.info("VERSION.json written to %s", self.ckpt_dir)

            def on_epoch_end(self, epoch: int, logs=None):
                logs = logs or {}
                ts = datetime.now().isoformat(timespec="seconds")

                # ── 1. Save last checkpoint (always) ──
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
                        log.info("  ✓ Best model updated  (val_loss = %.5f)",
                                 val_loss)
                    except Exception as exc:
                        log.warning("Could not save best checkpoint: %s", exc)

                # ── 3. Periodic epoch-N snapshot ──
                actual_epoch = epoch + 1
                if ckpt_every_n > 0 and actual_epoch % ckpt_every_n == 0:
                    snap_path = os.path.join(
                        self.ckpt_dir,
                        f"ckpt_epoch_{actual_epoch:04d}.keras",
                    )
                    try:
                        self.model.save(snap_path, overwrite=True)
                        self._epoch_ckpts.append(snap_path)
                        log.info("  📌 Epoch %d snapshot → %s",
                                 actual_epoch, snap_path)

                        # Prune old snapshots if over limit
                        while (max_kept_ckpts > 0 and
                               len(self._epoch_ckpts) > max_kept_ckpts):
                            old = self._epoch_ckpts.pop(0)
                            if os.path.exists(old):
                                os.remove(old)
                                log.debug("Pruned old snapshot: %s", old)
                    except Exception as exc:
                        log.warning("Could not save epoch snapshot: %s", exc)

                # ── 4. Persist training state ──
                self.state["completed_epochs"] = actual_epoch
                self.state["best_val_loss"] = self.best_val_loss
                self.state["run_id"] = self.run_id
                self.state["git"] = get_git_info(repo_dir)
                _save_state_json(self.state, self.ckpt_dir)

                # ── 5. CSV row ──
                row = {
                    "epoch": actual_epoch,
                    "timestamp": ts,
                    "run_id": self.run_id,
                    **{k: (f"{v:.6f}" if isinstance(v, float) else v)
                       for k, v in logs.items()},
                }
                try:
                    mode = "a" if self._csv_header_written else "w"
                    with open(self.csv_path, mode, newline="") as fh:
                        writer = csv.DictWriter(fh,
                                                fieldnames=list(row.keys()))
                        if not self._csv_header_written:
                            writer.writeheader()
                            self._csv_header_written = True
                        writer.writerow(row)
                except Exception as exc:
                    log.warning("CSV write error: %s", exc)

        return _Callback()


def _save_state_json(state: Dict, ckpt_dir: str) -> None:
    """Persist training state to JSON."""
    path = os.path.join(ckpt_dir, "training_state.json")
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ═════════════════════════════════════════════════════════════════════════════
# 5.  GRADIENT ACCUMULATION
# ═════════════════════════════════════════════════════════════════════════════

def build_gradient_accumulation_model(base_model, accum_steps: int = 1):
    """
    Wrap a compiled keras.Model with gradient accumulation.

    When ``accum_steps == 1``, returns the original model unchanged.

    Usage:
        model = build_resnet_model(cfg)
        compile_model(model, lr, cfg)
        model = build_gradient_accumulation_model(model, cfg["grad_accum_steps"])
        model.fit(...)

    The effective batch size becomes ``batch_size × accum_steps``.
    """
    if accum_steps <= 1:
        return base_model

    tf, keras = _ensure_tf()

    class GradientAccumulationModel(keras.Model):
        """
        Keras model wrapper that accumulates gradients over multiple
        micro-batches before applying a single optimizer step.

        This allows training with larger effective batch sizes on
        limited VRAM without increasing memory usage.
        """

        def __init__(self, inner_model, n_accum_steps: int, **kwargs):
            super().__init__(**kwargs)
            self.inner_model = inner_model
            self.n_accum_steps = n_accum_steps
            self._step_counter = tf.Variable(
                0, trainable=False, dtype=tf.int64, name="accum_step_counter",
            )
            self._grad_accumulators = None

        @property
        def metrics(self):
            return self.inner_model.metrics

        def _init_accumulators(self):
            if self._grad_accumulators is None:
                self._grad_accumulators = [
                    tf.Variable(
                        tf.zeros_like(v), trainable=False,
                        name=f"grad_accum_{i}",
                    )
                    for i, v in enumerate(self.inner_model.trainable_variables)
                ]

        def call(self, inputs, training=False, **kwargs):
            return self.inner_model(inputs, training=training, **kwargs)

        def train_step(self, data):
            x, y = data[0], data[1]
            sample_weight = data[2] if len(data) > 2 else None

            self._init_accumulators()

            # Forward + backward (compute gradients for this micro-batch)
            with tf.GradientTape() as tape:
                y_pred = self.inner_model(x, training=True)
                loss = self.compiled_loss(
                    y, y_pred, sample_weight=sample_weight,
                    regularization_losses=self.inner_model.losses,
                )
                # Scale loss by accumulation steps
                scaled_loss = loss / tf.cast(
                    self.n_accum_steps, dtype=loss.dtype)

            gradients = tape.gradient(
                scaled_loss, self.inner_model.trainable_variables)

            # Accumulate gradients
            for acc, grad in zip(self._grad_accumulators, gradients):
                if grad is not None:
                    acc.assign_add(grad)

            self._step_counter.assign_add(1)

            # Apply accumulated gradients every n_accum_steps
            tf.cond(
                tf.equal(
                    self._step_counter % self.n_accum_steps, 0),
                true_fn=lambda: self._apply_and_reset(),
                false_fn=lambda: None,
            )

            # Update metrics
            self.compiled_metrics.update_state(
                y, y_pred, sample_weight=sample_weight)

            return {m.name: m.result() for m in self.metrics}

        def _apply_and_reset(self):
            self.optimizer.apply_gradients(
                zip(self._grad_accumulators,
                    self.inner_model.trainable_variables)
            )
            for acc in self._grad_accumulators:
                acc.assign(tf.zeros_like(acc))

        def get_config(self):
            return {
                "n_accum_steps": self.n_accum_steps,
            }

        def save(self, filepath, **kwargs):
            """Save the inner model (not the wrapper)."""
            self.inner_model.save(filepath, **kwargs)

        @property
        def layers(self):
            return self.inner_model.layers

        def get_layer(self, name=None, index=None):
            return self.inner_model.get_layer(name=name, index=index)

        def summary(self, **kwargs):
            return self.inner_model.summary(**kwargs)

    # Build the wrapper
    ga_model = GradientAccumulationModel(base_model, accum_steps)

    # Transfer compilation (optimizer, loss, metrics) from inner model.
    # BUG-FIX: compiled_metrics._metrics was removed in Keras 3 / TF 2.16+.
    # Safely fall back to ["mae"] which is what every branch uses anyway.
    try:
        # Keras 2 path
        _metrics = base_model.compiled_metrics._metrics
    except AttributeError:
        try:
            # Keras 3 path: metrics are stored differently
            _metrics = [m for m in base_model.metrics
                        if m.name != "loss"] or ["mae"]
        except Exception:
            _metrics = ["mae"]
    ga_model.compile(
        optimizer=base_model.optimizer,
        loss=base_model.loss,
        metrics=_metrics,
    )

    logging.getLogger(__name__).info(
        "Gradient accumulation enabled: %d steps  "
        "(effective batch = batch_size × %d)",
        accum_steps, accum_steps,
    )

    return ga_model


# ═════════════════════════════════════════════════════════════════════════════
# 6.  CONVENIENCE: get_standard_callbacks (shared version)
# ═════════════════════════════════════════════════════════════════════════════

def get_enhanced_callbacks(cfg: Dict,
                           state: Dict,
                           run_id: str,
                           repo_dir: Optional[str] = None,
                           lr_scheduler=None,
                           logger: Optional[logging.Logger] = None,
                           ) -> list:
    """
    Return the standard set of callbacks used in all training phases,
    with the enhanced checkpoint callback replacing the old one.
    """
    tf, keras = _ensure_tf()
    log = logger or logging.getLogger(__name__)

    ckpt_dir = cfg["ckpt_dir"]
    csv_path = os.path.join(cfg["log_dir"], "training_log.csv")
    tb_dir   = cfg["tb_dir"]

    callbacks = [
        EnhancedCheckpointCallback.create(
            ckpt_dir=ckpt_dir,
            csv_path=csv_path,
            state=state,
            run_id=run_id,
            cfg=cfg,
            repo_dir=repo_dir,
            logger=log,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-5,
        ),
    ]

    # BUG-FIX: TensorBoard may not be installed — make it optional.
    # Install with: pip install tensorboard
    try:
        callbacks.append(keras.callbacks.TensorBoard(
            log_dir=tb_dir,
            histogram_freq=0,
            update_freq="epoch",
        ))
    except Exception as _tb_err:
        log.warning("TensorBoard callback skipped (%s). "
                    "Install tensorboard for live metrics: "
                    "pip install tensorboard", _tb_err)

    if lr_scheduler is not None:
        if isinstance(lr_scheduler, keras.callbacks.Callback):
            callbacks.insert(2, lr_scheduler)
        else:
            callbacks.insert(2,
                             keras.callbacks.LearningRateScheduler(lr_scheduler))
    else:
        callbacks.insert(2, keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1,
            min_delta=1e-5,
        ))

    return callbacks
