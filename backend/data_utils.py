"""
data_utils.py
─────────────────────────────────────────────────────────────────────────────
Shared EBSD data utilities for the Microstructure + Texture DL pipeline.
Handles: h5 reading, pole-figure generation, label computation,
         image pairing, dataset validation.

Dataset : Dryad  doi:10.5061/dryad.83bk3j9sj
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import csv
import glob
import warnings
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── optional scipy ────────────────────────────────────────────────────────────
try:
    from scipy.ndimage import gaussian_filter as _scipy_gauss
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    log.warning("scipy not found – pole figures will not be smoothed. "
                "Install with: pip install scipy")

# ── optional Pillow ───────────────────────────────────────────────────────────
try:
    from PIL import Image as _PIL_Image
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False
    log.warning("Pillow not found. Install with: pip install Pillow")


# ═════════════════════════════════════════════════════════════════════════════
# 1.  EBSD H5 READER
# ═════════════════════════════════════════════════════════════════════════════

# Known DREAM.3D / SIMPL dataset paths (tried in order)
_EULER_CANDIDATES = [
    "/DataContainers/ImageDataContainer/CellData/EulerAngles",
    "/DataContainers/SyntheticVolumeDataContainer/CellData/EulerAngles",
    "/DataContainers/StatsGeneratorDataContainer/CellData/EulerAngles",
    "EulerAngles",
]
_GRAIN_CANDIDATES = [
    "/DataContainers/ImageDataContainer/CellData/FeatureIds",
    "/DataContainers/ImageDataContainer/CellData/GrainIds",
    "/DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds",
    "FeatureIds",
    "GrainIds",
]
_KAM_CANDIDATES = [
    "Scan 1/EBSD/Data/KAM",
    "KAM",
]
_PHASE_CANDIDATES = [
    "/DataContainers/ImageDataContainer/CellData/Phases",
    "/DataContainers/SyntheticVolumeDataContainer/CellData/Phases",
    "Scan 1/EBSD/Data/Phase",
    "Phases",
    "Phase",
]
_DIM_CANDIDATES = [
    "/DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/DIMENSIONS",
    "DIMENSIONS",
]


def _h5_read_first(h5f: h5py.File, candidates: List[str]) -> Optional[np.ndarray]:
    """Try each candidate key; fall back to full-tree search by leaf name."""
    for key in candidates:
        if key in h5f:
            return h5f[key][:]

    # Brute-force search by leaf name
    leaf_names = [c.split("/")[-1] for c in candidates]
    found: List[np.ndarray] = []

    def _visitor(name: str, obj) -> Optional[str]:
        if isinstance(obj, h5py.Dataset):
            for leaf in leaf_names:
                if name.split("/")[-1] == leaf:
                    found.append(obj[:])
                    return name  # stop iteration
        return None

    h5f.visititems(_visitor)
    return found[0] if found else None


def read_h5(path: str) -> Dict[str, Optional[np.ndarray]]:
    """
    Read one EBSD h5 file.

    Returns
    -------
    dict with keys:
        euler  : (N, 3) float32  Euler angles in RADIANS (Bunge ZXZ)
        grains : (N,)  int32     grain / feature IDs
        phases : (N,)  int32     phase IDs
        dims   : (3,)  int32     [X, Y, Z] voxel dimensions
    Any unreadable field is None.
    """
    try:
        with h5py.File(path, "r") as f:
            euler  = _h5_read_first(f, _EULER_CANDIDATES)
            grains = _h5_read_first(f, _GRAIN_CANDIDATES)
            kam    = _h5_read_first(f, _KAM_CANDIDATES)
            phases = _h5_read_first(f, _PHASE_CANDIDATES)
            dims   = _h5_read_first(f, _DIM_CANDIDATES)

            # ── BUG-FIX: dim recovery must stay inside the `with` block ──────
            # Normalise euler shape first so len(euler) is available
            _euler_tmp = euler.reshape(-1, 3) if euler is not None else None
            _n_voxels  = len(_euler_tmp) if _euler_tmp is not None else 0

            if (dims is None or np.prod(dims) != _n_voxels) and _n_voxels > 0:
                try:
                    # Oxford / EDAX CTF-style header inside h5
                    if ("Scan 1/EBSD/Header/nColumns" in f and
                            "Scan 1/EBSD/Header/nRows" in f):
                        nx = int(np.array(f["Scan 1/EBSD/Header/nColumns"]).ravel()[0])
                        ny = int(np.array(f["Scan 1/EBSD/Header/nRows"]).ravel()[0])
                        if nx * ny == _n_voxels:
                            dims = np.array([nx, ny, 1], dtype=np.int32)
                    elif "Scan 1/EBSD/Header/Step X" in f:
                        pass   # step-size only; can't infer shape
                except Exception as _e:
                    log.debug("Dimension recovery failed: %s", _e)

        # ── Normalise shapes (file is now safely closed) ───────────────────
        if euler is not None:
            euler = euler.reshape(-1, 3).astype(np.float32)
            if euler.max() > 2 * np.pi + 0.1:
                euler = np.deg2rad(euler)

        if grains is not None:
            grains = grains.flatten().astype(np.int32)
        if kam is not None:
            kam = kam.flatten().astype(np.float32)
        if phases is not None:
            phases = phases.flatten().astype(np.int32)
        if dims is not None:
            dims = dims.flatten().astype(np.int32)

        return {
            "euler":  euler,
            "grains": grains,
            "kam":    kam,
            "phases": phases,
            "dims":   dims
        }

    except Exception as exc:
        log.debug("Failed to read %s: %s", path, exc)
        return {"euler": None, "grains": None, "phases": None, "dims": None}


# ═════════════════════════════════════════════════════════════════════════════
# 2.  POLE-FIGURE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def _euler_to_R(phi1: float, Phi: float, phi2: float) -> np.ndarray:
    """
    Bunge ZXZ rotation matrix (active convention), all angles in radians.
    Returns (3, 3) float64.
    """
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c,  s  = np.cos(Phi),  np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([
        [ c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s],
        [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
        [ s1*s,             -c1*s,             c   ],
    ], dtype=np.float64)


def _batch_euler_to_R(euler: np.ndarray) -> np.ndarray:
    """
    Vectorised Bunge ZXZ rotation matrices.

    Parameters
    ----------
    euler : (N, 3) float32 radians

    Returns
    -------
    R : (N, 3, 3) float64
    """
    phi1 = euler[:, 0].astype(np.float64)
    Phi  = euler[:, 1].astype(np.float64)
    phi2 = euler[:, 2].astype(np.float64)

    c1, s1 = np.cos(phi1), np.sin(phi1)
    c,  s  = np.cos(Phi),  np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    R = np.stack([
        np.stack([ c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s], axis=-1),
        np.stack([-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s], axis=-1),
        np.stack([ s1*s,             -c1*s,             c   ], axis=-1),
    ], axis=1)  # (N, 3, 3)
    return R


def _compute_pole_figure(euler: np.ndarray, hkl: np.ndarray,
                          n_grid: int = 224) -> np.ndarray:
    """
    Compute one stereographic pole figure.

    Parameters
    ----------
    euler  : (N, 3) float32 Euler angles in radians
    hkl    : (3,) direction in crystal frame, e.g. [1,0,0]
    n_grid : output image resolution

    Returns
    -------
    H : (n_grid, n_grid) float32  density map, values in [0, 1]
    """
    hkl = np.array(hkl, dtype=np.float64)
    hkl /= (np.linalg.norm(hkl) + 1e-12)

    R = _batch_euler_to_R(euler)         # (N, 3, 3)
    v = R.transpose(0, 2, 1) @ hkl      # (N, 3)  crystal → sample frame

    # Upper hemisphere
    neg = v[:, 2] < 0
    v[neg] = -v[neg]

    denom = 1.0 + v[:, 2] + 1e-9
    xs = v[:, 0] / denom
    ys = v[:, 1] / denom

    H, _, _ = np.histogram2d(xs, ys, bins=n_grid,
                              range=[[-1.0, 1.0], [-1.0, 1.0]])
    H = H.astype(np.float32)

    if _HAVE_SCIPY:
        H = _scipy_gauss(H, sigma=2.0)
    else:
        # Simple 3×3 box blur as fallback
        from numpy.lib.stride_tricks import sliding_window_view
        pad = np.pad(H, 1, mode="edge")
        windows = sliding_window_view(pad, (3, 3))
        H = windows.mean(axis=(-2, -1)).astype(np.float32)

    max_val = H.max()
    if max_val > 1e-9:
        H /= max_val
    return H


def generate_rgb_pole_figure(euler: np.ndarray, n_grid: int = 224,
                              subsample: int = 8000) -> np.ndarray:
    """
    Generate a 3-channel pole figure image (R={100}, G={110}, B={111}).

    Parameters
    ----------
    euler     : (N, 3) float32 Euler angles in radians
    n_grid    : output image size (pixels)
    subsample : max orientations to use (speed vs accuracy tradeoff)

    Returns
    -------
    img : (n_grid, n_grid, 3) float32  values in [0, 1]
    """
    if len(euler) > subsample:
        idx = np.random.choice(len(euler), subsample, replace=False)
        euler = euler[idx]

    pf100 = _compute_pole_figure(euler, [1, 0, 0], n_grid)
    pf110 = _compute_pole_figure(euler, [1, 1, 0], n_grid)
    pf111 = _compute_pole_figure(euler, [1, 1, 1], n_grid)
    return np.stack([pf100, pf110, pf111], axis=-1)  # (H, W, 3)


def save_pole_figure_png(euler: np.ndarray, save_path: str,
                          n_grid: int = 224, subsample: int = 8000) -> bool:
    """
    Generate and save a pole figure PNG.  Returns True on success.
    """
    if not _HAVE_PIL:
        raise RuntimeError("Pillow is required to save PNG images. "
                           "Install: pip install Pillow")
    try:
        img = generate_rgb_pole_figure(euler, n_grid=n_grid, subsample=subsample)
        img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        _PIL_Image.fromarray(img_uint8).save(save_path)
        return True
    except Exception as exc:
        log.debug("Failed to save pole figure %s: %s", save_path, exc)
        return False


# ═════════════════════════════════════════════════════════════════════════════
# 3.  LABEL COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_texture_index(euler: np.ndarray, n_bins: int = 36) -> float:
    """
    Texture index  J = ∫ f(g)² dg  approximated via ODF binning.
    Higher J means stronger texture (random → J ≈ 1).

    Parameters
    ----------
    euler  : (N, 3) float32 radians
    n_bins : histogram bins per Euler angle

    Returns
    -------
    J : float
    """
    phi1 = np.degrees(euler[:, 0]) % 360.0
    Phi  = np.degrees(euler[:, 1]) % 180.0
    phi2 = np.degrees(euler[:, 2]) % 360.0

    H, _ = np.histogramdd(
        np.stack([phi1, Phi, phi2], axis=1),
        bins=[n_bins, n_bins, n_bins],
        range=[[0, 360], [0, 180], [0, 360]],
        density=True,
    )
    J = float(np.sum(H ** 2) * (360.0 / n_bins) * (180.0 / n_bins) * (360.0 / n_bins))
    return J * (360.0 * 180.0 * 360.0)


def compute_average_misorientation(euler: np.ndarray,
                                    grain_ids: np.ndarray,
                                    max_grains: int = 200,
                                    pairs_per_grain: int = 30) -> float:
    """
    Approximate grain-average kernel average misorientation (KAM) in degrees.
    Samples random intra-grain orientation pairs.

    Parameters
    ----------
    euler          : (N, 3) float32 radians
    grain_ids      : (N,) int32
    max_grains     : limit computation to first N grains
    pairs_per_grain: random pairs sampled per grain

    Returns
    -------
    mean_misorientation_deg : float
    """
    unique_grains = np.unique(grain_ids)[:max_grains]
    misorientations: List[float] = []

    for gid in unique_grains:
        mask = grain_ids == gid
        ea = euler[mask]
        if len(ea) < 2:
            continue
        n = min(pairs_per_grain, len(ea))
        idx_a = np.random.randint(0, len(ea), n)
        idx_b = np.random.randint(0, len(ea), n)
        for ia, ib in zip(idx_a, idx_b):
            if ia == ib:
                continue
            g1 = _euler_to_R(*ea[ia])
            g2 = _euler_to_R(*ea[ib])
            M  = g1 @ g2.T
            tr = float(np.trace(M))
            cos_angle = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
            misorientations.append(float(np.degrees(np.arccos(cos_angle))))

    return float(np.mean(misorientations)) if misorientations else 0.0


def estimate_mean_grain_size(grain_ids: np.ndarray,
                              voxel_size_um: float = 1.0) -> float:
    """
    Mean equivalent-sphere-diameter grain size in microns.

    Parameters
    ----------
    grain_ids     : (N,) int32  grain IDs (0 = unindexed, skip)
    voxel_size_um : edge length of one voxel in microns

    Returns
    -------
    mean_diameter_um : float
    """
    unique, counts = np.unique(grain_ids, return_counts=True)
    # Remove background (ID 0 or negative)
    valid = unique > 0
    counts = counts[valid].astype(np.float64)
    volumes_um3 = counts * (voxel_size_um ** 3)
    # Equivalent sphere radius: V = (4/3) π r³
    radii_um = (3.0 * volumes_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(np.mean(radii_um) * 2.0)  # diameter


# ═════════════════════════════════════════════════════════════════════════════
# 4.  BATCH LABEL GENERATION  (cached to CSV)
# ═════════════════════════════════════════════════════════════════════════════

def generate_labels_csv(h5_dir: str,
                         csv_path: str,
                         voxel_size_um: float = 1.0,
                         compute_misorientation: bool = True,
                         force_recompute: bool = False) -> pd.DataFrame:
    """
    Iterate all .h5 files under ``h5_dir``, compute microstructure labels,
    and save a CSV.  If ``csv_path`` already exists and
    ``force_recompute`` is False the cached CSV is returned immediately.

    Columns in the output CSV
    ─────────────────────────
    sample_id            : str
    h5_path              : str
    grain_size           : float  (µm)
    texture_index        : float
    avg_misorientation   : float  (°)
    n_orientations       : int    (number of valid voxels)
    n_grains             : int
    """
    if os.path.exists(csv_path) and not force_recompute:
        log.info("Labels cache found at %s – loading.", csv_path)
        df = pd.read_csv(csv_path)
        log.info("  %d samples loaded.", len(df))
        return df

    h5_files = sorted(
        glob.glob(os.path.join(h5_dir, "**/*.h5"),  recursive=True) +
        glob.glob(os.path.join(h5_dir, "**/*.hdf5"), recursive=True) +
        glob.glob(os.path.join(h5_dir, "*.h5")) +
        glob.glob(os.path.join(h5_dir, "*.hdf5"))
    )
    h5_files = list(dict.fromkeys(h5_files))  # deduplicate

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 / .hdf5 files found under '{h5_dir}'. "
            "Check that 3D_EBSD_Dataset_Slices1to176_h5.zip has been extracted."
        )

    log.info("Found %d h5 files. Computing labels …", len(h5_files))
    records: List[Dict] = []

    for fpath in tqdm(h5_files, unit="file", desc="Labels"):
        sid = Path(fpath).stem
        try:
            data = read_h5(fpath)
            if data["euler"] is None:
                log.debug("No Euler angles in %s – skipping.", fpath)
                continue

            ea     = data["euler"]          # (N, 3) radians
            grains = data["grains"]
            n_ori  = len(ea)
            n_grn  = int(np.unique(grains[grains > 0]).size) if grains is not None else 0
            
            # If grains are missing, CGCNN will use pseudo-grains (estimated count)
            if n_grn == 0 and data["euler"] is not None:
                # Approximate number of pseudo-grains based on default grid size 8
                n_grn = 8 * 8 * 8  # 512

            grain_size   = float("nan")
            avg_misor    = float("nan")

            if grains is not None and len(grains) == len(ea):
                grain_size = estimate_mean_grain_size(grains, voxel_size_um)
                if compute_misorientation:
                    avg_misor = compute_average_misorientation(ea, grains)
            else:
                # ── BUG-FIX: estimate labels when FeatureIds are absent ────────
                # Raw EBSD files from the Dryad dataset do not contain pre-segmented
                # grain IDs.  We estimate grain_size from a pseudo-grid partition
                # (same 8×8×8 grid the CGCNN branch will use) and avg_misorientation
                # from random Euler-angle pairs across the whole volume.
                N = len(ea)
                grid_res = 8
                # Each pseudo-grain covers ~N/(grid_res³) voxels
                n_pseudo = float(grid_res ** 3)
                voxels_per_grain = max(1.0, N / n_pseudo)
                vol_um3  = voxels_per_grain * (voxel_size_um ** 3)
                radius   = (3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)
                grain_size = float(radius * 2.0)

                if data.get("kam") is not None:
                    avg_misor = float(np.mean(data["kam"]))
                elif compute_misorientation and N >= 2:
                    # Random-pair misorientation estimate (fast sample of 500 pairs)
                    n_pairs = min(500, N)
                    rng = np.random.default_rng(seed=42)
                    idx_a = rng.integers(0, N, n_pairs)
                    idx_b = rng.integers(0, N, n_pairs)
                    misor_list: List[float] = []
                    for ia, ib in zip(idx_a, idx_b):
                        if ia == ib:
                            continue
                        g1 = _euler_to_R(*ea[ia])
                        g2 = _euler_to_R(*ea[ib])
                        tr = float(np.trace(g1 @ g2.T))
                        cos_a = float(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
                        misor_list.append(float(np.degrees(np.arccos(cos_a))))
                    avg_misor = float(np.mean(misor_list)) if misor_list else 0.0

            tex_idx = compute_texture_index(ea)

            records.append({
                "sample_id":          sid,
                "h5_path":            fpath,
                "grain_size":         grain_size,
                "texture_index":      tex_idx,
                "avg_misorientation": avg_misor,
                "n_orientations":     n_ori,
                "n_grains":           n_grn,
            })

        except Exception:
            log.warning("Error processing %s:\n%s", fpath, traceback.format_exc())

    df = pd.DataFrame(records)
    df.dropna(subset=["texture_index"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info("Labels saved → %s  (%d samples)", csv_path, len(df))
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 5.  POLE-FIGURE IMAGE CACHE
# ═════════════════════════════════════════════════════════════════════════════

def build_pole_figure_cache(labels_df: pd.DataFrame,
                             cache_dir: str,
                             n_grid: int = 224,
                             subsample: int = 8000) -> pd.DataFrame:
    """
    For every row in ``labels_df``, generate a pole-figure PNG and store
    the path in a new column ``pole_fig_path``.  Already-existing files
    are skipped (idempotent).

    Parameters
    ----------
    labels_df : DataFrame with at least columns ``sample_id`` and ``h5_path``
    cache_dir : directory where PNGs are written
    n_grid    : pole-figure image size (pixels)
    subsample : max orientations per sample (speed vs accuracy)

    Returns
    -------
    DataFrame with ``pole_fig_path`` column added; rows that failed are dropped.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths: List[Optional[str]] = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df),
                       unit="sample", desc="Pole figures"):
        out = os.path.join(cache_dir, row["sample_id"] + "_pf.png")

        if os.path.exists(out):
            paths.append(out)
            continue

        try:
            data = read_h5(row["h5_path"])
            if data["euler"] is None:
                paths.append(None)
                continue
            ok = save_pole_figure_png(data["euler"], out,
                                      n_grid=n_grid, subsample=subsample)
            paths.append(out if ok else None)
        except Exception:
            log.debug("Pole figure failed for %s", row["sample_id"])
            paths.append(None)

    df = labels_df.copy()
    df["pole_fig_path"] = paths
    before = len(df)
    df.dropna(subset=["pole_fig_path"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info("Pole figures: %d/%d generated.", len(df), before)
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 6.  2-D EBSD IMAGE PAIRING
# ═════════════════════════════════════════════════════════════════════════════

def _normalise_stem(s: str) -> str:
    """Lower-case, strip common suffixes, collapse non-alphanumeric to '_'."""
    import re
    s = s.lower()
    for suffix in ("_ipf", "_ipf_x", "_ipf_z", "_ipf_y",
                   "_grainmap", "_grain_map", "_phase", "_ebsd"):
        s = s.replace(suffix, "")
    s = re.sub(r"[^a-z0-9]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def pair_microstructure_images(labels_df: pd.DataFrame,
                                micro_dir: str,
                                prefer_ipf: bool = True) -> pd.DataFrame:
    """
    Match each sample in ``labels_df`` to its 2-D EBSD image.

    Matching priority
    -----------------
    1. Exact stem match (case-insensitive)
    2. Normalised-stem match (strips common suffixes)
    3. Prefix match (first 6 characters)
    4. If multiple images exist for a sample and ``prefer_ipf`` is True,
       prefer filenames containing "ipf".

    Returns the DataFrame with ``micro_img_path`` added and
    unmatched rows dropped.
    """
    exts = ("*.png", "*.tif", "*.tiff", "*.bmp", "*.jpg", "*.jpeg")
    all_imgs = []
    for ext in exts:
        all_imgs.extend(
            glob.glob(os.path.join(micro_dir, "**", ext), recursive=True)
        )
    all_imgs = list(dict.fromkeys(all_imgs))

    if not all_imgs:
        raise FileNotFoundError(
            f"No images found under '{micro_dir}'. "
            "Check that 2D_EBSD_Dataset.zip has been extracted."
        )

    # Build lookup tables
    exact_map:  Dict[str, List[str]] = {}
    normal_map: Dict[str, List[str]] = {}
    for p in all_imgs:
        stem = Path(p).stem
        key_exact  = stem.lower()
        key_normal = _normalise_stem(stem)
        exact_map.setdefault(key_exact, []).append(p)
        normal_map.setdefault(key_normal, []).append(p)

    def _pick(candidates: List[str]) -> str:
        if len(candidates) == 1:
            return candidates[0]
        if prefer_ipf:
            ipf = [c for c in candidates if "ipf" in Path(c).stem.lower()]
            if ipf:
                return ipf[0]
        return candidates[0]

    micro_paths: List[Optional[str]] = []
    for _, row in labels_df.iterrows():
        sid = row["sample_id"]
        sid_lower  = sid.lower()
        sid_normal = _normalise_stem(sid)

        match = None
        if sid_lower in exact_map:
            match = _pick(exact_map[sid_lower])
        elif sid_normal in normal_map:
            match = _pick(normal_map[sid_normal])
        else:
            # Prefix match (first 6 chars)
            prefix = sid_normal[:6]
            for k, v in normal_map.items():
                if k.startswith(prefix):
                    match = _pick(v)
                    break
        micro_paths.append(match)

    df = labels_df.copy()
    df["micro_img_path"] = micro_paths
    before = len(df)
    df.dropna(subset=["micro_img_path"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    log.info("Microstructure images paired: %d/%d.", len(df), before)
    
    if len(df) == 0 and before > 0:
        log.error("CRITICAL: Zero microstructure images were paired with h5 samples.")
        log.error("Sample ID example: '%s'", labels_df.iloc[0]["sample_id"])
        log.error("Available image stems (first 5): %s", [Path(p).stem for p in all_imgs[:5]])
        log.error("Ensure that 2D EBSD images share the same stem as h5 files.")
        
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 7.  DATASET VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame,
                     img_col: str,
                     label_cols: List[str],
                     min_samples: int = 20) -> pd.DataFrame:
    """
    Validate a paired dataset DataFrame:
    - Drop rows where the image file does not exist on disk.
    - Drop rows with NaN in any label column.
    - Raise if fewer than ``min_samples`` remain.
    - Log per-label summary statistics.
    """
    # Verify image files exist
    exists_mask = df[img_col].apply(lambda p: os.path.isfile(str(p)))
    n_missing   = (~exists_mask).sum()
    if n_missing:
        log.warning("%d image files not found on disk – dropping.", n_missing)
    df = df[exists_mask].copy()

    # Drop NaN labels
    before = len(df)
    
    # Safety check for empty DF or missing columns to avoid KeyError
    if df.empty:
        log.warning("Dataset is empty – skipping NaN drop.")
    else:
        existing_cols = [c for c in label_cols if c in df.columns]
        missing_cols  = [c for c in label_cols if c not in df.columns]
        if missing_cols:
            log.warning("Some label columns missing from DataFrame: %s", missing_cols)
        
        if existing_cols:
            df.dropna(subset=existing_cols, inplace=True)
        
    n_nan = before - len(df)
    if n_nan:
        log.warning("Dropped %d rows with NaN labels.", n_nan)

    df.reset_index(drop=True, inplace=True)

    if len(df) < min_samples:
        raise RuntimeError(
            f"Only {len(df)} valid samples after validation "
            f"(minimum required: {min_samples}).\n"
            "Possible fixes:\n"
            "  • Verify that h5 files are extracted under the correct directory.\n"
            "  • Check that 2D EBSD images share the same sample IDs as h5 files.\n"
            "  • Lower min_samples in validate_dataset()."
        )

    log.info("Dataset validated: %d samples", len(df))
    for col in label_cols:
        v = df[col]
        log.info("  %-25s  min=%8.3f  max=%8.3f  mean=%8.3f  std=%6.3f",
                 col, v.min(), v.max(), v.mean(), v.std())
    return df
