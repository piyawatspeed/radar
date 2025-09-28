"""Inference script for radar cleaning TinyUNet model.

This utility mirrors the preprocessing steps used during training so that
predictions can be generated from raw radar RGB frames.  It assembles the
15-channel input tensor (6 intensity channels, 6 availability channels, and
3 helper channels) for a target timestamp, runs the trained TinyUNet model,
and optionally saves the probability mask, cleaned reflectivity maps,
georeferencing grids, assembled network inputs, and a JSON metadata summary.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from pyproj import Transformer

import ddp as training_cfg

from ddp import (
    LEFT_MASK_PX,
    MIN_TOL_MINUTES,
    NEIGHBOR_MINUTES,
    TOL_FRAC,
    HSVParams,
    TinyUNet,
    _load_bool_mask_npz,
    _weak_label_core,
    add_minutes,
    build_warp_grid,
    compute_center_px,
    compute_mpp,
    helper_channels,
    _fit_mask_to,
    list_images,
    warp_cache_path,
    _parse_center_tuple,
    parse_timestamp,
    ts_to_dt,
)


def _ensure_rgb(path: str) -> np.ndarray:
    """Read an image file as an RGB numpy array."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


DEFAULT_DATA_DIR = Path("data")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_ATLAS_DIR = Path("atlas")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CHECKPOINT_PATTERN = "*.pt"
DEFAULT_ATLAS_PATTERN = "*.npz"


def _minutes_diff(ts_a: int, ts_b: int) -> float:
    dt_a = ts_to_dt(ts_a)
    dt_b = ts_to_dt(ts_b)
    return abs((dt_a - dt_b).total_seconds() / 60.0)


class TimestampIndex:
    """Lightweight timestamp index for a radar source."""

    def __init__(self, root: str):
        paths = list_images(root)
        items: List[Tuple[int, str]] = []
        for path in paths:
            ts = parse_timestamp(path)
            if ts is not None:
                items.append((ts, path))
        if not items:
            raise RuntimeError(
                f"No timestamped images were found in '{root}'. "
                "Filenames must contain YYYYMMDDHHMM."
            )
        items.sort(key=lambda pair: pair[0])
        self.root = root
        self._items = items
        self.ts_sorted = [ts for ts, _ in items]
        self.ts_to_path = {ts: path for ts, path in items}

    def find_within(self, target_ts: int, tol_minutes: int) -> Optional[int]:
        """Return the timestamp closest to *target_ts* within the tolerance."""
        if target_ts in self.ts_to_path:
            return target_ts

        ts_sorted = self.ts_sorted
        idx = bisect.bisect_left(ts_sorted, target_ts)
        n = len(ts_sorted)
        best_ts: Optional[int] = None
        best_dt = float("inf")

        left = idx - 1
        right = idx

        while left >= 0 or right < n:
            advanced = False

            if right < n:
                ts = ts_sorted[right]
                dt_min = _minutes_diff(ts, target_ts)
                if dt_min <= tol_minutes:
                    advanced = True
                    if dt_min < best_dt:
                        best_dt = dt_min
                        best_ts = ts
                    right += 1
                else:
                    right = n

            if left >= 0:
                ts = ts_sorted[left]
                dt_min = _minutes_diff(ts, target_ts)
                if dt_min <= tol_minutes:
                    advanced = True
                    if dt_min < best_dt:
                        best_dt = dt_min
                        best_ts = ts
                    left -= 1
                else:
                    left = -1

            if not advanced:
                break

        return best_ts

    def neighbor_triplet(self, target_ts: int, neighbor_minutes: int = NEIGHBOR_MINUTES) -> List[Optional[str]]:
        """Return the (prev, current, next) image paths around *target_ts*."""
        offsets = (-neighbor_minutes, 0, neighbor_minutes)
        paths: List[Optional[str]] = []
        for offset in offsets:
            seek_ts = add_minutes(target_ts, offset)
            tol_base = neighbor_minutes if offset == 0 else abs(offset)
            tol = max(MIN_TOL_MINUTES, int(tol_base * TOL_FRAC))
            ts_match = self.find_within(seek_ts, tol)
            paths.append(self.ts_to_path.get(ts_match) if ts_match is not None else None)
        return paths

    def strict_triplet(self, target_ts: int, neighbor_minutes: int) -> Optional[List[str]]:
        """Return exact-offset triplet paths when every frame is present."""

        if neighbor_minutes <= 0:
            raise ValueError("neighbor_minutes must be positive for strict alignment.")

        offsets = (-neighbor_minutes, 0, neighbor_minutes)
        paths: List[str] = []
        for offset in offsets:
            seek_ts = add_minutes(target_ts, offset)
            path = self.ts_to_path.get(seek_ts)
            if path is None:
                return None
            paths.append(path)
        return paths


def _validate_dir(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} '{path}' does not exist.")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} '{path}' is not a directory.")
    return path


def _discover_radar_dirs(
    data_root: Path,
    radar_a_override: Optional[str],
    radar_b_override: Optional[str],
) -> Tuple[Path, Path]:
    data_root = _validate_dir(data_root, "Data root")

    def _resolve_override(override: Optional[str]) -> Optional[Path]:
        if override is None:
            return None
        path = Path(override)
        if not path.is_absolute():
            candidate = data_root / path
            path = candidate if candidate.exists() else path
        return _validate_dir(path, "Radar directory")

    resolved_a = _resolve_override(radar_a_override)
    resolved_b = _resolve_override(radar_b_override)

    remaining = [p for p in sorted(data_root.iterdir()) if p.is_dir()]

    def _pick_default(label: str, existing: Optional[Path]) -> Path:
        if existing is not None:
            return existing
        while remaining:
            candidate = remaining.pop(0)
            if candidate != resolved_a and candidate != resolved_b:
                return candidate
        raise RuntimeError(
            f"Unable to locate a default directory for Radar {label.upper()} in '{data_root}'."
        )

    radar_a = _pick_default("a", resolved_a)
    radar_b = _pick_default("b", resolved_b)
    return radar_a, radar_b


def _discover_checkpoint(checkpoint: Optional[str], checkpoint_dir: Path) -> Path:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.is_absolute():
            candidate = checkpoint_dir / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path

    checkpoint_dir = _validate_dir(checkpoint_dir, "Checkpoint directory")
    matches = sorted(checkpoint_dir.glob(DEFAULT_CHECKPOINT_PATTERN))
    if not matches:
        raise RuntimeError(
            f"No checkpoint matching '{DEFAULT_CHECKPOINT_PATTERN}' found in '{checkpoint_dir}'."
        )
    return matches[0]


def _discover_atlases(
    atlas_a: Optional[str],
    atlas_b: Optional[str],
    atlas_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    def _resolve(path_str: Optional[str]) -> Optional[Path]:
        if path_str is None:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            candidate = atlas_dir / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"Atlas '{path}' does not exist.")
        if not path.is_file():
            raise RuntimeError(f"Atlas '{path}' is not a file.")
        return path

    resolved_a = _resolve(atlas_a)
    resolved_b = _resolve(atlas_b)

    if resolved_a is not None or resolved_b is not None:
        return resolved_a, resolved_b

    if not atlas_dir.exists():
        return None, None
    if not atlas_dir.is_dir():
        raise NotADirectoryError(f"Atlas directory '{atlas_dir}' is not a directory.")

    matches = sorted(atlas_dir.glob(DEFAULT_ATLAS_PATTERN))
    if not matches:
        return None, None
    if len(matches) == 1:
        return matches[0], None
    return matches[0], matches[1]


def _infer_image_shape(
    paths: Sequence[Optional[str]],
    fallback: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int]:
    for path in paths:
        if path is None:
            continue
        with Image.open(path) as img:
            h, w = img.size[1], img.size[0]
            return h, w
    if fallback is not None:
        return fallback
    raise RuntimeError("Unable to determine image dimensions from provided paths.")


def _load_atlas(atlas_path: Optional[str]) -> Optional[np.ndarray]:
    if not atlas_path:
        return None
    return _load_bool_mask_npz(atlas_path)


def _pad_tensor(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int],
    *,
    mode: str = "reflect",
    value: float = 0.0,
) -> torch.Tensor:
    """Pad *tensor* to *target_shape* mirroring training-time behavior."""

    target_h, target_w = map(int, target_shape)
    h, w = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return tensor

    pad_mode = mode
    if pad_mode == "reflect" and (h <= 1 or w <= 1):
        pad_mode = "replicate"

    needs_threshold = tensor.dtype == torch.bool and pad_mode != "constant"
    work = tensor.to(torch.float32) if needs_threshold else tensor
    pad = (0, pad_w, 0, pad_h)
    work = work.unsqueeze(0).unsqueeze(0)
    if pad_mode == "constant":
        work = F.pad(work, pad, mode="constant", value=float(value))
    else:
        work = F.pad(work, pad, mode=pad_mode)
    work = work.squeeze(0).squeeze(0)
    if needs_threshold:
        work = work > 0.5
    return work


def _process_radar_source(
    paths: Sequence[Optional[str]],
    atlas: Optional[np.ndarray],
    hsvp: HSVParams,
    left_mask_px: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], Tuple[int, int], Optional[Tuple[int, int]], Optional[np.ndarray]]:
    entries: List[Optional[torch.Tensor]] = []
    shapes: List[Tuple[int, int]] = []
    base_shape: Optional[Tuple[int, int]] = None

    for path in paths:
        if path is None:
            entries.append(None)
            continue
        img_rgb = _ensure_rgb(path)
        _, inten_np = _weak_label_core(img_rgb, hsvp, left_mask_px, device=str(device))
        intensity = torch.from_numpy(inten_np).to(device=device, dtype=torch.float32)
        shape = (int(intensity.shape[0]), int(intensity.shape[1]))
        shapes.append(shape)
        if base_shape is None:
            base_shape = shape
        entries.append(intensity)

    if shapes:
        target_h = max(h for h, _ in shapes)
        target_w = max(w for _, w in shapes)
    elif atlas is not None:
        target_h, target_w = map(int, atlas.shape[:2])
    else:
        raise RuntimeError("No frames available to infer radar input shape.")

    target_shape = (int(target_h), int(target_w))

    atlas_fit: Optional[np.ndarray] = None
    atlas_torch: Optional[torch.Tensor] = None
    if atlas is not None:
        atlas_fit = _fit_mask_to(target_shape[0], target_shape[1], atlas)
        atlas_bool = np.asarray(atlas_fit, dtype=np.bool_)
        atlas_torch = torch.from_numpy(atlas_bool).to(device)

    intensity_channels: List[torch.Tensor] = []
    availability_channels: List[torch.Tensor] = []

    for entry in entries:
        if entry is None:
            intensity = torch.zeros(target_shape, dtype=torch.float32, device=device)
            availability = torch.zeros(target_shape, dtype=torch.float32, device=device)
        else:
            intensity = entry
            if intensity.shape != target_shape:
                intensity = _pad_tensor(intensity, target_shape, mode="replicate")
            if atlas_torch is not None:
                intensity = intensity.clone()
                intensity.masked_fill_(atlas_torch, 0.0)
            availability = torch.ones(target_shape, dtype=torch.float32, device=device)
            if atlas_torch is not None:
                availability = availability.clone()
                availability.masked_fill_(atlas_torch, 0.0)
        intensity_channels.append(intensity)
        availability_channels.append(availability)

    return intensity_channels, availability_channels, target_shape, base_shape, atlas_fit


def _compute_warp_meta(
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int],
    left_mask_px: int,
    radar_a_range_km: float,
    radar_b_range_km: float,
    center_override_a: Optional[Tuple[float, float]] = None,
    center_override_b: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    h_a, w_a = map(int, shape_a)
    h_b, w_b = map(int, shape_b)
    cx_a, cy_a = compute_center_px(h_a, w_a, left_mask_px, center_override_a)
    cx_b, cy_b = compute_center_px(h_b, w_b, left_mask_px, center_override_b)
    mpp_a = compute_mpp(h_a, w_a, left_mask_px, radar_a_range_km)
    mpp_b = compute_mpp(h_b, w_b, left_mask_px, radar_b_range_km)
    return {
        "cxA": float(cx_a),
        "cyA": float(cy_a),
        "cxB": float(cx_b),
        "cyB": float(cy_b),
        "mpp_A": float(mpp_a),
        "mpp_B": float(mpp_b),
        "W_eff_A": float(w_a - left_mask_px),
        "W_eff_B": float(w_b - left_mask_px),
    }


def _build_georef_grid(
    shape: Tuple[int, int],
    left_mask_px: int,
    radar_latlon: Tuple[float, float],
    radar_range_km: float,
    center_override: Optional[Tuple[float, float]] = None,
    *,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate latitude/longitude grids for Radar A pixels."""

    height, width = map(int, shape)
    cx, cy = compute_center_px(height, width, left_mask_px, center_override)
    mpp = compute_mpp(height, width, left_mask_px, radar_range_km)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x_m = (xx - cx).astype(np.float32) * mpp
    y_m = (cy - yy).astype(np.float32) * mpp

    proj = Transformer.from_crs(
        f"+proj=aeqd +lat_0={radar_latlon[0]} +lon_0={radar_latlon[1]} +datum=WGS84",
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat = proj.transform(x_m, y_m)

    return lat.astype(dtype, copy=False), lon.astype(dtype, copy=False)


def _normalize_warp_cache_path(path: str) -> Path:
    """Ensure the warp cache uses an .npy suffix so load/save agree."""

    path_obj = Path(path)
    if path_obj.suffix.lower() != ".npy":
        path_obj = Path(f"{path_obj}.npy")
    return path_obj


def _load_or_build_warp_grid(
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int],
    left_mask_px: int,
    warp_path: Optional[str],
    radar_a_latlon: Tuple[float, float],
    radar_b_latlon: Tuple[float, float],
    radar_a_range_km: float,
    radar_b_range_km: float,
    center_override_a: Optional[Tuple[float, float]] = None,
    center_override_b: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    shape_a = tuple(map(int, shape_a))
    shape_b = tuple(map(int, shape_b))
    meta = _compute_warp_meta(
        shape_a,
        shape_b,
        left_mask_px,
        radar_a_range_km,
        radar_b_range_km,
        center_override_a=center_override_a,
        center_override_b=center_override_b,
    )

    grid_np: Optional[np.ndarray] = None
    path_obj: Optional[Path] = None
    cache_hit = False
    if warp_path:
        path_obj = _normalize_warp_cache_path(warp_path)
        if path_obj.is_file():
            arr = np.load(path_obj, allow_pickle=False)
            grid_np = np.asarray(arr, dtype=np.float32)
            if grid_np.ndim != 3 or grid_np.shape[2] != 2:
                grid_np = None
            elif grid_np.shape[0] != shape_a[0] or grid_np.shape[1] != shape_a[1]:
                grid_np, _ = build_warp_grid(
                    shape_a,
                    shape_b,
                    left_mask_px,
                    radar_a_latlon,
                    radar_b_latlon,
                    radar_a_range_km,
                    radar_b_range_km,
                    center_override_a=center_override_a,
                    center_override_b=center_override_b,
                )
                try:
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path_obj, grid_np)
                except Exception:
                    pass
            else:
                grid_np = grid_np.astype(np.float32, copy=False)
                cache_hit = True

    if grid_np is None:
        grid_np, _ = build_warp_grid(
            shape_a,
            shape_b,
            left_mask_px,
            radar_a_latlon,
            radar_b_latlon,
            radar_a_range_km,
            radar_b_range_km,
            center_override_a=center_override_a,
            center_override_b=center_override_b,
        )
        if path_obj is None and warp_path:
            path_obj = _normalize_warp_cache_path(warp_path)
        if path_obj is not None:
            try:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                np.save(path_obj, grid_np)
            except Exception:
                pass

    warp_info = {
        "path": str(path_obj) if path_obj is not None else None,
        "cache_hit": bool(cache_hit),
    }
    grid_torch = torch.from_numpy(np.asarray(grid_np, dtype=np.float32))
    return grid_torch, meta, warp_info


def _resize_warp_grid(grid: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Resize a base warp grid to *target_shape* (height, width)."""

    target_h, target_w = map(int, target_shape)
    if grid.ndim != 3 or grid.shape[-1] != 2:
        raise ValueError(f"Warp grid must have shape (H, W, 2); got {tuple(grid.shape)}")
    if grid.shape[0] == target_h and grid.shape[1] == target_w:
        return grid

    base = grid.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(base, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).contiguous()


def _intensity_to_radar_rgb(cleaned_intensity: np.ndarray) -> np.ndarray:
    """Convert normalized intensities into the provided discrete radar palette."""

    if cleaned_intensity.ndim != 2:
        raise ValueError(
            "Cleaned intensity must be a 2D array to convert to RGB; "
            f"got shape {cleaned_intensity.shape}."
        )

    scaled = np.clip(cleaned_intensity, 0.0, 1.0)

    # Convert to dBZ assuming the palette maxes out at 76.67 dBZ.
    dbz_max = 76.67
    dbz_values = scaled.reshape(-1) * dbz_max

    # Discrete color bins supplied by the user. Each tuple is (high_edge, (R, G, B)).
    color_bins = np.array([11.76, 34.09, 53.64, 55.29, 55.99, 75.28], dtype=np.float32)
    colors = np.array(
        [
            (8, 9, 235),    # 9.5-11.76 dBZ
            (12, 198, 18),  # 12.37-34.09 dBZ
            (230, 176, 12), # 34.09-53.64 dBZ
            (232, 8, 9),    # 53.64-55.29 dBZ
            (229, 7, 27),   # 55.29-55.99 dBZ
            (231, 85, 178), # 55.99-75.28 dBZ
            (249, 249, 248) # 75.88-76.67 dBZ
        ],
        dtype=np.float32,
    )

    # Assign each pixel to a palette color.
    indices = np.digitize(dbz_values, color_bins, right=True)
    indices = np.clip(indices, 0, len(colors) - 1)
    rgb = colors[indices].reshape((*scaled.shape, 3))
    return rgb.astype(np.uint8)


def save_cleaned_image(cleaned_intensity: np.ndarray, out_path: str) -> None:
    """Persist the cleaned intensity map as a radar-style color image."""

    rgb = _intensity_to_radar_rgb(cleaned_intensity)
    image = Image.fromarray(rgb, mode="RGB")
    image.save(out_path)


def _apply_weak_label_config(cfg: Dict[str, Any]) -> HSVParams:
    """Update ddp weak label globals and return HSV params to mirror training."""

    mode = cfg.get("mode")
    if mode:
        training_cfg.WEAK_LABEL_MODE = mode

    multi = cfg.get("multi_ranges")
    if multi:
        training_cfg.HSV_MULTI_RANGES = tuple((float(lo), float(hi)) for lo, hi in multi)

    defaults = HSVParams()
    simple_cfg = cfg.get("simple", {})
    hue_lo = float(simple_cfg.get("hue_lo", defaults.hue_lo))
    hue_hi = float(simple_cfg.get("hue_hi", defaults.hue_hi))
    sat_min = float(simple_cfg.get("sat_min", defaults.sat_min))
    val_min = float(simple_cfg.get("val_min", defaults.val_min))
    training_cfg.HSV_H_LO = hue_lo
    training_cfg.HSV_H_HI = hue_hi
    training_cfg.HSV_S_MIN = sat_min
    training_cfg.HSV_V_MIN = val_min
    return HSVParams(hue_lo=hue_lo, hue_hi=hue_hi, sat_min=sat_min, val_min=val_min)


def _paths_dict(paths: Sequence[Optional[str]]) -> Dict[str, Optional[str]]:
    keys = ("prev", "current", "next")
    return {k: (v if v is None else str(v)) for k, v in zip(keys, paths)}


def run_inference(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    def _ensure_parent(path_str: Optional[str]) -> None:
        if path_str:
            Path(path_str).expanduser().parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        state_dict = checkpoint
    def _strip_prefix(sd: Any, prefix: str):
        if not isinstance(sd, dict) or not sd:
            return sd
        keys = list(sd.keys())
        if all(k.startswith(prefix) for k in keys):
            return type(sd)((k[len(prefix):], v) for k, v in sd.items())
        return sd

    # Handle wrappers such as torch.compile that prepend "_orig_mod." or DDP's "module.".
    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")

    cfg: Dict[str, Any] = checkpoint.get("cfg") or checkpoint.get("config") or {}

    model_base = args.model_base if args.model_base is not None else cfg.get("base", 32)
    args.model_base = model_base
    in_ch = int(cfg.get("in_ch", 15))
    if in_ch != 15:
        raise ValueError(f"Unsupported checkpoint input channels {in_ch}; expected 15.")

    left_mask_px = int(cfg.get("left_mask_px", LEFT_MASK_PX))
    weak_cfg = cfg.get("weak_label", {})
    hsvp = _apply_weak_label_config(weak_cfg)

    fuse_mode = str(cfg.get("fuse_mode", "max"))
    args.fuse_mode = fuse_mode

    if args.neighbor_minutes is None:
        args.neighbor_minutes = int(cfg.get("neighbor_minutes", NEIGHBOR_MINUTES))

    # Build timestamp indices and gather neighbors with strict alignment
    index_a = TimestampIndex(args.radar_a)
    index_b = TimestampIndex(args.radar_b)

    step = int(args.neighbor_minutes)
    if step <= 0:
        raise ValueError("neighbor_minutes must be positive.")

    def _strict_bundle(center_ts: int) -> Optional[Tuple[List[str], List[str]]]:
        paths_a = index_a.strict_triplet(center_ts, step)
        if paths_a is None:
            return None
        paths_b = index_b.strict_triplet(center_ts, step)
        if paths_b is None:
            return None
        return paths_a, paths_b

    def _missing_offsets(idx: TimestampIndex, center_ts: int) -> List[Tuple[int, int]]:
        offsets = (-step, 0, step)
        missing: List[Tuple[int, int]] = []
        for offset in offsets:
            seek_ts = add_minutes(center_ts, offset)
            if idx.ts_to_path.get(seek_ts) is None:
                missing.append((offset, seek_ts))
        return missing

    def _format_missing(label: str, missing: Sequence[Tuple[int, int]]) -> Optional[str]:
        if not missing:
            return None
        formatted = []
        for offset, ts_val in missing:
            dt = ts_to_dt(ts_val)
            formatted.append(f"{offset:+d}m ({dt.strftime('%Y-%m-%d %H:%M')})")
        return f"Radar {label} missing {', '.join(formatted)}"

    anchor_ts = args.timestamp
    bundle = _strict_bundle(anchor_ts)

    if bundle is None:
        visited = {anchor_ts}
        found: Optional[Tuple[List[str], List[str]]] = None
        for delta in range(step, step * 5, step):
            for direction in (1, -1):
                candidate_ts = add_minutes(anchor_ts, delta * direction)
                if candidate_ts in visited:
                    continue
                visited.add(candidate_ts)
                candidate_bundle = _strict_bundle(candidate_ts)
                if candidate_bundle is not None:
                    found = candidate_bundle
                    anchor_ts = candidate_ts
                    print(
                        "Shifted anchor to maintain strict neighbor alignment: "
                        f"{args.timestamp} -> {anchor_ts}"
                    )
                    break
            if found is not None:
                break
        bundle = found

    if bundle is None:
        missing_msgs = []
        msg_a = _format_missing("A", _missing_offsets(index_a, args.timestamp))
        msg_b = _format_missing("B", _missing_offsets(index_b, args.timestamp))
        if msg_a:
            missing_msgs.append(msg_a)
        if msg_b:
            missing_msgs.append(msg_b)
        detail = "; ".join(missing_msgs) if missing_msgs else "no synchronized triplets located"
        raise RuntimeError(
            f"Unable to locate strict +/-{step} minute triplets around {args.timestamp}: {detail}."
        )

    paths_a, paths_b = bundle

    args.anchor_timestamp = anchor_ts

    # Load atlases and process radar channels mirroring training-time padding
    atlas_a_raw = _load_atlas(args.atlas_a)
    atlas_b_raw = _load_atlas(args.atlas_b)

    (
        inten_a,
        avail_a,
        shape_a,
        base_shape_a,
        atlas_a_fit,
    ) = _process_radar_source(paths_a, atlas_a_raw, hsvp, left_mask_px, device)
    (
        inten_b,
        avail_b,
        shape_b,
        base_shape_b,
        atlas_b_fit,
    ) = _process_radar_source(paths_b, atlas_b_raw, hsvp, left_mask_px, device)

    radar_a_latlon = (float(args.radar_a_lat), float(args.radar_a_lon))
    radar_b_latlon = (float(args.radar_b_lat), float(args.radar_b_lon))
    radar_a_range_km = float(args.radar_a_range_km)
    radar_b_range_km = float(args.radar_b_range_km)
    center_override_a = args.radar_a_center or training_cfg.RADAR_A_CENTER_OVERRIDE
    center_override_b = args.radar_b_center or training_cfg.RADAR_B_CENTER_OVERRIDE

    base_shape_a = tuple(int(x) for x in (base_shape_a or shape_a))
    base_shape_b = tuple(int(x) for x in (base_shape_b or shape_b))
    shape_a = tuple(int(x) for x in shape_a)
    shape_b = tuple(int(x) for x in shape_b)

    warp_path = args.warp_cache_path or warp_cache_path()
    warp_grid, warp_meta, warp_info = _load_or_build_warp_grid(
        base_shape_a,
        base_shape_b,
        left_mask_px,
        warp_path,
        radar_a_latlon,
        radar_b_latlon,
        radar_a_range_km,
        radar_b_range_km,
        center_override_a=center_override_a,
        center_override_b=center_override_b,
    )
    if warp_info["path"] is not None:
        warp_path = warp_info["path"]
        args.warp_cache_path = warp_path
    warp_grid = warp_grid.to(device=device, dtype=torch.float32)
    warp_grid = _resize_warp_grid(warp_grid, shape_a)
    warp_meta = dict(
        warp_meta,
        base_shape_A=[int(base_shape_a[0]), int(base_shape_a[1])],
        base_shape_B=[int(base_shape_b[0]), int(base_shape_b[1])],
    )

    def _warp_channels(channels: List[torch.Tensor], mode: str) -> List[torch.Tensor]:
        if not channels:
            return channels
        stack = torch.stack(channels, dim=0).unsqueeze(1)
        grid = warp_grid.unsqueeze(0)
        if grid.shape[0] != stack.shape[0]:
            grid = grid.expand(stack.shape[0], -1, -1, -1)
        warped = F.grid_sample(stack, grid, mode=mode, padding_mode="zeros", align_corners=False)
        return [warped[i, 0].contiguous() for i in range(warped.shape[0])]

    inten_b = _warp_channels(inten_b, mode="bilinear")
    avail_b = _warp_channels(avail_b, mode="bilinear")
    args.warp_meta = warp_meta

    all_channels: List[torch.Tensor] = []
    all_channels.extend(inten_a)
    all_channels.extend(inten_b)
    all_channels.extend(avail_a)
    all_channels.extend(avail_b)

    helper_center = compute_center_px(shape_a[0], shape_a[1], left_mask_px, center_override_a)
    r_norm, cos_t, sin_t = helper_channels(*shape_a, center=helper_center)
    all_channels.append(torch.from_numpy(r_norm).to(device=device, dtype=torch.float32))
    all_channels.append(torch.from_numpy(cos_t).to(device=device, dtype=torch.float32))
    all_channels.append(torch.from_numpy(sin_t).to(device=device, dtype=torch.float32))

    input_tensor = torch.stack(all_channels, dim=0).unsqueeze(0)
    input_numpy: Optional[np.ndarray] = None
    if getattr(args, "out_input", None):
        input_numpy = input_tensor.detach().cpu().numpy()
    input_tensor = input_tensor.to(device=device).contiguous(memory_format=torch.channels_last)

    # Load model
    model = TinyUNet(in_ch=in_ch, base=model_base)
    model.load_state_dict(state_dict)
    model = model.to(device).to(memory_format=torch.channels_last)
    model.eval()

    with torch.no_grad():
        logits_mask, dbz_pred = model(input_tensor)
        prob_mask = torch.sigmoid(logits_mask)
        cleaned_intensity = torch.clamp(dbz_pred, 0.0, 1.0)

    prob_mask_np = prob_mask.squeeze().detach().cpu().numpy()
    cleaned_intensity_np = cleaned_intensity.squeeze().detach().cpu().numpy()

    if args.out_mask:
        _ensure_parent(args.out_mask)
        np.save(args.out_mask, prob_mask_np)
    if args.out_intensity:
        _ensure_parent(args.out_intensity)
        np.save(args.out_intensity, cleaned_intensity_np)
    if args.out_cleaned_image:
        _ensure_parent(args.out_cleaned_image)
        save_cleaned_image(cleaned_intensity_np, args.out_cleaned_image)

    georef_latlon: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if getattr(args, "out_georef", None):
        _ensure_parent(args.out_georef)
        georef_latlon = _build_georef_grid(
            shape_a,
            left_mask_px,
            radar_a_latlon,
            radar_a_range_km,
            center_override=center_override_a,
        )
        lat_grid, lon_grid = georef_latlon
        np.savez(args.out_georef, lat=lat_grid, lon=lon_grid)

    metadata: Dict[str, Any] = {
        "timestamp": int(args.timestamp),
        "anchor_timestamp": int(anchor_ts),
        "neighbor_minutes": int(args.neighbor_minutes),
        "strict_neighbors": True,
        "fuse_mode": fuse_mode,
        "checkpoint": str(args.checkpoint),
        "model_base": int(model_base),
        "input_channels": int(in_ch),
        "left_mask_px": int(left_mask_px),
        "radar_a": {
            "root": str(args.radar_a),
            "paths": _paths_dict(paths_a),
            "shape": [int(shape_a[0]), int(shape_a[1])],
            "base_shape": [int(base_shape_a[0]), int(base_shape_a[1])],
            "atlas": {
                "path": args.atlas_a,
                "applied": bool(atlas_a_fit is not None),
                "coverage": float(np.mean(atlas_a_fit)) if atlas_a_fit is not None else None,
            },
            "latlon": list(map(float, radar_a_latlon)),
            "range_km": float(radar_a_range_km),
        },
        "radar_b": {
            "root": str(args.radar_b),
            "paths": _paths_dict(paths_b),
            "shape": [int(shape_b[0]), int(shape_b[1])],
            "base_shape": [int(base_shape_b[0]), int(base_shape_b[1])],
            "atlas": {
                "path": args.atlas_b,
                "applied": bool(atlas_b_fit is not None),
                "coverage": float(np.mean(atlas_b_fit)) if atlas_b_fit is not None else None,
            },
            "latlon": list(map(float, radar_b_latlon)),
            "range_km": float(radar_b_range_km),
        },
        "geometry": {
            "helper_center": [float(helper_center[0]), float(helper_center[1])],
            "meters_per_pixel": {
                "radar_a": float(compute_mpp(base_shape_a[0], base_shape_a[1], left_mask_px, radar_a_range_km)),
                "radar_b": float(compute_mpp(base_shape_b[0], base_shape_b[1], left_mask_px, radar_b_range_km)),
            },
            "warp": dict(warp_meta, **{"source": "cache" if warp_info["cache_hit"] else "computed", "path": warp_info["path"]}),
        },
        "warp_cache_path": warp_path,
        "weak_label": {
            "mode": training_cfg.WEAK_LABEL_MODE,
            "hsv": {
                "hue_lo": float(hsvp.hue_lo),
                "hue_hi": float(hsvp.hue_hi),
                "sat_min": float(hsvp.sat_min),
                "val_min": float(hsvp.val_min),
            },
            "multi_ranges": [(float(lo), float(hi)) for (lo, hi) in training_cfg.HSV_MULTI_RANGES],
        },
    }

    if georef_latlon is not None:
        lat_grid, lon_grid = georef_latlon
        metadata["georef"] = {
            "crs": "EPSG:4326",
            "lat_range": [float(lat_grid.min()), float(lat_grid.max())],
            "lon_range": [float(lon_grid.min()), float(lon_grid.max())],
            "shape": [int(lat_grid.shape[0]), int(lat_grid.shape[1])],
        }

    if getattr(args, "out_metadata", None):
        _ensure_parent(args.out_metadata)
        with open(args.out_metadata, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, sort_keys=True)

    if input_numpy is not None and getattr(args, "out_input", None):
        _ensure_parent(args.out_input)
        np.save(args.out_input, input_numpy)

    extras: Dict[str, Any] = {"metadata": metadata}
    if input_numpy is not None:
        extras["input_tensor"] = input_numpy
    if georef_latlon is not None:
        extras["georef"] = {"lat": georef_latlon[0], "lon": georef_latlon[1]}

    return prob_mask_np, cleaned_intensity_np, extras


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TinyUNet radar inference.")
    parser.add_argument("timestamp", type=int, help="Target timestamp YYYYMMDDHHMM.")
    parser.add_argument("--radar-a", help="Directory with Radar A frames (default: auto-detected in data root).")
    parser.add_argument("--radar-b", help="Directory with Radar B frames (default: auto-detected in data root).")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_DIR), help="Root directory containing radar data.")
    parser.add_argument("--checkpoint", help="Path to the trained model checkpoint (.pt).")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory to search for a checkpoint when --checkpoint is not provided.",
    )
    parser.add_argument("--atlas-a", help="Optional atlas mask (.npz) for Radar A.")
    parser.add_argument("--atlas-b", help="Optional atlas mask (.npz) for Radar B.")
    parser.add_argument(
        "--atlas-dir",
        default=str(DEFAULT_ATLAS_DIR),
        help="Directory to search for atlas files when --atlas-a/--atlas-b are not provided.",
    )
    parser.add_argument(
        "--warp-cache-path",
        help="Path to a cached Radar B→A warp grid (.npy). Defaults to the training cache location if omitted.",
    )
    parser.add_argument("--radar-a-lat", type=float, default=training_cfg.RADAR_A_LATLON[0], help="Radar A latitude (degrees).")
    parser.add_argument("--radar-a-lon", type=float, default=training_cfg.RADAR_A_LATLON[1], help="Radar A longitude (degrees).")
    parser.add_argument("--radar-b-lat", type=float, default=training_cfg.RADAR_B_LATLON[0], help="Radar B latitude (degrees).")
    parser.add_argument("--radar-b-lon", type=float, default=training_cfg.RADAR_B_LATLON[1], help="Radar B longitude (degrees).")
    parser.add_argument(
        "--radar-a-range-km",
        type=float,
        default=training_cfg.RADAR_A_RANGE_KM,
        help="Radar A range to the far edge (km).",
    )
    parser.add_argument(
        "--radar-b-range-km",
        type=float,
        default=training_cfg.RADAR_B_RANGE_KM,
        help="Radar B range to the far edge (km).",
    )
    parser.add_argument(
        "--radar-a-center",
        type=_parse_center_tuple,
        default=training_cfg.RADAR_A_CENTER_OVERRIDE,
        help="Override Radar A center pixel as 'cx,cy' (default: training configuration).",
    )
    parser.add_argument(
        "--radar-b-center",
        type=_parse_center_tuple,
        default=training_cfg.RADAR_B_CENTER_OVERRIDE,
        help="Override Radar B center pixel as 'cx,cy' (default: training configuration).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--neighbor-minutes",
        type=int,
        default=None,
        help="Neighbor spacing in minutes (default: value saved in checkpoint or training default).",
    )
    parser.add_argument(
        "--model-base",
        type=int,
        default=None,
        help="Base channel width (default: value saved in checkpoint).",
    )
    parser.add_argument("--out-mask", help="Output .npy path for probability mask (default: outputs/mask_<ts>.npy).")
    parser.add_argument(
        "--out-intensity",
        help="Output .npy path for cleaned intensity (default: outputs/intensity_<ts>.npy).",
    )
    parser.add_argument(
        "--out-cleaned-image",
        help="Output image file (default: outputs/cleaned_<ts>.png) for the cleaned intensity map.",
    )
    parser.add_argument(
        "--out-metadata",
        help="Optional JSON file to store preprocessing and georeference metadata (default: outputs/meta_<ts>.json).",
    )
    parser.add_argument(
        "--out-georef",
        help="Optional .npz file to store 'lat' and 'lon' georeference grids (Radar A frame).",
    )
    parser.add_argument(
        "--out-input",
        help="Optional .npy file to dump the assembled 15-channel network input tensor.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to place default outputs when explicit paths are not provided.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    timestamp = args.timestamp

    data_root = Path(args.data_root)
    radar_a_dir, radar_b_dir = _discover_radar_dirs(data_root, args.radar_a, args.radar_b)
    args.radar_a = str(radar_a_dir)
    args.radar_b = str(radar_b_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = _discover_checkpoint(args.checkpoint, checkpoint_dir)
    args.checkpoint = str(checkpoint_path)

    atlas_dir = Path(args.atlas_dir)
    atlas_a_path, atlas_b_path = _discover_atlases(args.atlas_a, args.atlas_b, atlas_dir)
    args.atlas_a = str(atlas_a_path) if atlas_a_path is not None else None
    args.atlas_b = str(atlas_b_path) if atlas_b_path is not None else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.out_mask is None:
        args.out_mask = str(output_dir / f"mask_{timestamp}.npy")
    if args.out_intensity is None:
        args.out_intensity = str(output_dir / f"intensity_{timestamp}.npy")
    if args.out_cleaned_image is None:
        args.out_cleaned_image = str(output_dir / f"cleaned_{timestamp}.png")
    if args.out_metadata is None:
        args.out_metadata = str(output_dir / f"meta_{timestamp}.json")

    prob_mask_np, cleaned_intensity_np, _ = run_inference(args)
    anchor_ts = getattr(args, "anchor_timestamp", args.timestamp)
    message = (
        "Successfully generated prediction. Mask shape: "
        f"{prob_mask_np.shape}, intensity shape: {cleaned_intensity_np.shape}"
    )
    if anchor_ts != args.timestamp:
        message += f". Anchor timestamp adjusted to {anchor_ts}"
    if args.out_cleaned_image:
        message += f". Cleaned image saved to {args.out_cleaned_image}"
    if args.out_metadata:
        message += f". Metadata saved to {args.out_metadata}"
    if args.out_georef:
        message += f". Georeference grid saved to {args.out_georef}"
    if args.out_input:
        message += f". Input tensor saved to {args.out_input}"
    print(message)


if __name__ == "__main__":
    main()
