"""Inference script for radar cleaning TinyUNet model.

This utility mirrors the preprocessing steps used during training so that
predictions can be generated from raw radar RGB frames.  It assembles the
15-channel input tensor (6 intensity channels, 6 availability channels, and
3 helper channels) for a target timestamp, runs the trained TinyUNet model,
and optionally saves the probability mask and cleaned reflectivity maps.
"""
from __future__ import annotations

import argparse
import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

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


def _load_atlas(atlas_path: Optional[str], expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not atlas_path:
        return None
    atlas = _load_bool_mask_npz(atlas_path)
    if atlas.shape != expected_shape:
        raise ValueError(
            f"Atlas at '{atlas_path}' has shape {atlas.shape}, expected {expected_shape}."
        )
    return atlas


def _process_radar_source(
    paths: Sequence[Optional[str]],
    atlas: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    hsvp: HSVParams,
    left_mask_px: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    height, width = image_shape
    atlas_torch: Optional[torch.Tensor] = None
    if atlas is not None:
        atlas_torch = torch.from_numpy(atlas.astype(np.bool_)).to(device)

    intensity_channels: List[torch.Tensor] = []
    availability_channels: List[torch.Tensor] = []

    for path in paths:
        if path is None:
            intensity = torch.zeros((height, width), dtype=torch.float32, device=device)
            availability = torch.zeros((height, width), dtype=torch.float32, device=device)
        else:
            img_rgb = _ensure_rgb(path)
            _, inten_np = _weak_label_core(img_rgb, hsvp, left_mask_px, device=str(device))
            intensity = torch.from_numpy(inten_np).to(device=device, dtype=torch.float32)
            if intensity.shape != (height, width):
                raise ValueError(
                    f"Intensity shape {intensity.shape} from '{path}' does not match expected {(height, width)}."
                )
            if atlas_torch is not None:
                intensity = intensity.clone()
                intensity.masked_fill_(atlas_torch, 0.0)
            availability = torch.ones((height, width), dtype=torch.float32, device=device)
            if atlas_torch is not None:
                availability = availability.clone()
                availability.masked_fill_(atlas_torch, 0.0)
        intensity_channels.append(intensity)
        availability_channels.append(availability)

    return intensity_channels, availability_channels


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
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    if warp_path:
        path_obj = Path(warp_path)
        if path_obj.is_file():
            arr = np.load(path_obj)
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
            path_obj = Path(warp_path)
        if path_obj is not None:
            try:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                np.save(path_obj, grid_np)
            except Exception:
                pass

    grid_torch = torch.from_numpy(np.asarray(grid_np, dtype=np.float32))
    return grid_torch, meta


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


def run_inference(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
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

    if args.neighbor_minutes is None:
        args.neighbor_minutes = int(cfg.get("neighbor_minutes", NEIGHBOR_MINUTES))

    # Build timestamp indices and gather neighbors
    index_a = TimestampIndex(args.radar_a)
    index_b = TimestampIndex(args.radar_b)
    anchor_ts = args.timestamp
    paths_a = index_a.neighbor_triplet(anchor_ts, args.neighbor_minutes)
    paths_b = index_b.neighbor_triplet(anchor_ts, args.neighbor_minutes)

    if paths_a[1] is None and paths_b[1] is None:
        step = max(1, args.neighbor_minutes)
        tol_center = MIN_TOL_MINUTES
        shifted = False
        for delta in range(step, step * 5, step):
            for direction in (1, -1):
                candidate_ts = add_minutes(anchor_ts, delta * direction)
                has_center_a = index_a.find_within(candidate_ts, tol_center) is not None
                has_center_b = index_b.find_within(candidate_ts, tol_center) is not None
                if has_center_a or has_center_b:
                    anchor_ts = candidate_ts
                    paths_a = index_a.neighbor_triplet(anchor_ts, args.neighbor_minutes)
                    paths_b = index_b.neighbor_triplet(anchor_ts, args.neighbor_minutes)
                    shifted = True
                    break
            if shifted:
                print(
                    "Both radars missing center frame; "
                    f"shifted anchor from {args.timestamp} to {anchor_ts}"
                )
                break

    args.anchor_timestamp = anchor_ts

    # Determine native shapes per radar
    try:
        shape_a = _infer_image_shape(paths_a)
    except RuntimeError:
        shape_a = _infer_image_shape(paths_b)
    shape_b = _infer_image_shape(paths_b, fallback=shape_a)

    # Load atlases if provided
    atlas_a = _load_atlas(args.atlas_a, shape_a)
    atlas_b = _load_atlas(args.atlas_b, shape_b)

    inten_a, avail_a = _process_radar_source(paths_a, atlas_a, shape_a, hsvp, left_mask_px, device)
    inten_b, avail_b = _process_radar_source(paths_b, atlas_b, shape_b, hsvp, left_mask_px, device)

    radar_a_latlon = (float(args.radar_a_lat), float(args.radar_a_lon))
    radar_b_latlon = (float(args.radar_b_lat), float(args.radar_b_lon))
    radar_a_range_km = float(args.radar_a_range_km)
    radar_b_range_km = float(args.radar_b_range_km)
    center_override_a = args.radar_a_center or training_cfg.RADAR_A_CENTER_OVERRIDE
    center_override_b = args.radar_b_center or training_cfg.RADAR_B_CENTER_OVERRIDE

    warp_path = args.warp_cache_path or warp_cache_path()
    warp_grid, warp_meta = _load_or_build_warp_grid(
        shape_a,
        shape_b,
        left_mask_px,
        warp_path,
        radar_a_latlon,
        radar_b_latlon,
        radar_a_range_km,
        radar_b_range_km,
        center_override_a=center_override_a,
        center_override_b=center_override_b,
    )
    warp_grid = warp_grid.to(device=device, dtype=torch.float32)

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
        np.save(args.out_mask, prob_mask_np)
    if args.out_intensity:
        np.save(args.out_intensity, cleaned_intensity_np)
    if args.out_cleaned_image:
        save_cleaned_image(cleaned_intensity_np, args.out_cleaned_image)

    return prob_mask_np, cleaned_intensity_np


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

    prob_mask_np, cleaned_intensity_np = run_inference(args)
    anchor_ts = getattr(args, "anchor_timestamp", args.timestamp)
    message = (
        "Successfully generated prediction. Mask shape: "
        f"{prob_mask_np.shape}, intensity shape: {cleaned_intensity_np.shape}"
    )
    if anchor_ts != args.timestamp:
        message += f". Anchor timestamp adjusted to {anchor_ts}"
    if args.out_cleaned_image:
        message += f". Cleaned image saved to {args.out_cleaned_image}"
    print(message)


if __name__ == "__main__":
    main()
