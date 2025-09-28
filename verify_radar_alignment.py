"""Utility to validate the radar-to-radar warp grid used during training.

This script rebuilds the B→A warp grid with the exact parameters from
``ddp.py`` and generates a color overlay (A in red, warped B in blue)
so that alignment quality can be visually inspected.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ddp import (
    build_warp_grid,
    imread_rgb,
    compute_center_px,
    compute_mpp,
    RADAR_A_LATLON,
    RADAR_B_LATLON,
    RADAR_A_RANGE_KM,
    RADAR_B_RANGE_KM,
    RADAR_A_CENTER_OVERRIDE,
    RADAR_B_CENTER_OVERRIDE,
    LEFT_MASK_PX,
)

# PyProj is required for the azimuthal equidistant projection used in ddp.py.
from pyproj import Transformer  # noqa: F401  # Imported for parity with ddp.py dependencies


def _validate_image_path(path: str, label: str) -> str:
    """Ensure the provided image path exists before attempting to read it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} image not found: {path}")
    return path


def _tuple_from_cli(value: Optional[str], default: Tuple[float, float]) -> Tuple[float, float]:
    """Parse simple "lat,lon"/"cx,cy" CLI overrides."""
    if value is None:
        return default
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected value in the form 'val1,val2'.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Could not parse tuple value: {value}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify radar alignment by rebuilding the warp grid.")
    parser.add_argument("path_a", help="Path to the Radar A (target) RGB image used during training.")
    parser.add_argument("path_b", help="Path to the Radar B (source) RGB image used during training.")
    parser.add_argument(
        "--output",
        default="radar_alignment_overlay.png",
        help="Path to save the red/blue alignment overlay (default: %(default)s).",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Also save the decoded Radar A image and warped Radar B image for side-by-side comparison.",
    )
    parser.add_argument(
        "--radar-a-latlon",
        type=lambda v: _tuple_from_cli(v, RADAR_A_LATLON),
        default=None,
        help="Optional override for Radar A latitude/longitude (lat,lon). Defaults to values in ddp.py.",
    )
    parser.add_argument(
        "--radar-b-latlon",
        type=lambda v: _tuple_from_cli(v, RADAR_B_LATLON),
        default=None,
        help="Optional override for Radar B latitude/longitude (lat,lon). Defaults to values in ddp.py.",
    )
    parser.add_argument(
        "--radar-a-center",
        type=lambda v: _tuple_from_cli(v, RADAR_A_CENTER_OVERRIDE) if RADAR_A_CENTER_OVERRIDE else _tuple_from_cli(v, (0.0, 0.0)),
        default=None,
        help="Optional override for Radar A center in pixels (cx,cy).",
    )
    parser.add_argument(
        "--radar-b-center",
        type=lambda v: _tuple_from_cli(v, RADAR_B_CENTER_OVERRIDE) if RADAR_B_CENTER_OVERRIDE else _tuple_from_cli(v, (0.0, 0.0)),
        default=None,
        help="Optional override for Radar B center in pixels (cx,cy).",
    )
    parser.add_argument(
        "--range-a-km",
        type=float,
        default=RADAR_A_RANGE_KM,
        help="Radar A range in kilometers (default matches ddp.py).",
    )
    parser.add_argument(
        "--range-b-km",
        type=float,
        default=RADAR_B_RANGE_KM,
        help="Radar B range in kilometers (default matches ddp.py).",
    )
    parser.add_argument(
        "--left-mask-px",
        type=int,
        default=LEFT_MASK_PX,
        help="Number of masked pixels on the left side of each image (default matches ddp.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    radar_a_latlon = args.radar_a_latlon or RADAR_A_LATLON
    radar_b_latlon = args.radar_b_latlon or RADAR_B_LATLON

    center_override_a = None
    if args.radar_a_center is not None:
        center_override_a = args.radar_a_center
    elif RADAR_A_CENTER_OVERRIDE is not None:
        center_override_a = RADAR_A_CENTER_OVERRIDE

    center_override_b = None
    if args.radar_b_center is not None:
        center_override_b = args.radar_b_center
    elif RADAR_B_CENTER_OVERRIDE is not None:
        center_override_b = RADAR_B_CENTER_OVERRIDE

    print("Loading images...")
    img_a_np = imread_rgb(_validate_image_path(args.path_a, "Radar A"))
    img_b_np = imread_rgb(_validate_image_path(args.path_b, "Radar B"))

    shape_a = img_a_np.shape[:2]
    shape_b = img_b_np.shape[:2]
    print(f"Image A shape: {shape_a}")
    print(f"Image B shape: {shape_b}")

    print("Computing centers and meters-per-pixel metrics...")
    cx_a, cy_a = compute_center_px(*shape_a, args.left_mask_px, center_override_a)
    cx_b, cy_b = compute_center_px(*shape_b, args.left_mask_px, center_override_b)
    mpp_a = compute_mpp(*shape_a, args.left_mask_px, args.range_a_km)
    mpp_b = compute_mpp(*shape_b, args.left_mask_px, args.range_b_km)
    print(f"  Radar A center: ({cx_a:.2f}, {cy_a:.2f}), mpp: {mpp_a:.5f}")
    print(f"  Radar B center: ({cx_b:.2f}, {cy_b:.2f}), mpp: {mpp_b:.5f}")

    print("Building warp grid...")
    warp_grid_np, meta = build_warp_grid(
        shape_a=shape_a,
        shape_b=shape_b,
        left_mask_px=args.left_mask_px,
        radar_a_latlon=radar_a_latlon,
        radar_b_latlon=radar_b_latlon,
        range_a_km=args.range_a_km,
        range_b_km=args.range_b_km,
        center_override_a=center_override_a,
        center_override_b=center_override_b,
    )
    print("Warp grid metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    print("Warping Radar B image to Radar A frame...")
    img_b_tensor = torch.from_numpy(img_b_np).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    warp_grid_tensor = torch.from_numpy(warp_grid_np).unsqueeze(0)

    with torch.no_grad():
        warped_b_tensor = F.grid_sample(
            img_b_tensor,
            warp_grid_tensor,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    warped_b_np = (
        warped_b_tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 255.0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    print("Generating overlay visualization...")
    overlay = np.zeros_like(img_a_np)
    overlay[..., 0] = img_a_np[..., 0]
    overlay[..., 2] = warped_b_np[..., 2]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(overlay).save(args.output)
    print(f"Saved overlay visualization to {args.output}")

    if args.save_components:
        base, ext = os.path.splitext(args.output)
        a_path = f"{base}_A{ext or '.png'}"
        b_path = f"{base}_B_warped{ext or '.png'}"
        Image.fromarray(img_a_np).save(a_path)
        Image.fromarray(warped_b_np).save(b_path)
        print(f"Saved original Radar A image to {a_path}")
        print(f"Saved warped Radar B image to {b_path}")


if __name__ == "__main__":
    main()
