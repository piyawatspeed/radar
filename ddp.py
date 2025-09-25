# ddp.py
# All-in-one training script (single L4 default, DDP-ready) — strict cache reuse + fast RGB cpu-prep
# - Two stages:
#   * cpu-prep: CPU-only indexing + optional cache precompute (weak-labels, RGB npy) [fast I/O mode supported]
#   * gpu-train: training with strict cache reuse (no weak-label recompute), AMP, optional DDP
# - Strict knobs:
#   * --strict-cache-only  -> never recompute weak labels; error on miss
#   * --strict-atlas-only  -> never rebuild atlas; error on miss
# - I/O perf:
#   * pillow-simd friendly decode path; optional RGB npy cache to skip JPEG
#   * weak caches use .npz with bitpack/float16 (or raw npy when --fast-io)
# - Train perf:
#   * channels-last, cudnn.benchmark, non_blocking H2D, configurable workers/prefetch
#   * AMP via torch.amp.*, optional torch.compile on Ampere+/Ada (L4 OK)
# - Eval:
#   * EMA, optional SWA with cosine; TTA off during train, optional final TTA
# - CLI examples (single L4):
#   cpu-prep (fast): python ddp.py --stage cpu-prep --do-weak-cache --do-rgb-cache --fast-io --radar-dirs /path/njk,/path/nkm --cache-root /big/cache
#   train (strict):  python ddp.py --stage gpu-train --strict-cache-only --strict-atlas-only --radar-dirs /path/njk,/path/nkm --cache-root /big/cache

from __future__ import annotations
import os, re, glob, math, time, random, gc, contextlib, bisect, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile  # pillow-simd will override if installed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm.auto import tqdm

# ---------------- Global torch & PIL perf switches ----------------
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate slightly damaged JPEGs

# ===========================
# USER CONFIG (DEFAULTS)
# ===========================
RADAR_DIRS = [
    "/teamspace/studios/this_studio/15mins/njk",  # Radar A
    "/teamspace/studios/this_studio/15mins/nkm",  # Radar B
]
TIMESTAMP_REGEX = r"(\d{12})(?=\D*$)"   # matches YYYYMMDDHHMM in filenames

# Where to save results & caches
WORK_DIR = "/teamspace/studios/this_studio/work"
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/teamspace/studios/this_studio/tmp")
CACHE_DIR = os.path.join(CACHE_ROOT, "cache")             # weak-label/atlas caches
RGB_CACHE_DIR = os.path.join(CACHE_ROOT, "img_npy_cache") # RGB caches
for _d in [WORK_DIR, CACHE_DIR, RGB_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)

# RGB caching: "off" | "npy" | "npz"
RGB_CACHE_MODE = "off"   # will be set to "npy" during cpu-prep when --fast-io/--do-rgb-cache

# Weak label cache settings (tuned by --fast-io)
WEAK_CACHE_COMPRESS = True
WEAK_INTEN_DTYPE = np.float16
WEAK_PACK_MASK_BITS = True

# Reuse vs recompute toggles
STRICT_CACHE_ONLY = False   # forbid weak-label recompute in gpu-train
STRICT_ATLAS_ONLY = False   # forbid atlas rebuild if cache missing

# Weak-label device (CPU default; toggled by CLI)
WEAK_LABEL_DEVICE = "cpu"

# ---- Weak-cache key/lookup robustness ----
WEAK_KEY_IGNORE_MTIME = True      # ignore file mtime in cache key (prevents drift)
WEAK_CACHE_FUZZY_READ = True      # if exact key not found, reuse any *.weak.w.npz for same basename


# Weak labels (HSV)
WEAK_LABEL_MODE = "hsv_multi"   # ["palette_bar","hsv_multi","hsv_simple"]
HSV_S_MIN = 0.25
HSV_V_MIN = 0.25
# hue ranges: red wrap-around + orange + yellow + green
HSV_MULTI_RANGES = [(0.97,1.00),(0.00,0.03),(0.03,0.10),(0.10,0.18),(0.18,0.38)]
LEFT_MASK_PX = 80
HSV_H_LO = 0.03
HSV_H_HI = 0.38
PALETTE_LEFT_W = max(LEFT_MASK_PX, 90)
PALETTE_SAMPLES = 96
PALETTE_DIST_THR = 0.10

# Eval / regularization
LABEL_SMOOTH_EPS = 0.01
DROPOUT_P = 0.10
EVAL_TTA_TRAIN = False
EVAL_TTA_FINAL = True

# Cadence & neighbors
NEIGHBOR_MINUTES = 15
TOL_FRAC = 0.6
MIN_TOL_MINUTES = 5

# Atlas
ATLAS_SAMPLES = 200
ATLAS_STD_THRESH = 2.0

# Training base
EPOCHS = 20
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
BASE_CHANNELS = 32
VAL_SPLIT = 0.10
CROP = 256
MIXED_PRECISION = True
WEIGHT_DECAY = 1e-4
LAMBDA_DBZ = 1.0
DICE_WEIGHT = 0.5
LOG_EVERY = 100
CKPT_PATH = os.path.join(WORK_DIR, "radar_cleaner_best.pt")

# LR & sched
SCHEDULER = "onecycle"              # ["onecycle","cosine","plateau"]
USE_WARMUP = True
WARMUP_STEPS = 500
USE_LR_FINDER = True
LR_FINDER_STEPS = 300
LR_MIN = 1e-5
LR_MAX = 1e-1
LR_FALLBACK = 1e-3

# Early stopping
EARLY_STOP = True
EARLY_STOP_PATIENCE = 5

# EMA / SWA
USE_EMA = True
EMA_DECAY = 0.999
USE_SWA = True                      # effective with cosine
SWA_START_FRAC = 0.75
SWA_LR = 5e-5

# Grad clip
GRAD_CLIP_NORM = 1.0

# Positive-aware cropping
POS_CROP_PROB = 0.7
POS_CROP_THR = 0.005
POS_CROP_TRIES = 6

# DataLoader (tunable via CLI)
NUM_WORKERS = max(2, (os.cpu_count() or 2) - 1)
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

LR_FINDER_MAX_STEPS = 64
LR_FINDER_TIME_LIMIT_SEC = 120
LR_FINDER_SUBSET_SAMPLES = 1024
LR_FINDER_NUM_WORKERS = max(1, min(2, NUM_WORKERS))

# Random seeds
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ===========================
# DDP helpers
# ===========================
def ddp_env():
    return (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )

def ddp_is_dist(): return int(os.environ.get("WORLD_SIZE","1")) > 1

def ddp_setup():
    if not ddp_is_dist():
        return (1, 0, 0)
    world_size, rank, local_rank = ddp_env()
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(hours=2))
    return world_size, rank, local_rank

def ddp_cleanup():
    if ddp_is_dist() and dist.is_initialized():
        try: dist.barrier()
        except Exception: pass
        dist.destroy_process_group()

def is_main(rank: int): return (rank == 0)

def weighted_reduce_sum_count(loss_sum: float, count_sum: float, device):
    t = torch.tensor([loss_sum, count_sum], device=device)
    if ddp_is_dist():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t[0].item(), t[1].item()

# ===========================
# Utilities & I/O
# ===========================
def list_images(folder: str) -> List[str]:
    exts = ("*.jpg","*.jpeg","*.JPG","*.JPEG","*.png","*.bmp")
    out = []
    for e in exts: out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)

def parse_timestamp(fname: str) -> Optional[int]:
    m = re.search(TIMESTAMP_REGEX, os.path.basename(fname))
    return int(m.group(1)) if m else None

def ts_to_dt(ts: int) -> datetime: return datetime.strptime(str(ts), "%Y%m%d%H%M")
def dt_to_ts(dt: datetime) -> int: return int(dt.strftime("%Y%m%d%H%M"))
def add_minutes(ts: int, minutes: int) -> int: return dt_to_ts(ts_to_dt(ts) + timedelta(minutes=minutes))

def ensure_dir(d: str):
    if d: os.makedirs(d, exist_ok=True)


def _image_timestamp_token(path: str) -> str:
    """Derive a stable token from the filename timestamp (fallback to basename)."""
    ts = parse_timestamp(path)
    if ts is not None:
        return str(ts)
    return os.path.splitext(os.path.basename(path))[0]


def _tokenize_value(value) -> str:
    if isinstance(value, float):
        token = format(value, ".6g")
        if token.endswith(".0"):
            token = token[:-2]
    else:
        token = str(value)
    token = token.replace(".", "p").replace("-", "m")
    token = re.sub(r"[^A-Za-z0-9]+", "-", token).strip("-")
    return token or "0"


def _rgb_cache_path(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    token = _image_timestamp_token(path)
    stem = base if token in base else f"{base}.{token}"
    ext = ".npy" if RGB_CACHE_MODE == "npy" else ".npz"
    return os.path.join(RGB_CACHE_DIR, f"{stem}{ext}")

def _atomic_write(target_path: str, writer_fn):
    tmp_dir = os.path.dirname(target_path) or "."
    import tempfile as _tf
    with _tf.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=".tmp") as tf:
        tmp_path = tf.name
    try:
        writer_fn(tmp_path)
        os.replace(tmp_path, target_path)
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

def imread_rgb(path: str) -> np.ndarray:
    """Fast decoding via Pillow-SIMD; optional compact RGB cache."""
    if RGB_CACHE_MODE != "off":
        cpath = _rgb_cache_path(path)
        if os.path.isfile(cpath):
            try:
                if RGB_CACHE_MODE == "npy":
                    arr = np.load(cpath, mmap_mode="r")
                    return arr.copy()
                else:
                    with np.load(cpath) as z:
                        arr = z["arr"]
                    return arr.copy()
            except Exception:
                pass
    with Image.open(path) as img:
        try: img.draft("RGB", img.size)
        except Exception: pass
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    if RGB_CACHE_MODE != "off":
        def _writer_np(p): np.save(p, arr)
        def _writer_npz(p): np.savez_compressed(p, arr=arr)
        cpath = _rgb_cache_path(path)
        try:
            _atomic_write(cpath, _writer_np if RGB_CACHE_MODE=="npy" else _writer_npz)
        except Exception:
            pass
    return arr

# --------- Weak-label compressed cache (.npz with bit-packed mask + float16 inten) ----------
def _weak_cache_key_base(path: str, hsvp, left_mask_px: int) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    token = _image_timestamp_token(path)
    stem = base if token in base else f"{base}.{token}"
    params = [
        f"mode-{_tokenize_value(WEAK_LABEL_MODE)}",
        f"h-{_tokenize_value(hsvp.hue_lo)}-{_tokenize_value(hsvp.hue_hi)}",
        f"s-{_tokenize_value(hsvp.sat_min)}",
        f"v-{_tokenize_value(hsvp.val_min)}",
        f"left-{_tokenize_value(left_mask_px)}",
    ]
    param_str = ".".join(params)
    return os.path.join(CACHE_DIR, f"{stem}.{param_str}.weak")


def _weak_paths(base: str):
    w_npz = base + ".w.npz"
    m_npy = base + ".mask.npy"
    i_npy = base + ".inten.npy"
    return w_npz, m_npy, i_npy

def _weak_fuzzy_candidates(path: str) -> List[str]:
    """Return any existing weak caches for this basename, regardless of hash (path-agnostic, no mdate)."""
    base = os.path.splitext(os.path.basename(path))[0]
    pat = os.path.join(CACHE_DIR, f"{base}.*.weak.w.npz")
    cands = glob.glob(pat)
    try:
        cands.sort(key=lambda p: (-os.path.getsize(p), p))
    except Exception:
        cands.sort()
    return cands


def _save_weak_npz(path: str, mask: np.ndarray, inten: np.ndarray):
    H, W = mask.shape
    # Decide intensity payload
    iq8 = 1 if WEAK_INTEN_DTYPE is np.uint8 else 0
    if iq8:
        inten_disk = np.clip(np.rint(inten * 255.0), 0, 255).astype(np.uint8)
    else:
        inten_disk = inten.astype(WEAK_INTEN_DTYPE if WEAK_CACHE_COMPRESS else np.float32)

    # Decide mask payload
    if WEAK_PACK_MASK_BITS:
        packed = np.packbits(mask.astype(np.uint8), axis=1)
        def _writer(p):
            np.savez_compressed(
                p,
                shape=np.array([H, W], np.int32),
                mask=packed,
                inten=inten_disk,
                iq8=np.array([iq8], np.uint8),
            )
    else:
        def _writer(p):
            np.savez_compressed(
                p,
                shape=np.array([H, W], np.int32),
                mask=mask.astype(np.uint8),
                inten=inten_disk,
                iq8=np.array([iq8], np.uint8),
            )
    _atomic_write(path, _writer)

def _load_weak_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        H, W = map(int, z["shape"])
        m = z["mask"]
        iq8 = int(z["iq8"][0]) if "iq8" in z.files else 0

        if WEAK_PACK_MASK_BITS and m.ndim == 2:
            mask = np.unpackbits(m, axis=1)[:, :W].astype(bool)
        else:
            mask = (m.astype(np.uint8) > 0)

        inten_raw = z["inten"]
        inten = (inten_raw.astype(np.float32) / 255.0) if iq8 else inten_raw.astype(np.float32)
    return mask, inten


# ===========================
# HSV + weak labels (cacheable)
# ===========================
@dataclass
class HSVParams:
    hue_lo: float = 0.03
    hue_hi: float = 0.38
    sat_min: float = HSV_S_MIN
    val_min: float = HSV_V_MIN

def rgb_to_hsv_np(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32) / 255.0
    r, g, b = arr[...,0], arr[...,1], arr[...,2]
    mx = np.max(arr, axis=-1); mn = np.min(arr, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx); mask = diff != 0
    r_eq = (mx == r) & mask; g_eq = (mx == g) & mask; b_eq = (mx == b) & mask
    h[r_eq] = (60 * ((g[r_eq] - b[r_eq]) / diff[r_eq]) + 0) % 360
    h[g_eq] = (60 * ((b[g_eq] - r[g_eq]) / diff[g_eq]) + 120) % 360
    h[b_eq] = (60 * ((r[b_eq] - g[b_eq]) / diff[b_eq]) + 240) % 360
    s = np.zeros_like(mx); nz = mx != 0
    s[nz] = diff[nz] / mx[nz]; v = mx
    return np.stack([h/360.0, s, v], axis=-1)

def _union_hue_ranges(H, ranges):
    segs = []; total_width = 0.0
    for lo, hi in ranges:
        total_width += ((hi - lo) if hi >= lo else (1.0 - lo + hi))
    cum = 0.0
    for lo, hi in ranges:
        width = (hi - lo) if hi >= lo else (1.0 - lo + hi)
        a, b = cum, cum + (width / max(1e-6, total_width))
        segs.append((lo, hi, a, b)); cum = b
    return segs

def _in_hue_range(H, lo, hi):
    return (H >= lo) & (H <= hi) if hi >= lo else ((H >= lo) | (H <= hi))

def _morph_open_close_torch(mask_in, k: int = 3, device: str = "cpu") -> torch.Tensor:
    use_cuda = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device) if use_cuda else torch.device("cpu")

    if isinstance(mask_in, torch.Tensor):
        x = mask_in.to(dev)
    else:
        x = torch.as_tensor(mask_in, dtype=torch.float32, device=dev)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)

    x = x.float()
    kernel = torch.ones((1, 1, k, k), device=dev, dtype=x.dtype)
    pad = k // 2

    hit = F.conv2d(x, kernel, padding=pad)
    eroded = (hit >= (k * k) - 1e-6).float()
    hit2 = F.conv2d(eroded, kernel, padding=pad)
    opened = (hit2 > 1e-6).float()
    hit3 = F.conv2d(opened, kernel, padding=pad)
    dilated = (hit3 > 1e-6).float()
    hit4 = F.conv2d(dilated, kernel, padding=pad)
    closed = (hit4 >= (k * k) - 1e-6)
    return closed.squeeze(0).squeeze(0).to(dtype=torch.bool)

def _weak_label_core(img_rgb: np.ndarray, hsvp: HSVParams, left_mask_px: int, device: str = "cpu"):
    use_cuda = device is not None and str(device).startswith("cuda") and torch.cuda.is_available()
    dev = torch.device(device) if use_cuda else torch.device("cpu")

    arr = torch.as_tensor(img_rgb, dtype=torch.float32, device=dev) / 255.0
    eps = 1e-6

    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if dev.type == "cuda" else contextlib.nullcontext()
    with amp_ctx:
        r, g, b = arr.unbind(dim=-1)
        mx = arr.max(dim=-1).values
        mn = arr.min(dim=-1).values
        diff = mx - mn
        mask_nz = diff > eps

        h = torch.zeros_like(mx)
        denom = diff + eps
        h = torch.where(mask_nz & (mx == r), torch.remainder((g - b) / denom, 6.0), h)
        h = torch.where(mask_nz & (mx == g), ((b - r) / denom) + 2.0, h)
        h = torch.where(mask_nz & (mx == b), ((r - g) / denom) + 4.0, h)
        h = torch.remainder(h / 6.0, 1.0)

        s = torch.where(mx > eps, diff / (mx + eps), torch.zeros_like(mx))
        v = mx

        sat_min = float(hsvp.sat_min)
        val_min = float(hsvp.val_min)

        if WEAK_LABEL_MODE == "hsv_multi":
            ranges = _union_hue_ranges(None, HSV_MULTI_RANGES)
            mask = torch.zeros_like(mx, dtype=torch.bool, device=dev)
            inten = torch.zeros_like(mx, dtype=torch.float32, device=dev)
            for lo, hi, a, b in ranges:
                rng = _in_hue_range(h, lo, hi) & (s >= sat_min) & (v >= val_min)
                mask = mask | rng
                width = (hi - lo) if hi >= lo else (1.0 - lo + hi)
                width = width + 1e-6
                if hi >= lo:
                    pos = (h - lo) / width
                else:
                    pos = torch.where(h >= lo, (h - lo) / width, (h + 1.0 - lo) / width)
                inten = torch.where(rng, a + pos * (b - a), inten)
        else:
            hue_lo = float(hsvp.hue_lo)
            hue_hi = float(hsvp.hue_hi)
            mask = (h >= hue_lo) & (h <= hue_hi) & (s >= sat_min) & (v >= val_min)
            inten = torch.clamp((h - hue_lo) / max(1e-6, (hue_hi - hue_lo)), 0.0, 1.0).to(dtype=torch.float32)
            inten = torch.where(mask, inten, torch.zeros_like(inten))

    inten = inten.float()
    mask = mask.to(dtype=torch.bool)

    if left_mask_px > 0:
        mask[:, :left_mask_px] = False
        inten[:, :left_mask_px] = 0.0

    mask = _morph_open_close_torch(mask.float(), k=3, device=str(dev))
    area_k = 5
    m = mask.float().unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, area_k, area_k), device=m.device)
    s_area = F.conv2d(m, kernel, padding=area_k // 2)
    mask = (s_area.squeeze(0).squeeze(0) >= 3.0)
    inten = torch.where(mask, inten, torch.zeros_like(inten))

    mask_np = mask.detach().to("cpu").numpy().astype(bool)
    inten_np = inten.detach().to("cpu").numpy().astype(np.float32)
    return mask_np, inten_np

def _weak_cache_exists_only(path: str, hsvp: HSVParams, left_mask_px: int) -> bool:
    base = _weak_cache_key_base(path, hsvp, left_mask_px)
    w_npz, m_npy, i_npy = _weak_paths(base)
    if os.path.isfile(w_npz) or (os.path.isfile(m_npy) and os.path.isfile(i_npy)):
        return True
    if WEAK_CACHE_FUZZY_READ:
        # any older cache for same basename counts as “present” for strict reuse
        return len(_weak_fuzzy_candidates(path)) > 0
    return False

def get_weak_label_cached(path: str, hsvp: HSVParams, left_mask_px: int = LEFT_MASK_PX, device: Optional[str] = None):
    base = _weak_cache_key_base(path, hsvp, left_mask_px)
    w_npz, m_npy, i_npy = _weak_paths(base)

    # Reuse if present
    if os.path.isfile(w_npz):
        try:
            return _load_weak_npz(w_npz)
        except Exception:
            pass

    if os.path.isfile(m_npy) and os.path.isfile(i_npy):
        try:
            m = np.load(m_npy, mmap_mode="r").astype(bool)
            inten = np.load(i_npy, mmap_mode="r").astype(np.float32)
            if WEAK_CACHE_COMPRESS:
                try:
                    _save_weak_npz(w_npz, m, inten)
                except Exception:
                    pass
            return m, inten
        except Exception:
            pass

    # Fuzzy reuse: accept any *.weak.w.npz for same basename (mtime/param drift)
    if WEAK_CACHE_FUZZY_READ:
        cands = _weak_fuzzy_candidates(path)
        for cp in cands:
            try:
                return _load_weak_npz(cp)
            except Exception:
                continue

    # Strict mode: never recompute
    if STRICT_CACHE_ONLY:
        raise FileNotFoundError(
            f"Weak-label cache missing for {path} (strict mode). Run cpu-prep to precompute."
        )


    # Recompute (allowed only when strict off)
    img = imread_rgb(path)
    device_str = device if device is not None else WEAK_LABEL_DEVICE
    if device_str != "cpu" and not torch.cuda.is_available():
        device_str = "cpu"
    m, inten = _weak_label_core(img, hsvp, left_mask_px, device=device_str)
    if WEAK_CACHE_COMPRESS:
        try: _save_weak_npz(w_npz, m, inten)
        except Exception: pass
    return m, inten

# ===========================
# Atlas (cached, compact)
# ===========================
def build_atlas_mask(image_paths: List[str], left_mask_px: int = LEFT_MASK_PX,
                     max_samples: int = ATLAS_SAMPLES, std_thresh: float = ATLAS_STD_THRESH) -> np.ndarray:
    if len(image_paths) == 0: raise ValueError("No images for atlas.")
    sample_paths = image_paths if len(image_paths) <= max_samples else random.sample(image_paths, max_samples)
    arrs = [imread_rgb(p).astype(np.float32) for p in sample_paths]
    Hmax = max(a.shape[0] for a in arrs); Wmax = max(a.shape[1] for a in arrs)
    arrs = [np.pad(a, ((0, Hmax - a.shape[0]), (0, Wmax - a.shape[1]), (0, 0)), mode="reflect") for a in arrs]
    stack = np.stack(arrs, axis=0)
    std = stack.std(axis=0).mean(axis=-1)
    static = std < std_thresh
    static[:, :max(0, left_mask_px)] = True
    return static.astype(bool)

def _fit_mask_to(H, W, mask):
    if mask is None: return np.zeros((H, W), dtype=bool)
    h, w = mask.shape[:2]
    if h < H or w < W: mask = np.pad(mask, ((0, H-h), (0, W-w)), mode="edge")
    if mask.shape[0] > H or mask.shape[1] > W: mask = mask[:H, :W]
    return mask.astype(bool)

def _atlas_cache_paths(root: str):
    base = os.path.join(CACHE_DIR, f"{os.path.basename(root)}_atlas")
    return base + ".npz", base + ".npy"

def _save_bool_mask_npz(path: str, mask: np.ndarray):
    """Robust, atomic save for atlas: write -> flush+fsync -> verify -> rename."""
    ensure_dir(os.path.dirname(path) or ".")
    H, W = mask.shape
    packed = np.packbits(mask.astype(np.uint8), axis=1)

    import tempfile  # don't shadow global `os`

    # create a *closed* temp file path
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)

    try:
        # write compressed npz to tmp
        with open(tmp_path, "wb") as f:
            np.savez_compressed(f, shape=np.array([H, W], np.int32), mask=packed)
            f.flush()
            os.fsync(f.fileno())

        # light verification read
        with np.load(tmp_path) as z:
            H2, W2 = map(int, z["shape"])
            _ = z["mask"]
            if H2 != H or W2 != W:
                raise RuntimeError(f"atlas shape mismatch: ({H2},{W2}) vs ({H},{W})")

        # atomic move
        os.replace(tmp_path, path)

    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise

def _load_bool_mask_npz(path: str) -> np.ndarray:
    with np.load(path) as z:
        H, W = map(int, z["shape"])
        packed = z["mask"]
        return np.unpackbits(packed, axis=1)[:, :W].astype(bool)

# ===========================
# Helper channels
# ===========================
def helper_channels(h: int, w: int, center: Optional[Tuple[int,int]] = None):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w//2, h//2) if center is None else center
    dx = (xx - cx).astype(np.float32); dy = (yy - cy).astype(np.float32)
    r = np.sqrt(dx*dx + dy*dy); r_norm = r / (r.max() + 1e-6)
    ang = np.arctan2(dy, dx)
    return r_norm.astype(np.float32), np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

# ===========================
# Source indexing (atlas reuse strict)
# ===========================
@dataclass
class SourceData:
    root: str
    items: List[Dict]            # [{ts:int, path:str}]
    ts_to_path: Dict[int,str]
    atlas: Optional[np.ndarray]
    ts_sorted: List[int]

def index_source(root: str, rank: int = 0) -> SourceData:
    paths = list_images(root)
    items = []
    for p in paths:
        ts = parse_timestamp(p)
        if ts is not None: items.append({"ts": ts, "path": p})
    items = sorted(items, key=lambda x: x["ts"])
    if len(items) == 0:
        raise RuntimeError(f"No timestamped images in {root}. Expect YYYYMMDDHHMM in filename.")

    atlas_npz, atlas_npy = _atlas_cache_paths(root)
    atlas = None
    if os.path.isfile(atlas_npz):
        try:
            atlas = _load_bool_mask_npz(atlas_npz)
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Atlas load failed for {root}: {e}", flush=True)
            atlas = None
    elif os.path.isfile(atlas_npy):
        try:
            atlas = np.load(atlas_npy).astype(bool)
            try: _save_bool_mask_npz(atlas_npz, atlas)
            except Exception as e:
                if rank == 0: print(f"⚠️  Atlas re-save (.npz) failed: {e}", flush=True)
        except Exception:
            atlas = None

    if atlas is None:
        if STRICT_ATLAS_ONLY:
            raise FileNotFoundError(
                f"Atlas cache missing or corrupted for {root} (strict atlas mode). Run cpu-prep to rebuild."
            )
        atlas = build_atlas_mask([it["path"] for it in items])
        try: _save_bool_mask_npz(atlas_npz, atlas)
        except Exception as e:
            if rank == 0: print(f"⚠️  Failed to save atlas npz: {e}", flush=True)

    ts_to_path = {it["ts"]: it["path"] for it in items}
    ts_sorted = [it["ts"] for it in items]
    if rank == 0:
        h, w = imread_rgb(items[0]["path"]).shape[:2]
        print(f"Indexing: {root}\n  frames={len(items)} size={w}x{h} static={atlas.mean()*100:.1f}%", flush=True)
    return SourceData(root=root, items=items, ts_to_path=ts_to_path, atlas=atlas, ts_sorted=ts_sorted)

# ===========================
# Dataset (two sources, prev/t/next fusion) — uses cached weak labels
# ===========================
@dataclass
class DataConfig:
    left_mask_px: int = LEFT_MASK_PX
    hsv: HSVParams = field(default_factory=HSVParams)
    neighbor_minutes: int = NEIGHBOR_MINUTES
    crop: int = CROP
    pos_crop_prob: float = POS_CROP_PROB
    pos_crop_thr: float = POS_CROP_THR
    pos_crop_tries: int = POS_CROP_TRIES
    fuse_mode: str = "max"
    weak_label_device: str = "cpu"

class TwoRadarFusionDataset(Dataset):
    def __init__(self, sources: List[SourceData], split: str = "train", val_split: float = VAL_SPLIT, dcfg: Optional[DataConfig] = None):
        assert len(sources) == 2, "Provide exactly two radar sources."
        self.A, self.B = sources
        self.dcfg = dcfg or DataConfig()
        if self.dcfg.weak_label_device != "cpu" and not torch.cuda.is_available():
            self.dcfg.weak_label_device = "cpu"
        ts_all = sorted(set([it["ts"] for it in self.A.items] + [it["ts"] for it in self.B.items]))
        n = len(ts_all); n_val = max(1, int(n * val_split))
        self.ts_list = ts_all[:n - n_val] if split == "train" else ts_all[n - n_val:]
        self._hc_cache: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self): return len(self.ts_list)

    def _tol_minutes(self, offset_min: int) -> int:
        return max(MIN_TOL_MINUTES, int(abs(offset_min) * TOL_FRAC))

    def _nearest_within(self, s: SourceData, target_ts: int, tol_min: int) -> Optional[int]:
        if target_ts in s.ts_to_path: return target_ts
        ts_list = s.ts_sorted
        i = bisect.bisect_left(ts_list, target_ts)
        best_t, best_dt = None, float("inf")
        n = len(ts_list)
        for off in range(0, 8):
            any_within = False
            idxs = (i - off, i + off) if off > 0 else (i,)
            for j in idxs:
                if 0 <= j < n:
                    t = ts_list[j]
                    dtmin = abs((ts_to_dt(t) - ts_to_dt(target_ts)).total_seconds() / 60.0)
                    if dtmin <= tol_min:
                        any_within = True
                        if dtmin < best_dt:
                            best_dt, best_t = dtmin, t
            if off > 0 and not any_within:
                break
        return best_t

    def _neighbor(self, s: SourceData, center_ts: int, offset_min: int) -> Optional[int]:
        targ = add_minutes(center_ts, offset_min)
        tol = self._tol_minutes(offset_min if offset_min != 0 else self.dcfg.neighbor_minutes)
        return self._nearest_within(s, targ, tol)

    def _rand_view(self):
        return (random.random() < 0.5, random.random() < 0.5, random.randint(0,3))

    def _apply_view(self, arr, view):
        flip_h, flip_v, k = view
        if flip_h: arr = arr[:, ::-1]
        if flip_v: arr = arr[::-1, :]
        if k: arr = np.rot90(arr, k)
        return arr

    def _pad_to(self, a: np.ndarray, th: int, tw: int):
        ph = max(0, th - a.shape[0]); pw = max(0, tw - a.shape[1])
        if ph or pw: a = np.pad(a, ((0,ph),(0,pw)), mode="reflect")
        return a

    def _choose_crop(self, chs: List[np.ndarray], y_mask: np.ndarray, crop: int):
        H, W = chs[0].shape
        if random.random() < self.dcfg.pos_crop_prob:
            for _ in range(self.dcfg.pos_crop_tries):
                y0 = 0 if H <= crop else random.randint(0, H - crop)
                x0 = 0 if W <= crop else random.randint(0, W - crop)
                if y_mask[y0:y0+crop, x0:x0+crop].mean() > self.dcfg.pos_crop_thr:
                    return y0, x0
        y0 = 0 if H <= crop else random.randint(0, H - crop)
        x0 = 0 if W <= crop else random.randint(0, W - crop)
        return y0, x0

    def __getitem__(self, idx: int):
        ts = self.ts_list[idx]
        offs = [-self.dcfg.neighbor_minutes, 0, +self.dcfg.neighbor_minutes]
        a_ts = [self._neighbor(self.A, ts, o) for o in offs]
        b_ts = [self._neighbor(self.B, ts, o) for o in offs]
        if a_ts[1] is None and b_ts[1] is None:
            step = self.dcfg.neighbor_minutes
            found = False
            for delta in range(step, step*5, step):
                for cand in [add_minutes(ts, delta), add_minutes(ts, -delta)]:
                    if self._nearest_within(self.A, cand, self._tol_minutes(0)) or self._nearest_within(self.B, cand, self._tol_minutes(0)):
                        ts = cand
                        a_ts = [self._neighbor(self.A, ts, o) for o in offs]
                        b_ts = [self._neighbor(self.B, ts, o) for o in offs]
                        found = True; break
                if found: break

        pairs = []; present_flags = []
        for tlist, S in [(a_ts, self.A), (b_ts, self.B)]:
            for t in tlist:
                if t is None:
                    pairs.append(None); present_flags.append(0.0)
                else:
                    path = S.ts_to_path[t]
                    m, inten = get_weak_label_cached(
                        path,
                        self.dcfg.hsv,
                        self.dcfg.left_mask_px,
                        device=self.dcfg.weak_label_device,
                    )
                    pairs.append((m, inten)); present_flags.append(1.0)

        crop_sz = max(1, int(self.dcfg.crop))
        shapes = [p[0].shape for p in pairs if p is not None]
        H, W = (crop_sz, crop_sz) if len(shapes) == 0 else (max(s[0] for s in shapes), max(s[1] for s in shapes))

        def to_hw(pair):
            if pair is None:
                return (np.zeros((H,W), bool), np.zeros((H,W), np.float32))
            m, inten = pair
            if m.shape == (H,W):
                return (m, inten)
            ph = max(0, H - m.shape[0]); pw = max(0, W - m.shape[1])
            if ph or pw:
                m = np.pad(m, ((0,ph),(0,pw)), mode="edge")
                inten = np.pad(inten, ((0,ph),(0,pw)), mode="edge")
            return (m[:H,:W], inten[:H,:W])

        frames_hw = []; avs_hw = []
        for pair, pflag in zip(pairs, present_flags):
            m, inten = to_hw(pair)
            frames_hw.append((m, inten))
            avs_hw.append(np.full((H, W), float(pflag), dtype=np.float32))

        atlas_A = _fit_mask_to(H, W, self.A.atlas); atlas_B = _fit_mask_to(H, W, self.B.atlas)

        def proc_triplet(triplet, av_triplet, atlas):
            out_inten = []; out_av = []; mask_t = None; inten_t = None
            for i, ((m, inten), av) in enumerate(zip(triplet, av_triplet)):
                inten = np.where(atlas, 0.0, inten).astype(np.float32)
                m = np.where(atlas, 0, m).astype(bool)
                out_inten.append(inten)
                out_av.append(av.astype(np.float32))
                if i == 1:
                    mask_t, inten_t = m, inten
            return out_inten, out_av, mask_t, inten_t

        A_prev, A_t, A_next = frames_hw[0:3]; B_prev, B_t, B_next = frames_hw[3:6]
        A_prev_av, A_t_av, A_next_av = avs_hw[0:3]; B_prev_av, B_t_av, B_next_av = avs_hw[3:6]

        intenA, avA, maskA_t, intenA_t = proc_triplet([A_prev, A_t, A_next], [A_prev_av, A_t_av, A_next_av], atlas_A)
        intenB, avB, maskB_t, intenB_t = proc_triplet([B_prev, B_t, B_next], [B_prev_av, B_t_av, B_next_av], atlas_B)
        y_dbz = (intenA_t + intenB_t)/2.0 if self.dcfg.fuse_mode == "mean" else np.maximum(intenA_t, intenB_t)
        y_mask = (maskA_t | maskB_t)

        key = (H,W)
        if key not in self._hc_cache: self._hc_cache[key] = helper_channels(H, W, center=None)
        r_norm, cos_t, sin_t = self._hc_cache[key]

        chs = [intenA[0], intenA[1], intenA[2], intenB[0], intenB[1], intenB[2],
               avA[0],    avA[1],    avA[2],    avB[0],    avB[1],    avB[2],    r_norm, cos_t, sin_t]

        view = self._rand_view()
        chs = [self._apply_view(a, view) for a in chs]
        y_mask = self._apply_view(y_mask.astype(np.float32), view) > 0.5
        y_dbz  = self._apply_view(y_dbz, view)

        def pad_min(a): return self._pad_to(a, max(crop_sz, a.shape[0]), max(crop_sz, a.shape[1]))
        chs = [pad_min(a) for a in chs]
        y_mask = pad_min(y_mask.astype(np.float32)) > 0.5
        y_dbz  = pad_min(y_dbz)

        crop = crop_sz
        y0, x0 = self._choose_crop(chs, y_mask.astype(np.float32), crop)
        chs = [a[y0:y0+crop, x0:x0+crop] for a in chs]
        y_mask = y_mask[y0:y0+crop, x0:x0+crop]
        y_dbz  = y_dbz[y0:y0+crop, x0:x0+crop]

        x = np.stack(chs, axis=0).astype(np.float32)
        y_mask = y_mask.astype(np.float32)[None, ...]
        y_dbz  = y_dbz.astype(np.float32)[None, ...]
        return {"x": torch.from_numpy(x), "y_mask": torch.from_numpy(y_mask), "y_dbz": torch.from_numpy(y_dbz)}

# ===========================
# Model: Tiny U-Net with GroupNorm
# ===========================
class ConvGNReLU(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups, cout), num_channels=cout)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.gn(self.conv(x)))

class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(ConvGNReLU(cin, cout), ConvGNReLU(cout, cout))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        y = self.block(x)
        return self.pool(y), y

class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin//2, 2, stride=2)
        self.block = nn.Sequential(ConvGNReLU(cin, cout), ConvGNReLU(cout, cout))
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2); dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy - dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class TinyUNet(nn.Module):
    def __init__(self, in_ch=15, base=32):
        super().__init__()
        self.stem = nn.Sequential(ConvGNReLU(in_ch, base), ConvGNReLU(base, base))
        self.d1 = Down(base, base*2)
        self.d2 = Down(base*2, base*4)
        self.d3 = Down(base*4, base*8)
        self.bot = nn.Sequential(ConvGNReLU(base*8, base*16), nn.Dropout2d(DROPOUT_P), ConvGNReLU(base*16, base*16))
        self.u3 = Up(base*16, base*8)
        self.u2 = Up(base*8, base*4)
        self.u1 = Up(base*4, base*2)
        self.head_mask = nn.Conv2d(base*2, 1, 1)
        self.head_dbz  = nn.Conv2d(base*2, 1, 1)
    def forward(self, x):
        x0 = self.stem(x)
        x1, s1 = self.d1(x0)
        x2, s2 = self.d2(x1)
        x3, s3 = self.d3(x2)
        xb = self.bot(x3)
        x = self.u3(xb, s3); x = self.u2(x, s2); x = self.u1(x, s1)
        return self.head_mask(x), self.head_dbz(x)

# ===========================
# Losses
# ===========================
class FocalBCE(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = p*targets + (1-p)*(1-targets)
        w  = self.alpha*targets + (1-self.alpha)*(1-targets)
        loss = -w * (1-pt).pow(self.gamma) * torch.log(pt.clamp(1e-6, 1.0))
        return loss.mean() if self.reduction=="mean" else (loss.sum() if self.reduction=="sum" else loss)

def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    num = 2*(p*targets).sum(dim=(1,2,3))
    den = (p+targets).sum(dim=(1,2,3)).clamp_min(eps)
    return (1 - (num + eps) / (den + eps)).mean()

# ===========================
# EMA helper
# ===========================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone()
    def update(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[name] = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[name] = p.data.detach().clone()
            p.data = self.shadow[name].detach().clone()
    def restore(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data = self.backup[name]
        self.backup = {}

# ===========================
# AMP & compile gate
# ===========================
def allow_compile():
    if not torch.cuda.is_available(): return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 8  # Ampere+/Ada (L4 OK)

# ===========================
# TQDM helpers
# ===========================
def _gb(bytes_): 
    return bytes_ / (1024**3)

def _cuda_mem():
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    peak  = torch.cuda.max_memory_allocated()
    return _gb(alloc), _gb(reserv), _gb(peak)

def _make_bar(loader, desc, rank):
    return tqdm(
        loader,
        total=len(loader),
        disable=(not is_main(rank)),
        dynamic_ncols=True,
        leave=True,
        desc=desc,
        mininterval=0.1,
        miniters=1,
    )

# ===========================
# LR Finder (rank-0 only)
# ===========================
def lr_range_test(model, train_loader, device, loss_focal, steps=LR_FINDER_STEPS, lr_min=LR_MIN, lr_max=LR_MAX,
                  AMP_ENABLED=False, amp_dtype=None, amp_ctx=contextlib.nullcontext, scaler=None, pin=False,
                  time_limit=LR_FINDER_TIME_LIMIT_SEC, dice_weight=DICE_WEIGHT, lambda_dbz=LAMBDA_DBZ):
    model.train()
    tmp_opt = torch.optim.AdamW(model.parameters(), lr=lr_min, weight_decay=WEIGHT_DECAY)
    num = min(int(steps), int(LR_FINDER_MAX_STEPS))
    gamma = (lr_max / lr_min) ** (1 / max(1, num-1))
    best_loss = float("inf"); best_lr = lr_min; loss_smooth = None; iters = 0
    t0 = time.time()
    for batch in train_loader:
        iters += 1
        if iters > num or (time.time() - t0) > float(time_limit): break
        x = batch["x"].to(device, non_blocking=pin).contiguous(memory_format=torch.channels_last)
        y_mask = batch["y_mask"].to(device, non_blocking=pin)
        y_dbz = batch["y_dbz"].to(device, non_blocking=pin)
        tmp_opt.zero_grad(set_to_none=True)
        with amp_ctx():
            logits, dbz_pred = model(x)
            loss_mask = loss_focal(logits, y_mask) + dice_weight * dice_loss(logits, y_mask)
            gate = (y_mask > 0.5).float()
            l1_all = F.smooth_l1_loss(dbz_pred, y_dbz, reduction="none")
            pos = gate.sum()
            loss_dbz = (l1_all * gate).sum() / (pos + 1e-6)
            loss = loss_mask + lambda_dbz * loss_dbz
        if AMP_ENABLED:
            scaler.scale(loss).backward(); scaler.step(tmp_opt); scaler.update()
        else:
            loss.backward(); tmp_opt.step()
        loss_val = float(loss.item())
        loss_smooth = loss_val if loss_smooth is None else 0.98*loss_smooth + 0.02*loss_val
        if loss_smooth < best_loss:
            best_loss = loss_smooth; best_lr = tmp_opt.param_groups[0]["lr"]
        for pg in tmp_opt.param_groups: pg["lr"] *= gamma
        if not math.isfinite(loss_val): break
    return max(lr_min, best_lr / 3.0)

# ===========================
# TTA helper
# ===========================
def _predict_tta(model, x, do_tta: bool):
    if not do_tta:
        return model(x)
    acc_logits = None; acc_dbz = None; n = 0
    for k in range(4):
        xx = torch.rot90(x, k, dims=(2, 3))
        l, d = model(xx)
        l = torch.rot90(l, 4 - k, dims=(2, 3)); d = torch.rot90(d, 4 - k, dims=(2, 3))
        acc_logits = l if acc_logits is None else acc_logits + l
        acc_dbz    = d if acc_dbz    is None else acc_dbz    + d
        n += 1
    return acc_logits / n, acc_dbz / n

def bcast_float_from0(val: float) -> float:
    if not ddp_is_dist(): return val
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    t = torch.tensor([val], device=device)
    dist.broadcast(t, src=0)
    return t.item()

# ===========================
# Loader builders
# ===========================
def build_sources(rank=0):
    sources = []
    for d in RADAR_DIRS:
        if os.path.isdir(d):
            src = index_source(d, rank=rank)
            sources.append(src)
        else:
            if rank == 0: print(f"WARNING: not a folder: {d}", flush=True)
    if len(sources) != 2:
        if rank == 0: print("Need exactly two valid radar sources in RADAR_DIRS.", flush=True)
        raise SystemExit(1)
    return sources

def make_loaders(sources, device, rank, world_size, num_workers, prefetch_factor):
    weak_dev = WEAK_LABEL_DEVICE
    if weak_dev != "cpu" and not torch.cuda.is_available():
        weak_dev = "cpu"
    dcfg = DataConfig(weak_label_device=weak_dev)
    train_ds = TwoRadarFusionDataset(sources, split="train", val_split=VAL_SPLIT, dcfg=dcfg)
    val_ds   = TwoRadarFusionDataset(sources, split="val",   val_split=VAL_SPLIT, dcfg=dcfg)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if ddp_is_dist() else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_is_dist() else None

    def _wif(worker_id):
        np.random.seed(SEED + worker_id + rank*1000)
        random.seed(SEED + worker_id + rank*1000)

    g = torch.Generator(); g.manual_seed(SEED + rank)
    pin = (device.type == "cuda")

    def mk(ds, bs, shuffle, sampler):
        kwargs = dict(dataset=ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin, worker_init_fn=_wif,
                      generator=g, drop_last=False)
        if num_workers > 0:
            kwargs["persistent_workers"] = PERSISTENT_WORKERS
            kwargs["prefetch_factor"] = prefetch_factor
        return DataLoader(**kwargs)

    train_loader = mk(train_ds, BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = mk(val_ds,   BATCH_SIZE, shuffle=False,                   sampler=val_sampler)

    if is_main(rank):
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)
        eff_bsz = BATCH_SIZE * (world_size if ddp_is_dist() else 1) * max(1, GRAD_ACCUM_STEPS)
        print(f"Effective batch size (per optimizer step): {eff_bsz}", flush=True)

    return dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler

# ===========================
# Training function
# ===========================
def run_training(hparams, device, rank, local_rank, world_size,
                 dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler,
                 save_ckpt_path=CKPT_PATH, verbose=True):
    base = hparams.get("base_channels", BASE_CHANNELS)
    lambda_dbz = hparams.get("lambda_dbz", LAMBDA_DBZ)
    scheduler_kind = hparams.get("scheduler", SCHEDULER)
    use_warmup = hparams.get("use_warmup", USE_WARMUP)
    warmup_steps = hparams.get("warmup_steps", WARMUP_STEPS)
    grad_clip = hparams.get("grad_clip", GRAD_CLIP_NORM)
    use_ema = hparams.get("use_ema", USE_EMA)
    epochs = hparams.get("epochs", EPOCHS)
    focal_alpha = hparams.get("focal_alpha", 0.25)
    focal_gamma = hparams.get("focal_gamma", 2.0)
    dice_w = hparams.get("dice_weight", DICE_WEIGHT)
    use_swa = hparams.get("use_swa", (USE_SWA and scheduler_kind == "cosine"))
    grad_accum = max(1, hparams.get("grad_accum_steps", GRAD_ACCUM_STEPS))

    AMP_DEVICE = "cuda" if (device.type == "cuda") else "cpu"
    AMP_ENABLED = bool(MIXED_PRECISION and (AMP_DEVICE == "cuda"))
    scaler = torch.amp.GradScaler(AMP_DEVICE, enabled=AMP_ENABLED)
    amp_dtype = torch.bfloat16 if (AMP_ENABLED and torch.cuda.is_bf16_supported()) else torch.float16
    amp_ctx = (lambda: torch.amp.autocast(AMP_DEVICE, dtype=amp_dtype, enabled=AMP_ENABLED))

    model = TinyUNet(in_ch=15, base=base).to(device).to(memory_format=torch.channels_last)
    if allow_compile():
        try: model = torch.compile(model)
        except Exception: pass
    if ddp_is_dist():
        ddp_kwargs = dict(find_unused_parameters=False)
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    loss_focal = FocalBCE(alpha=focal_alpha, gamma=focal_gamma)

    if hparams.get("use_lr_finder", USE_LR_FINDER):
        if is_main(rank):
            from torch.utils.data import Subset
            subset_idx = np.linspace(0, len(train_ds) - 1, num=min(LR_FINDER_SUBSET_SAMPLES, len(train_ds)), dtype=int).tolist()
            tmp_ds = Subset(train_ds, subset_idx)
            tmp_loader = DataLoader(tmp_ds, batch_size=min(BATCH_SIZE, 8), shuffle=True,
                                    num_workers=LR_FINDER_NUM_WORKERS, pin_memory=False, drop_last=False)
            model_copy = TinyUNet(in_ch=15, base=base).to(device).to(memory_format=torch.channels_last)
            if allow_compile():
                try: model_copy = torch.compile(model_copy)
                except Exception: pass
            best_lr = lr_range_test(model_copy, tmp_loader, device, loss_focal,
                                    steps=hparams.get("lr_finder_steps", LR_FINDER_STEPS),
                                    lr_min=hparams.get("lr_min", LR_MIN),
                                    lr_max=hparams.get("lr_max", LR_MAX),
                                    AMP_ENABLED=AMP_ENABLED, amp_ctx=amp_ctx, scaler=scaler, pin=False,
                                    time_limit=LR_FINDER_TIME_LIMIT_SEC, dice_weight=dice_w,
                                    lambda_dbz=lambda_dbz)
            del model_copy; gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
        else:
            best_lr = 0.0
        init_lr = bcast_float_from0(best_lr if is_main(rank) else 0.0)
        if init_lr <= 0: init_lr = LR_FALLBACK
        if is_main(rank) and verbose: print(f"[LR Finder] best_lr≈{init_lr:.3e}", flush=True)
    else:
        init_lr = hparams.get("lr", LR_FALLBACK)

    opt = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=hparams.get("weight_decay", WEIGHT_DECAY))
    steps_per_epoch = max(1, len(train_loader))
    opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    total_opt_steps = opt_steps_per_epoch * epochs
    warmup_steps_effective = min(warmup_steps, total_opt_steps-1) if use_warmup else 0

    sched = None
    swa_model = None
    swa_sched = None

    if scheduler_kind == "onecycle":
        max_lr = init_lr
        for pg in opt.param_groups: pg["lr"] = max_lr / 25.0
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, epochs=epochs, steps_per_epoch=opt_steps_per_epoch,
            pct_start=0.1, anneal_strategy="cos", div_factor=25.0, final_div_factor=1e4
        )
    elif scheduler_kind == "cosine":
        if use_warmup and warmup_steps_effective > 0:
            warm = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps_effective)
            cos_Tmax = max(1, total_opt_steps - warmup_steps_effective)
            cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cos_Tmax)
            sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[warmup_steps_effective])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_opt_steps))
        if use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR
            base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            swa_model = AveragedModel(base_module)
            swa_sched = SWALR(opt, swa_lr=SWA_LR)
    elif scheduler_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5, verbose=(verbose and is_main(rank)))

    ema = None
    if use_ema:
        base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
        ema = EMA(base_module, decay=hparams.get("ema_decay", EMA_DECAY))

    best_val = float("inf"); no_improve = 0

    for epoch in range(1, epochs+1):
        if train_sampler is not None: train_sampler.set_epoch(epoch)

        if is_main(rank):
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
            print(f"▶️ Epoch {epoch}/{epochs} | scheduler={scheduler_kind} | "
                  f"EMA={'on' if (ema is not None) else 'off'} | "
                  f"SWA={'on' if (scheduler_kind=='cosine' and swa_model is not None) else 'off'}", flush=True)

        model.train()
        running = 0.0
        bar = _make_bar(train_loader, f"Train e{epoch}", rank)
        opt.zero_grad(set_to_none=True)

        steps_processed = 0

        def optimizer_step():
            if device.type == "cuda":
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler_kind in ("onecycle", "cosine") and sched is not None:
                sched.step()
            if ema is not None:
                base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                ema.update(base_module)

        for step, batch in enumerate(bar, 1):
            iter_t0 = time.time()
            x = batch["x"].to(device, non_blocking=(device.type=="cuda")).contiguous(memory_format=torch.channels_last)
            y_mask_raw = batch["y_mask"].to(device, non_blocking=(device.type=="cuda"))
            y_mask = y_mask_raw if LABEL_SMOOTH_EPS <= 0 else y_mask_raw*(1.0 - LABEL_SMOOTH_EPS) + 0.5*LABEL_SMOOTH_EPS
            y_dbz = batch["y_dbz"].to(device, non_blocking=(device.type=="cuda"))

            with (torch.amp.autocast("cuda", dtype=(torch.bfloat16 if (device.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16), enabled=(device.type=="cuda" and MIXED_PRECISION)) if device.type=="cuda" else contextlib.nullcontext()):
                logits, dbz_pred = model(x)
                loss_mask = loss_focal(logits, y_mask) + dice_w * dice_loss(logits, y_mask)
                gate = (y_mask_raw > 0.5).float()
                l1 = F.smooth_l1_loss(dbz_pred, y_dbz, reduction="none")
                pos = gate.sum()
                loss_dbz = (l1 * gate).sum() / (pos + 1e-6)
                loss = (loss_mask + lambda_dbz * loss_dbz) / grad_accum

            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            steps_processed = step

            if step % grad_accum == 0:
                optimizer_step()

            running = (0.98 * running + 0.02 * float((loss * grad_accum).item())) if step > 1 else float((loss * grad_accum).item())

            if is_main(rank):
                mem_alloc, mem_res, mem_peak = _cuda_mem()
                cur_lr = opt.param_groups[0]["lr"]
                global_bsz = x.size(0) * (world_size if ddp_is_dist() else 1)
                dt = max(1e-6, time.time() - iter_t0)
                ips = global_bsz / dt
                bar.set_postfix(loss=f"{running:.4f}", lr=f"{cur_lr:.2e}", ips=f"{ips:.1f}/s", mem=f"{mem_alloc:.2f}G|{mem_peak:.2f}G")

        if (steps_processed % grad_accum) != 0:
            optimizer_step()

        if (scheduler_kind == "cosine") and (swa_model is not None) and (epoch >= int(max(1, epochs * SWA_START_FRAC))):
            base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
            swa_model.update_parameters(base_module)
            from torch.optim.swa_utils import SWALR
            swa_sched.step()

        if val_sampler is not None: val_sampler.set_epoch(epoch)

        def eval_model(with_ema: bool, which_model=None, desc="Val", do_tta=False) -> float:
            if with_ema and ema is not None:
                base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                ema.apply_shadow(base_module)
            m = which_model if which_model is not None else (model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model)
            m.eval()
            loss_sum = 0.0
            count_sum = 0
            barv = _make_bar(val_loader, desc, rank)
            with torch.no_grad():
                for batch in barv:
                    iter_t0 = time.time()
                    x = batch["x"].to(device, non_blocking=(device.type=="cuda")).contiguous(memory_format=torch.channels_last)
                    y_mask_raw = batch["y_mask"].to(device, non_blocking=(device.type=="cuda"))
                    y_mask = y_mask_raw if LABEL_SMOOTH_EPS <= 0 else y_mask_raw*(1.0 - LABEL_SMOOTH_EPS) + 0.5*LABEL_SMOOTH_EPS
                    y_dbz = batch["y_dbz"].to(device, non_blocking=(device.type=="cuda"))
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), enabled=MIXED_PRECISION):
                            logits, dbz_pred = _predict_tta(m, x, do_tta=do_tta)
                    else:
                        logits, dbz_pred = _predict_tta(m, x, do_tta=do_tta)
                    loss_mask = loss_focal(logits, y_mask) + dice_w * dice_loss(logits, y_mask)
                    gate = (y_mask_raw > 0.5).float()
                    l1 = F.smooth_l1_loss(dbz_pred, y_dbz, reduction="none")
                    pos = gate.sum()
                    step_loss = float((loss_mask + lambda_dbz * (l1 * gate).sum() / (pos + 1e-6)).item())
                    bs = x.size(0)
                    loss_sum += step_loss * bs
                    count_sum += bs
                    if is_main(rank):
                        mem_alloc, _, mem_peak = _cuda_mem()
                        ips = (x.size(0) * (world_size if ddp_is_dist() else 1)) / max(1e-6, time.time() - iter_t0)
                        barv.set_postfix(loss=f"{(loss_sum/max(1,count_sum)):.4f}", ips=f"{ips:.1f}/s", mem=f"{mem_alloc:.2f}G|{mem_peak:.2f}G")
            loss_sum, count_sum = weighted_reduce_sum_count(loss_sum, count_sum, device)
            if with_ema and ema is not None:
                base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                ema.restore(base_module)
            return (loss_sum / max(1.0, count_sum))

        vloss_raw = eval_model(False, desc="Val raw", do_tta=EVAL_TTA_TRAIN)
        vloss_ema = eval_model(True,  desc="Val EMA", do_tta=EVAL_TTA_TRAIN) if ema is not None else vloss_raw
        vloss_swa = float("inf")
        if (scheduler_kind == "cosine") and (swa_model is not None) and (epoch == epochs):
            vloss_swa = eval_model(False, which_model=swa_model, desc="Val SWA", do_tta=EVAL_TTA_FINAL)
        vloss = min(vloss_raw, vloss_ema, vloss_swa)

        if is_main(rank) and verbose:
            msg = f"Epoch {epoch} | val_raw={vloss_raw:.4f} | val_ema={vloss_ema:.4f}"
            if math.isfinite(vloss_swa): msg += f" | val_swa={vloss_swa:.4f}"
            msg += f" -> using {vloss:.4f}"
            print(msg, flush=True)

        if scheduler_kind == "plateau": sched.step(vloss)

        improved = vloss < best_val
        if improved:
            best_val = vloss; no_improve = 0
            if is_main(rank):
                ensure_dir(os.path.dirname(save_ckpt_path))
                if ema is not None:
                    base_module = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                    ema.apply_shadow(base_module)
                base_to_save = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
                torch.save({
                    "model": base_to_save.state_dict(),
                    "cfg": {
                        "in_ch": 15, "base": base,
                        "left_mask_px": LEFT_MASK_PX,
                        "weak_label": {
                            "mode": WEAK_LABEL_MODE,
                            "simple": dict(hue_lo=HSV_H_LO, hue_hi=HSV_H_HI, sat_min=HSV_S_MIN, val_min=HSV_V_MIN),
                            "multi_ranges": [(float(a),float(b)) for (a,b) in HSV_MULTI_RANGES],
                            "palette": dict(left_w=PALETTE_LEFT_W, samples=PALETTE_SAMPLES, dist_thr=PALETTE_DIST_THR)
                        },
                        "neighbor_minutes": NEIGHBOR_MINUTES,
                        "crop": CROP,
                        "fuse_mode": dcfg.fuse_mode
                    }
                }, save_ckpt_path)
                if ema is not None:
                    ema.restore(base_to_save)
                if verbose:
                    print(f"  ✅ Saved best checkpoint to {save_ckpt_path} (val={best_val:.4f})", flush=True)
        else:
            no_improve += 1
            if EARLY_STOP and no_improve >= EARLY_STOP_PATIENCE:
                if is_main(rank) and verbose:
                    print(f"Early stopping at epoch {epoch} (no improvement for {no_improve} epochs).", flush=True)
                break

    return best_val

# ===========================
# CPU Stage (prep): index + optional caches
# ===========================
def _init_worker(omp_threads: int = 1):
    os.environ["OMP_NUM_THREADS"]   = str(omp_threads)
    os.environ["MKL_NUM_THREADS"]   = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"]  = str(omp_threads)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

def _prep_single(path: str, left_mask_px: int, hsvp: HSVParams, do_rgb: bool, device: str):
    # Compute & write weak-label cache; optionally write RGB cache
    # NOTE: STRICT_CACHE_ONLY is ignored in cpu-prep; we always compute here.
    base = _weak_cache_key_base(path, hsvp, left_mask_px)
    w_npz, m_npy, i_npy = _weak_paths(base)
    # Only recompute if truly missing
    if not (os.path.isfile(w_npz) or (os.path.isfile(m_npy) and os.path.isfile(i_npy))):
        img = imread_rgb(path)
        m, inten = _weak_label_core(img, hsvp, left_mask_px, device=device)
        if WEAK_CACHE_COMPRESS:
            _save_weak_npz(w_npz, m, inten)
        else:
            np.save(m_npy, m.astype(np.uint8))
            np.save(i_npy, inten.astype(np.float32))
    if do_rgb and (RGB_CACHE_MODE != "off"):
        _ = imread_rgb(path)  # will write if missing
    return os.path.basename(path)

def weak_cache_status(sources, hsvp, left_mask_px):
    total = 0; have = 0; missing = []
    for src in sources:
        for it in src.items:
            total += 1
            if _weak_cache_exists_only(it["path"], hsvp, left_mask_px):
                have += 1
            else:
                missing.append(it["path"])
    return have, (total - have), missing


def stage_cpu_prep(args):
    global RGB_CACHE_MODE, WEAK_CACHE_COMPRESS, WEAK_PACK_MASK_BITS, WEAK_INTEN_DTYPE

    os.environ.setdefault("OMP_NUM_THREADS", str(max(2, (os.cpu_count() or 2)//2)))
    print("🔧 CPU Prep Stage — indexing sources (and optional cache precompute)", flush=True)

    # 1) FAST-IO (forced big/fast caches)
    if getattr(args, "fast_io", False):
        RGB_CACHE_MODE = "npy"          # bigger, faster
        WEAK_CACHE_COMPRESS = False     # store raw .npy
        WEAK_PACK_MASK_BITS = False     # no bit packing
        print("⚡ fast-io enabled: RGB_CACHE_MODE=npy, weak-compress=off, bitpack=off", flush=True)
    else:
        # 2) Compact knobs (do NOT run when fast-io is on)
        # RGB cache backend
        if getattr(args, "rgb_cache_mode", None):
            RGB_CACHE_MODE = args.rgb_cache_mode
        elif args.do_rgb_cache and RGB_CACHE_MODE == "off":
            # default to compressed npz when user wants RGB cache but didn't specify a mode
            RGB_CACHE_MODE = "npz"

        # Weak-cache size controls
        if getattr(args, "no_weak_compress", False):
            WEAK_CACHE_COMPRESS = False
        if getattr(args, "no_weak_packbits", False):
            WEAK_PACK_MASK_BITS = False
        if getattr(args, "weak_quant8", False):
            WEAK_INTEN_DTYPE = np.uint8  # quantize weak intensity to u8 inside .npz

    print(
        f"RGB_CACHE_MODE={RGB_CACHE_MODE} | weak: compress={WEAK_CACHE_COMPRESS} "
        f"packbits={WEAK_PACK_MASK_BITS} inten_dtype={'u8' if WEAK_INTEN_DTYPE is np.uint8 else 'f16'}",
        flush=True
    )

    # ---- index & summary ----
    sources = build_sources(rank=0)
    print("Summary per source:", flush=True)
    for src in sources:
        h, w = imread_rgb(src.items[0]["path"]).shape[:2]
        print(f"  {os.path.basename(src.root)}: frames={len(src.items)} size={w}x{h}", flush=True)
        # verify atlas file is readable on disk
        npz_path, _ = _atlas_cache_paths(src.root)
        try:
            _ = _load_bool_mask_npz(npz_path)
            print(f"  ✅ atlas OK at {npz_path}", flush=True)
        except Exception as e:
            print(f"  🔁 rebuilding atlas for {src.root} ({e})", flush=True)
            atlas = build_atlas_mask([it['path'] for it in src.items])
            _save_bool_mask_npz(npz_path, atlas)
            # re-verify
            _ = _load_bool_mask_npz(npz_path)
            print(f"  ✅ atlas rebuilt OK at {npz_path}", flush=True)


     # ---- precompute ----
    hsvp = HSVParams()
    # quick status before any work
    have0, miss0, missing0 = weak_cache_status(sources, hsvp, LEFT_MASK_PX)
    print(f"🔎 Weak-cache status BEFORE: have={have0} missing={miss0} of {have0+miss0}", flush=True)

    if args.do_weak_cache or args.do_rgb_cache:
        # build todo list
        todo_paths = []
        for src in sources:
            items = src.items
            if args.precompute_limit and args.precompute_limit > 0:
                items = items[:args.precompute_limit]
            todo_paths.extend([it["path"] for it in items])

        # only fill missing if requested
        if getattr(args, "precompute_missing_only", False):
            todo_paths = [p for p in todo_paths if not _weak_cache_exists_only(p, hsvp, LEFT_MASK_PX)]

        random.shuffle(todo_paths)
        work = [(p, LEFT_MASK_PX, hsvp, args.do_rgb_cache, WEAK_LABEL_DEVICE) for p in todo_paths]

        max_workers = args.prep_workers or max(1, (os.cpu_count() or 2) - 1)
        if WEAK_LABEL_DEVICE != "cpu" and max_workers != 1:
            print(
                "⚠️  Forcing --prep-workers=1 because weak labels are running on the GPU",
                flush=True,
            )
            max_workers = 1
        omp_per_worker = max(1, args.omp_per_worker)
        print(f"🧵 Spawning {max_workers} workers (OMP threads/worker={omp_per_worker}) on {len(work)} files", flush=True)

        errors = []
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(omp_per_worker,)) as ex:
            fut_map = {ex.submit(_prep_single, *task): task[0] for task in work}
            for fut in tqdm(as_completed(fut_map), total=len(fut_map), dynamic_ncols=True, mininterval=0.1, miniters=1):
                try:
                    fut.result()  # <- surface exceptions!
                except Exception as e:
                    errors.append((fut_map[fut], repr(e)))

        if errors:
            print(f"⚠️  {len(errors)} files failed during precompute. Showing up to 10:", flush=True)
            for p, msg in errors[:10]:
                print(f"   - {p} :: {msg}", flush=True)
            if getattr(args, "stop_on_error", False):
                raise RuntimeError("Stopping because --stop-on-error is set and there were precompute failures.")

        # recompute status after
        have1, miss1, _ = weak_cache_status(sources, hsvp, LEFT_MASK_PX)
        print(f"✅ CPU precompute done. Weak-cache status AFTER: have={have1} missing={miss1} of {have1+miss1}", flush=True)
    else:
        print("No precompute requested. Done.", flush=True)


# ===========================
# GPU/DDP Stage (training)
# ===========================
def stage_gpu_train(args):
    global STRICT_CACHE_ONLY, STRICT_ATLAS_ONLY
    STRICT_CACHE_ONLY = bool(args.strict_cache_only)
    STRICT_ATLAS_ONLY = bool(args.strict_atlas_only)

    world_size, rank, local_rank = ddp_setup()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main(rank):
        print("DDP:", dict(world_size=world_size, rank=rank, local_rank=local_rank), flush=True)

    # Build sources (will respect STRICT_ATLAS_ONLY)
    sources = build_sources(rank=rank)

    # Optional preflight: if strict, quickly check presence of weak caches (cheap path)
    if STRICT_CACHE_ONLY and is_main(rank):
        hsvp = HSVParams()
        total = 0; misses = 0
        for src in sources:
            for it in src.items:
                total += 1
                if not _weak_cache_exists_only(it["path"], hsvp, LEFT_MASK_PX):
                    misses += 1
        if misses > 0:
            raise RuntimeError(f"Strict mode: {misses}/{total} weak-label caches are missing. Run cpu-prep first.")

    # Make loaders with user knobs
    n_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS
    prefetch = args.prefetch_factor if args.prefetch_factor is not None else PREFETCH_FACTOR
    dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler = make_loaders(
        sources, device, rank, world_size, n_workers, prefetch
    )

    final_val = run_training({
        "base_channels": BASE_CHANNELS,
        "lambda_dbz": LAMBDA_DBZ,
        "scheduler": SCHEDULER,
        "use_warmup": USE_WARMUP,
        "warmup_steps": WARMUP_STEPS,
        "grad_clip": GRAD_CLIP_NORM,
        "use_ema": USE_EMA,
        "ema_decay": EMA_DECAY,
        "use_lr_finder": USE_LR_FINDER,
        "lr_min": LR_MIN,
        "lr_max": LR_MAX,
        "epochs": EPOCHS,
        "dice_weight": DICE_WEIGHT,
        "use_swa": (USE_SWA and SCHEDULER == "cosine"),
        "grad_accum_steps": GRAD_ACCUM_STEPS,
    }, device, rank, local_rank, world_size,
       dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler,
       save_ckpt_path=CKPT_PATH, verbose=True)

    if is_main(rank):
        print("Training complete. Best val loss:", final_val, flush=True)
        print("Model saved at:", CKPT_PATH, flush=True)

    ddp_cleanup()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ===========================
# CLI
# ===========================
def parse_args():
    p = argparse.ArgumentParser(description="Two-stage (CPU prep / GPU train) radar cleaner (strict-cache + fast RGB)")
    p.add_argument("--stage", choices=["cpu-prep","gpu-train"], default="gpu-train")
    # CPU prep options
    p.add_argument("--do-weak-cache", action="store_true", help="Precompute weak labels cache (.npz or .npy)")
    p.add_argument("--do-rgb-cache",  action="store_true", help="Predecode & cache RGB as .npy/.npz")
    p.add_argument("--precompute-limit", type=int, default=None, help="Limit items per source during precompute (0/all)")
    p.add_argument("--prep-workers", type=int, default=None, help="CPU-prep parallel workers (default: cpu_count-1)")
    p.add_argument("--omp-per-worker", type=int, default=1, help="OpenMP/MKL threads per worker")
    p.add_argument("--fast-io", action="store_true", help="Use faster, larger caches (RGB npy, no weak compress/bitpack)")
    p.add_argument("--prep-on-gpu", action="store_true", help="Run weak-label prep on GPU when available")
    # Shared paths
    p.add_argument("--radar-dirs", type=str, default=None, help="Comma-separated radar folders, e.g. /data/njk,/data/nkm")
    p.add_argument("--work-dir", type=str, default=os.environ.get("WORK_DIR", WORK_DIR), help="Where to save checkpoints/logs")
    p.add_argument("--cache-root", type=str, default=os.environ.get("CACHE_ROOT", CACHE_ROOT), help="Where to store caches")
    # Train strict knobs + loader knobs
    p.add_argument("--strict-cache-only", action="store_true", help="Error if weak-label cache missing (no recompute)")
    p.add_argument("--strict-atlas-only", action="store_true", help="Error if atlas cache missing (no rebuild)")
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (override default)")
    p.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (override default)")
    p.add_argument("--rgb-cache-mode", choices=["off","npz","npy"], default=None,
               help="RGB cache backend; default: npz when --do-rgb-cache and not --fast-io")
    p.add_argument("--weak-quant8", action="store_true",
               help="Quantize weak intensities to uint8 in .npz (smaller)")
    p.add_argument("--no-weak-packbits", action="store_true",
               help="Disable mask bit-packing (debug; larger)")
    p.add_argument("--no-weak-compress", action="store_true",
               help="Store weak cache as raw .npy instead of .npz (debug; larger)")
    p.add_argument("--precompute-missing-only", action="store_true",
                   help="During cpu-prep, only build weak/RGB caches that are missing.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort cpu-prep if any file fails to cache.")


    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    requested_device = "cuda" if getattr(args, "prep_on_gpu", False) else "cpu"
    if requested_device != "cpu" and not torch.cuda.is_available():
        print("⚠️  --prep-on-gpu requested but CUDA is unavailable; falling back to CPU.", flush=True)
        requested_device = "cpu"
    WEAK_LABEL_DEVICE = requested_device
    if WEAK_LABEL_DEVICE != "cpu":
        print(f"🖥️  Weak-label preprocessing device: {WEAK_LABEL_DEVICE}", flush=True)

    if args.radar_dirs:
        RADAR_DIRS[:] = [p.strip() for p in args.radar_dirs.split(",") if p.strip()]
    WORK_DIR = args.work_dir
    CACHE_ROOT = args.cache_root
    CACHE_DIR = os.path.join(CACHE_ROOT, "cache")
    RGB_CACHE_DIR = os.path.join(CACHE_ROOT, "img_npy_cache")
    for _d in [WORK_DIR, CACHE_DIR, RGB_CACHE_DIR]:
        os.makedirs(_d, exist_ok=True)

    if args.stage == "cpu-prep":
        # In cpu-prep we might want to enforce the atlas exists too.
        # If STRICT_ATLAS_ONLY passed here and cache missing, index_source will error.
        stage_cpu_prep(args)

    elif args.stage == "gpu-train":
        stage_gpu_train(args)