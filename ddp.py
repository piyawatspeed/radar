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
#   * weak caches use .npz with bitpack/float16 (or raw npy when --fast-io) and
#     auto-detect legacy bit-packed loads even if runtime bitpacking is off
# - Train perf:
#   * channels-last, cudnn.benchmark, non_blocking H2D, configurable workers/prefetch
#   * AMP via torch.amp.*, optional torch.compile on Ampere+/Ada (L4 OK)
# - Eval:
#   * EMA, optional SWA with cosine; TTA off during train, optional final TTA
# - CLI examples (single L4):
#   cpu-prep (fast): python ddp.py --stage cpu-prep --do-weak-cache --do-rgb-cache --fast-io --radar-dirs /path/njk,/path/nkm --cache-root /big/cache
#   train (strict):  python ddp.py --stage gpu-train --strict-cache-only --strict-atlas-only --radar-dirs /path/njk,/path/nkm --cache-root /big/cache

from __future__ import annotations
import os, re, glob, math, time, random, gc, contextlib, argparse, io, hashlib
from functools import partial
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterable, Any
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image, ImageFile  # pillow-simd will override if installed

from pyproj import Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm.auto import tqdm

try:
    from torch.cuda import CudaError as _TorchCudaError  # type: ignore
except Exception:  # pragma: no cover - fallback when CUDA is unavailable
    _TorchCudaError = RuntimeError

_TorchAcceleratorError = getattr(torch, "AcceleratorError", RuntimeError)


_PIN_MEMORY_ENABLED = torch.cuda.is_available()
_PIN_MEMORY_WARNED = False
_CUDA_FAILED = False
_CUDA_WARNED = False


def _maybe_pin_memory(tensor: torch.Tensor) -> torch.Tensor:
    global _PIN_MEMORY_ENABLED, _PIN_MEMORY_WARNED
    if not _PIN_MEMORY_ENABLED:
        return tensor
    try:
        return tensor.pin_memory()
    except (_TorchCudaError, _TorchAcceleratorError, RuntimeError) as exc:
        if not _PIN_MEMORY_WARNED:
            print(f"⚠️  pin_memory disabled after failure: {exc}", flush=True)
            _PIN_MEMORY_WARNED = True
        _PIN_MEMORY_ENABLED = False
        return tensor


def _cuda_is_usable() -> bool:
    global _CUDA_FAILED, _CUDA_WARNED, _PIN_MEMORY_ENABLED
    if _CUDA_FAILED:
        return False
    if not torch.cuda.is_available():
        _PIN_MEMORY_ENABLED = False
        _CUDA_FAILED = True
        return False
    try:
        torch.cuda.current_device()
        return True
    except (_TorchCudaError, _TorchAcceleratorError, RuntimeError) as exc:
        if not _CUDA_WARNED:
            print(f"⚠️  CUDA unavailable; falling back to CPU: {exc}", flush=True)
            _CUDA_WARNED = True
        _PIN_MEMORY_ENABLED = False
        _CUDA_FAILED = True
        return False


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
    "/teamspace/studios/this_studio/15mins/nkm",  # Radar A (NKM)
    "/teamspace/studios/this_studio/15mins/njk",  # Radar B (NJK)
]
RADAR_A_ID = "NKM"
RADAR_B_ID = "NJK"
RADAR_A_LATLON = (13.737729372744127, 100.3588712604009)
RADAR_B_LATLON = (13.834873535452727, 100.84641070939088)
RADAR_A_RANGE_KM = 120.0
RADAR_B_RANGE_KM = 120.0
RADAR_A_CENTER_OVERRIDE = None
RADAR_B_CENTER_OVERRIDE = None
WARP_CACHE_BASENAME = "warp_B_to_A.npy"
TIMESTAMP_REGEX = r"(\d{12})(?=\D*$)"   # matches YYYYMMDDHHMM in filenames

# Where to save results & caches
WORK_DIR = "/teamspace/studios/this_studio/work"
CACHE_ROOT = os.environ.get("CACHE_ROOT", "/teamspace/studios/this_studio/tmp")
CACHE_DIR = os.path.join(CACHE_ROOT, "cache")             # weak-label/atlas caches
RGB_CACHE_DIR = os.path.join(CACHE_ROOT, "img_npy_cache") # RGB caches
for _d in [WORK_DIR, CACHE_DIR, RGB_CACHE_DIR]:
    os.makedirs(_d, exist_ok=True)


def warp_cache_path() -> str:
    return os.path.join(CACHE_DIR, WARP_CACHE_BASENAME)

# Weak-cache sharding / shared store configuration
def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in {"0", "false", "no", "off", ""}


WEAK_CACHE_BACKEND = os.environ.get("WEAK_CACHE_BACKEND", "").strip().lower() or None
WEAK_CACHE_SHARD_BY = os.environ.get("WEAK_CACHE_SHARD_BY", "day").strip().lower()
try:
    WEAK_CACHE_LMDB_MAP_SIZE = int(os.environ.get("WEAK_CACHE_LMDB_MAP_SIZE", str(1 << 34)))
except ValueError:
    WEAK_CACHE_LMDB_MAP_SIZE = 1 << 34
_WEAK_CACHE_WRITE_FILES_ENV = os.environ.get("WEAK_CACHE_WRITE_FILES")
WEAK_CACHE_WRITE_FILES = _env_flag("WEAK_CACHE_WRITE_FILES", True)
WEAK_CACHE_WRITE_FILES_FORCED_ON = bool(_WEAK_CACHE_WRITE_FILES_ENV and WEAK_CACHE_WRITE_FILES)

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
    backend = "nccl" if _cuda_is_usable() else "gloo"
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
    target_ext = os.path.splitext(target_path)[1]
    tmp_suffix = target_ext if target_ext else ".tmp"
    import tempfile as _tf
    with _tf.NamedTemporaryFile(delete=False, dir=tmp_dir, suffix=tmp_suffix) as tf:
        tmp_path = tf.name
    extra_tmp = tmp_path + target_ext if target_ext else None
    try:
        writer_fn(tmp_path)
        actual_path = tmp_path
        if not os.path.exists(actual_path) and extra_tmp and os.path.exists(extra_tmp):
            actual_path = extra_tmp
        if actual_path != tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
        os.replace(actual_path, target_path)
    except Exception:
        for cand in (tmp_path, extra_tmp):
            if not cand:
                continue
            try:
                if os.path.exists(cand):
                    os.remove(cand)
            except Exception:
                pass
        raise

def _write_rgb_cache(path: str, arr: np.ndarray) -> None:
    if RGB_CACHE_MODE == "off":
        return
    cpath = _rgb_cache_path(path)
    if os.path.isfile(cpath):
        return

    arr_to_write = np.ascontiguousarray(arr)

    def _writer_np(p):
        np.save(p, arr_to_write)

    def _writer_npz(p):
        np.savez_compressed(p, arr=arr_to_write)

    try:
        _atomic_write(cpath, _writer_np if RGB_CACHE_MODE == "npy" else _writer_npz)
    except Exception:
        pass


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
        try:
            img.draft("RGB", img.size)
        except Exception:
            pass
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    _write_rgb_cache(path, arr)
    return arr

# --------- Weak-label compressed cache (.npz with bit-packed mask + float16 inten) ----------
@dataclass(frozen=True)
class WeakCacheKey:
    stem: str
    param_tokens: Tuple[str, ...]
    base_path: str

    @property
    def param_str(self) -> str:
        return ".".join(self.param_tokens)

    @property
    def signature(self) -> str:
        return f"{self.stem}::{self.param_str}" if self.param_tokens else self.stem

    def timestamp_token(self) -> Optional[str]:
        m = re.search(r"\d{12}", self.stem)
        return m.group(0) if m else None

    def shard_key(self, shard_by: str = "day") -> str:
        ts = self.timestamp_token()
        if shard_by == "day" and ts:
            return ts[:8]
        if shard_by == "hour" and ts:
            return ts[:10]
        if shard_by == "month" and ts:
            return ts[:6]
        return ts or "misc"

    @classmethod
    def from_components(cls, stem: str, param_tokens: Iterable[str], cache_dir: str) -> "WeakCacheKey":
        pt = tuple(param_tokens)
        if pt:
            param_str = ".".join(pt)
            base_path = os.path.join(cache_dir, f"{stem}.{param_str}.weak")
        else:
            base_path = os.path.join(cache_dir, f"{stem}.weak")
        return cls(stem=stem, param_tokens=pt, base_path=base_path)


def _weak_cache_key_from_legacy_path(path: str, cache_dir: Optional[str] = None) -> WeakCacheKey:
    fname = os.path.basename(path)
    for suf in (".w.npz", ".mask.npy", ".inten.npy"):
        if fname.endswith(suf):
            fname = fname[:-len(suf)]
            break
    if fname.endswith(".weak"):
        fname = fname[:-5]
    stem = fname
    params: Tuple[str, ...] = ()
    marker = ".mode-"
    if marker in fname:
        idx = fname.index(marker)
        stem = fname[:idx]
        param_str = fname[idx + 1 :]
        params = tuple(p for p in param_str.split(".") if p)
    cache_dir = cache_dir or os.path.dirname(path) or CACHE_DIR
    return WeakCacheKey.from_components(stem, params, cache_dir)


def _weak_cache_key_base(path: str, hsvp, left_mask_px: int) -> WeakCacheKey:
    base = os.path.splitext(os.path.basename(path))[0]
    token = _image_timestamp_token(path)
    stem = base if token in base else f"{base}.{token}"
    params = (
        f"mode-{_tokenize_value(WEAK_LABEL_MODE)}",
        f"h-{_tokenize_value(hsvp.hue_lo)}-{_tokenize_value(hsvp.hue_hi)}",
        f"s-{_tokenize_value(hsvp.sat_min)}",
        f"v-{_tokenize_value(hsvp.val_min)}",
        f"left-{_tokenize_value(left_mask_px)}",
    )
    key = WeakCacheKey.from_components(stem, params, CACHE_DIR)
    store = get_weak_cache_store()
    if store:
        return store.normalize_key(key)
    return key


def _weak_paths(key: WeakCacheKey):
    store = get_weak_cache_store()
    if store:
        return store.paths_for_key(key)
    base = key.base_path
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


def _weak_record_components(mask: np.ndarray, inten: np.ndarray):
    H, W = mask.shape
    iq8 = 1 if WEAK_INTEN_DTYPE is np.uint8 else 0
    if iq8:
        inten_disk = np.clip(np.rint(inten * 255.0), 0, 255).astype(np.uint8)
    else:
        inten_disk = inten.astype(WEAK_INTEN_DTYPE if WEAK_CACHE_COMPRESS else np.float32)
    if WEAK_PACK_MASK_BITS:
        mask_payload = np.packbits(mask.astype(np.uint8), axis=1)
    else:
        mask_payload = mask.astype(np.uint8)
    return {
        "shape": np.array([H, W], np.int32),
        "mask": mask_payload,
        "inten": inten_disk,
        "iq8": np.array([iq8], np.uint8),
    }


def _weak_record_to_bytes(mask: np.ndarray, inten: np.ndarray) -> bytes:
    payload = _weak_record_components(mask, inten)
    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return buf.getvalue()


def _weak_record_from_arrays(z) -> Tuple[np.ndarray, np.ndarray]:
    H, W = map(int, z["shape"])
    m = z["mask"]
    iq8 = int(z["iq8"][0]) if "iq8" in getattr(z, "files", z) else 0

    # Older caches may have been saved with bit-packed masks even if the current
    # runtime has WEAK_PACK_MASK_BITS disabled. Rely on the stored tensor shape
    # instead of the global flag so we can transparently read legacy artifacts.
    if m.ndim == 2 and m.shape[1] != W:
        mask = np.unpackbits(m, axis=1)[:, :W].astype(bool)
    else:
        mask = (np.asarray(m).astype(np.uint8) > 0)
    inten_raw = z["inten"]
    inten = (inten_raw.astype(np.float32) / 255.0) if iq8 else inten_raw.astype(np.float32)
    return mask, inten


def _weak_record_from_bytes(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(io.BytesIO(payload)) as z:
        return _weak_record_from_arrays(z)


class _WeakCacheBackend:
    def __init__(self, root: str):
        self.root = root

    def get(self, shard_id: str, store_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def set(self, shard_id: str, store_key: str, mask: np.ndarray, inten: np.ndarray):
        raise NotImplementedError

    def exists(self, shard_id: str, store_key: str) -> bool:
        raise NotImplementedError

    def paths_for_key(self, key: WeakCacheKey) -> Tuple[str, str, str]:
        base = key.base_path
        return base + ".w.npz", base + ".mask.npy", base + ".inten.npy"

    def close(self):
        pass


class _LMDBWeakCacheBackend(_WeakCacheBackend):
    def __init__(self, root: str, map_size: int):
        super().__init__(root)
        try:
            import lmdb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("lmdb backend requested but python-lmdb is not installed") from exc
        self._lmdb = lmdb
        ensure_dir(root)
        self._envs: Dict[str, Any] = {}
        self._map_size = map_size

    def _env(self, shard_id: str):
        env = self._envs.get(shard_id)
        if env is None:
            path = os.path.join(self.root, f"{shard_id}.lmdb")
            ensure_dir(os.path.dirname(path) or ".")
            env = self._lmdb.open(path, map_size=self._map_size, subdir=True, create=True, lock=True, readahead=False)
            self._envs[shard_id] = env
        return env

    def get(self, shard_id: str, store_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        env = self._env(shard_id)
        with env.begin(write=False) as txn:
            payload = txn.get(store_key.encode("utf-8"))
        if not payload:
            return None
        return _weak_record_from_bytes(bytes(payload))

    def set(self, shard_id: str, store_key: str, mask: np.ndarray, inten: np.ndarray):
        env = self._env(shard_id)
        payload = _weak_record_to_bytes(mask, inten)
        with env.begin(write=True) as txn:
            txn.put(store_key.encode("utf-8"), payload)

    def exists(self, shard_id: str, store_key: str) -> bool:
        env = self._env(shard_id)
        with env.begin(write=False) as txn:
            return txn.get(store_key.encode("utf-8")) is not None

    def close(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()


class _ZarrWeakCacheBackend(_WeakCacheBackend):
    def __init__(self, root: str):
        super().__init__(root)
        try:
            import zarr  # type: ignore
        except ImportError as exc:
            raise RuntimeError("zarr backend requested but zarr is not installed") from exc
        self._zarr = zarr
        ensure_dir(root)
        store = zarr.DirectoryStore(root)
        self._root = zarr.group(store=store, overwrite=False)

    def _group(self, shard_id: str):
        return self._root.require_group(shard_id)

    def get(self, shard_id: str, store_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        grp = self._group(shard_id)
        if store_key not in grp:
            return None
        node = grp[store_key]
        mask = node["mask"][...].astype(bool)
        inten = node["inten"][...].astype(np.float32)
        return mask, inten

    def set(self, shard_id: str, store_key: str, mask: np.ndarray, inten: np.ndarray):
        grp = self._group(shard_id)
        node = grp.require_group(store_key)
        if "mask" in node:
            del node["mask"]
        if "inten" in node:
            del node["inten"]
        node.create_dataset("mask", data=mask.astype(bool), chunks=True)
        node.create_dataset("inten", data=inten.astype(np.float32), chunks=True)

    def exists(self, shard_id: str, store_key: str) -> bool:
        grp = self._group(shard_id)
        return store_key in grp


class _ParquetWeakCacheBackend(_WeakCacheBackend):
    def __init__(self, root: str):
        super().__init__(root)
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:
            raise RuntimeError("parquet backend requested but pyarrow is not installed") from exc
        self._pa = pa
        self._pq = pq
        ensure_dir(root)

    def _key_path(self, shard_id: str, store_key: str) -> str:
        ensure_dir(os.path.join(self.root, shard_id))
        digest = hashlib.sha1(store_key.encode("utf-8")).hexdigest()
        return os.path.join(self.root, shard_id, f"{digest}.parquet")

    def get(self, shard_id: str, store_key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        path = self._key_path(shard_id, store_key)
        if not os.path.isfile(path):
            return None
        table = self._pq.read_table(path)
        if table.num_rows == 0:
            return None
        payload = table.column("payload")[0].as_buffer().to_pybytes()
        return _weak_record_from_bytes(payload)

    def set(self, shard_id: str, store_key: str, mask: np.ndarray, inten: np.ndarray):
        payload = _weak_record_to_bytes(mask, inten)
        path = self._key_path(shard_id, store_key)
        table = self._pa.table({
            "key": self._pa.array([store_key]),
            "payload": self._pa.array([self._pa.py_buffer(payload)], type=self._pa.binary()),
        })
        self._pq.write_table(table, path)

    def exists(self, shard_id: str, store_key: str) -> bool:
        path = self._key_path(shard_id, store_key)
        return os.path.isfile(path)


class WeakCacheStore:
    """Shared weak-cache store supporting LMDB/Zarr/Parquet backends."""

    def __init__(self, root: str, backend: Optional[str], shard_by: str = "day", lmdb_map_size: int = 1 << 34):
        backend = (backend or "").strip().lower()
        self.enabled = bool(backend)
        self.root = os.path.join(root, "weak_shards")
        self.backend_name = backend
        self.shard_by = shard_by
        self._backend: Optional[_WeakCacheBackend] = None
        if not self.enabled:
            return
        ensure_dir(self.root)
        if backend == "lmdb":
            self._backend = _LMDBWeakCacheBackend(self.root, map_size=lmdb_map_size)
        elif backend == "zarr":
            self._backend = _ZarrWeakCacheBackend(self.root)
        elif backend == "parquet":
            self._backend = _ParquetWeakCacheBackend(self.root)
        else:
            raise ValueError(f"Unsupported weak-cache backend: {backend}")

    def close(self):
        if self._backend:
            self._backend.close()

    def normalize_key(self, key: WeakCacheKey) -> WeakCacheKey:
        # Hook for future customization (e.g., slugify stem). Currently no-op.
        return key

    def paths_for_key(self, key: WeakCacheKey) -> Tuple[str, str, str]:
        if self._backend:
            return self._backend.paths_for_key(key)
        base = key.base_path
        return base + ".w.npz", base + ".mask.npy", base + ".inten.npy"

    def _store_key(self, key: WeakCacheKey) -> str:
        return key.signature

    def _shard_id(self, key: WeakCacheKey) -> str:
        return key.shard_key(self.shard_by)

    def get(self, key: WeakCacheKey) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not (self.enabled and self._backend):
            return None
        shard = self._shard_id(key)
        return self._backend.get(shard, self._store_key(key))

    def set(self, key: WeakCacheKey, mask: np.ndarray, inten: np.ndarray):
        if not (self.enabled and self._backend):
            return
        shard = self._shard_id(key)
        self._backend.set(shard, self._store_key(key), mask, inten)

    def exists(self, key: WeakCacheKey) -> bool:
        if not (self.enabled and self._backend):
            return False
        shard = self._shard_id(key)
        return self._backend.exists(shard, self._store_key(key))


_WEAK_CACHE_STORE: Optional[WeakCacheStore] = None


def _maybe_auto_disable_weak_files(store: Optional[WeakCacheStore]):
    global WEAK_CACHE_WRITE_FILES
    if not store or not store.enabled:
        return
    if WEAK_CACHE_WRITE_FILES_FORCED_ON:
        return
    if WEAK_CACHE_WRITE_FILES:
        WEAK_CACHE_WRITE_FILES = False


def get_weak_cache_store() -> Optional[WeakCacheStore]:
    global _WEAK_CACHE_STORE
    if _WEAK_CACHE_STORE is not None:
        store = _WEAK_CACHE_STORE if _WEAK_CACHE_STORE.enabled else None
        if store:
            _maybe_auto_disable_weak_files(store)
        return store
    if WEAK_CACHE_BACKEND:
        try:
            _WEAK_CACHE_STORE = WeakCacheStore(
                CACHE_DIR,
                backend=WEAK_CACHE_BACKEND,
                shard_by=WEAK_CACHE_SHARD_BY,
                lmdb_map_size=WEAK_CACHE_LMDB_MAP_SIZE,
            )
        except Exception as exc:
            print(f"⚠️ Weak-cache store init failed: {exc}", flush=True)
            _WEAK_CACHE_STORE = None
    else:
        _WEAK_CACHE_STORE = None
    store = _WEAK_CACHE_STORE if (_WEAK_CACHE_STORE and _WEAK_CACHE_STORE.enabled) else None
    if store:
        _maybe_auto_disable_weak_files(store)
    return store

def _save_weak_npz(path: str, mask: np.ndarray, inten: np.ndarray):
    payload = _weak_record_components(mask, inten)

    def _writer(p):
        np.savez_compressed(p, **payload)

    _atomic_write(path, _writer)

def _load_weak_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return _weak_record_from_arrays(z)


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
    use_cuda = device is not None and str(device).startswith("cuda") and _cuda_is_usable()
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
    use_cuda = device is not None and str(device).startswith("cuda") and _cuda_is_usable()
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
    key = _weak_cache_key_base(path, hsvp, left_mask_px)
    store = get_weak_cache_store()
    if store and store.exists(key):
        return True
    w_npz, m_npy, i_npy = _weak_paths(key)
    if os.path.isfile(w_npz) or (os.path.isfile(m_npy) and os.path.isfile(i_npy)):
        return True
    if WEAK_CACHE_FUZZY_READ:
        # any older cache for same basename counts as “present” for strict reuse
        return len(_weak_fuzzy_candidates(path)) > 0
    return False

def get_weak_label_cached(path: str, hsvp: HSVParams, left_mask_px: int = LEFT_MASK_PX, device: Optional[str] = None):
    key = _weak_cache_key_base(path, hsvp, left_mask_px)
    store = get_weak_cache_store()
    if store:
        data = store.get(key)
        if data is not None:
            return data
    w_npz, m_npy, i_npy = _weak_paths(key)

    # Reuse if present
    if os.path.isfile(w_npz):
        try:
            data = _load_weak_npz(w_npz)
            if store:
                try: store.set(key, *data)
                except Exception: pass
            return data
        except Exception:
            pass

    if os.path.isfile(m_npy) and os.path.isfile(i_npy):
        try:
            m = np.load(m_npy, mmap_mode="r").astype(bool)
            inten = np.load(i_npy, mmap_mode="r").astype(np.float32)
            if WEAK_CACHE_COMPRESS and WEAK_CACHE_WRITE_FILES:
                try:
                    _save_weak_npz(w_npz, m, inten)
                except Exception:
                    pass
            elif WEAK_CACHE_WRITE_FILES and not WEAK_CACHE_COMPRESS:
                try:
                    np.save(m_npy, m.astype(np.uint8))
                    np.save(i_npy, inten.astype(np.float32))
                except Exception:
                    pass
            if store:
                try: store.set(key, m, inten)
                except Exception: pass
            return m, inten
        except Exception:
            pass

    # Fuzzy reuse: accept any *.weak.w.npz for same basename (mtime/param drift)
    if WEAK_CACHE_FUZZY_READ:
        cands = _weak_fuzzy_candidates(path)
        for cp in cands:
            try:
                data = _load_weak_npz(cp)
                if store:
                    try: store.set(key, *data)
                    except Exception: pass
                return data
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
    if device_str != "cpu" and not _cuda_is_usable():
        device_str = "cpu"
    m, inten = _weak_label_core(img, hsvp, left_mask_px, device=device_str)
    if store:
        try: store.set(key, m, inten)
        except Exception: pass
    if WEAK_CACHE_COMPRESS and WEAK_CACHE_WRITE_FILES:
        try: _save_weak_npz(w_npz, m, inten)
        except Exception: pass
    elif WEAK_CACHE_WRITE_FILES and not WEAK_CACHE_COMPRESS:
        try:
            np.save(m_npy, m.astype(np.uint8))
            np.save(i_npy, inten.astype(np.float32))
        except Exception:
            pass
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
# Geometry helpers
# ===========================
def compute_center_px(h: int, w: int, left_mask_px: int, override: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    if override is not None:
        return float(override[0]), float(override[1])
    w_eff = max(1.0, float(w - left_mask_px))
    cx = float(left_mask_px) + 0.5 * w_eff
    cy = 0.5 * float(h)
    return cx, cy


def compute_mpp(h: int, w: int, left_mask_px: int, range_km: float) -> float:
    w_eff = max(1.0, float(w - left_mask_px))
    r_px = max(w_eff, float(h)) * 0.5
    return (range_km * 1000.0) / max(r_px, 1e-6)


def build_warp_grid(
    shape_a: Tuple[int, int],
    shape_b: Tuple[int, int],
    left_mask_px: int,
    radar_a_latlon: Tuple[float, float],
    radar_b_latlon: Tuple[float, float],
    range_a_km: float,
    range_b_km: float,
    center_override_a: Optional[Tuple[float, float]] = None,
    center_override_b: Optional[Tuple[float, float]] = None,
):
    hA, wA = map(int, shape_a)
    hB, wB = map(int, shape_b)
    cxA, cyA = compute_center_px(hA, wA, left_mask_px, center_override_a)
    cxB, cyB = compute_center_px(hB, wB, left_mask_px, center_override_b)
    mppA = compute_mpp(hA, wA, left_mask_px, range_a_km)
    mppB = compute_mpp(hB, wB, left_mask_px, range_b_km)

    proj = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=aeqd +lat_0={radar_a_latlon[0]} +lon_0={radar_a_latlon[1]} +datum=WGS84",
        always_xy=True,
    )
    dx_b, dy_b = proj.transform(radar_b_latlon[1], radar_b_latlon[0])

    yy, xx = np.mgrid[0:hA, 0:wA].astype(np.float32)
    xA_m = (xx - cxA).astype(np.float32) * mppA
    # Image rows increase southward, but AEQD's Y axis increases northward.
    # Flip the sign when moving between pixel rows and metric northings so
    # that the warp aligns properly in latitude.
    yA_m = (cyA - yy).astype(np.float32) * mppA

    inv_mppB = 1.0 / max(mppB, 1e-6)
    xB_pix = (xA_m - dx_b) * inv_mppB + cxB
    yB_pix = cyB - (yA_m - dy_b) * inv_mppB

    norm_x = ((xB_pix + 0.5) / max(float(wB), 1.0)) * 2.0 - 1.0
    norm_y = ((yB_pix + 0.5) / max(float(hB), 1.0)) * 2.0 - 1.0
    grid = np.stack([norm_x, norm_y], axis=-1).astype(np.float32)

    meta = {
        "cxA": float(cxA),
        "cyA": float(cyA),
        "cxB": float(cxB),
        "cyB": float(cyB),
        "mpp_A": float(mppA),
        "mpp_B": float(mppB),
        "W_eff_A": float(wA - left_mask_px),
        "W_eff_B": float(wB - left_mask_px),
    }
    return grid, meta


# ===========================
# CLI helpers
# ===========================
def _parse_center_tuple(value: str) -> Tuple[float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Center must be specified as 'cx,cy'.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid center tuple '{value}'") from exc


# ===========================
# Helper channels
# ===========================
def helper_channels(h: int, w: int, center: Optional[Tuple[float, float]] = None):
    yy, xx = np.mgrid[0:h, 0:w]
    if center is None:
        cx, cy = (w // 2, h // 2)
    else:
        cx, cy = center
    dx = (xx - cx).astype(np.float32)
    dy = (yy - cy).astype(np.float32)
    r = np.sqrt(dx * dx + dy * dy)
    r_norm = r / (r.max() + 1e-6)
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
    warp_cache_path: Optional[str] = None
    radar_a_latlon: Tuple[float, float] = field(default_factory=lambda: RADAR_A_LATLON)
    radar_b_latlon: Tuple[float, float] = field(default_factory=lambda: RADAR_B_LATLON)
    radar_a_range_km: float = field(default_factory=lambda: RADAR_A_RANGE_KM)
    radar_b_range_km: float = field(default_factory=lambda: RADAR_B_RANGE_KM)
    center_override_a: Optional[Tuple[float, float]] = None
    center_override_b: Optional[Tuple[float, float]] = None
    do_inpaint: bool = False
    inpaint_frac: float = 0.0

class TwoRadarFusionDataset(Dataset):
    def __init__(self, sources: List[SourceData], split: str = "train", val_split: float = VAL_SPLIT, dcfg: Optional[DataConfig] = None):
        assert len(sources) == 2, "Provide exactly two radar sources."
        self.A, self.B = sources
        self.dcfg = dcfg or DataConfig()
        if self.dcfg.weak_label_device != "cpu" and not _cuda_is_usable():
            self.dcfg.weak_label_device = "cpu"
        self.left_mask_px = int(self.dcfg.left_mask_px)
        self.warp_path = self.dcfg.warp_cache_path or warp_cache_path()
        self.radar_a_latlon = self.dcfg.radar_a_latlon
        self.radar_b_latlon = self.dcfg.radar_b_latlon
        self.radar_a_range_km = float(self.dcfg.radar_a_range_km)
        self.radar_b_range_km = float(self.dcfg.radar_b_range_km)
        self.center_override_a = self.dcfg.center_override_a
        self.center_override_b = self.dcfg.center_override_b
        self.do_inpaint = bool(self.dcfg.do_inpaint)
        self.inpaint_frac = max(0.0, float(self.dcfg.inpaint_frac))
        self._warp_base: Optional[torch.Tensor] = None
        self._warp_meta: Optional[Dict[str, float]] = None
        self._warp_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._warp_shape: Optional[Tuple[int, int]] = None
        self._warp_warned = False

        step = int(self.dcfg.neighbor_minutes)
        if step <= 0:
            raise ValueError("neighbor_minutes must be positive")

        offsets = (-step, 0, step)
        ts_a = set(self.A.ts_to_path.keys())
        ts_b = set(self.B.ts_to_path.keys())
        common_ts = sorted(ts_a & ts_b)
        strict_samples: Dict[int, Dict[str, List[int]]] = {}

        for ts in common_ts:
            dt = ts_to_dt(ts)
            if dt.minute % step != 0:
                continue

            per_src: Dict[str, List[int]] = {}
            valid = True
            for label, source in (("A", self.A), ("B", self.B)):
                seq: List[int] = []
                for off in offsets:
                    t = add_minutes(ts, off)
                    if t not in source.ts_to_path:
                        valid = False
                        break
                    seq.append(t)
                if not valid:
                    break
                per_src[label] = seq
            if valid:
                strict_samples[ts] = per_src

        ts_all = sorted(strict_samples.keys())
        if not ts_all:
            raise RuntimeError("No synchronized radar pairs with full neighbor context were found.")

        n = len(ts_all)
        n_val = max(1, int(n * val_split))
        selected = ts_all[:n - n_val] if split == "train" else ts_all[n - n_val:]
        if not selected:
            split_name = "training" if split == "train" else "validation"
            raise RuntimeError(f"No samples available for {split_name} split under strict timestamp alignment.")

        self.ts_list = selected
        self._sample_map = {ts: strict_samples[ts] for ts in selected}
        self._hc_cache: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self): return len(self.ts_list)

    def _rand_view(self):
        flip_h = bool(torch.randint(0, 2, (1,)).item())
        flip_v = bool(torch.randint(0, 2, (1,)).item())
        k = int(torch.randint(0, 4, (1,)).item())
        return flip_h, flip_v, k

    def _apply_view(self, tensor: torch.Tensor, view):
        flip_h, flip_v, k = view
        if flip_h:
            tensor = torch.flip(tensor, dims=(-1,))
        if flip_v:
            tensor = torch.flip(tensor, dims=(-2,))
        if k:
            tensor = torch.rot90(tensor, k, dims=(-2, -1))
        return tensor

    def _helper_center(self, H: int, W: int) -> Tuple[float, float]:
        return compute_center_px(H, W, self.left_mask_px, self.center_override_a)

    def _ensure_warp_base(self, shape_a: Tuple[int, int], shape_b: Tuple[int, int]):
        if self._warp_base is not None:
            return
        path = self.warp_path
        base_shape_a = tuple(map(int, shape_a))
        base_shape_b = tuple(map(int, shape_b))
        if path and os.path.isfile(path):
            arr = np.load(path)
            self._warp_base = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            self._warp_shape = (int(arr.shape[0]), int(arr.shape[1]))
            if self._warp_shape != base_shape_a:
                grid_np, meta = build_warp_grid(
                    base_shape_a,
                    base_shape_b,
                    self.left_mask_px,
                    self.radar_a_latlon,
                    self.radar_b_latlon,
                    self.radar_a_range_km,
                    self.radar_b_range_km,
                    center_override_a=self.center_override_a,
                    center_override_b=self.center_override_b,
                )
                self._warp_base = torch.from_numpy(grid_np)
                self._warp_shape = (grid_np.shape[0], grid_np.shape[1])
                self._warp_meta = meta
                try:
                    ensure_dir(os.path.dirname(path))
                    np.save(path, grid_np)
                except Exception:
                    pass
                if not self._warp_warned:
                    print("⚠️  warp grid resized to match current geometry", flush=True)
                    self._warp_warned = True
            else:
                cxA, cyA = compute_center_px(base_shape_a[0], base_shape_a[1], self.left_mask_px, self.center_override_a)
                cxB, cyB = compute_center_px(base_shape_b[0], base_shape_b[1], self.left_mask_px, self.center_override_b)
                mppA = compute_mpp(base_shape_a[0], base_shape_a[1], self.left_mask_px, self.radar_a_range_km)
                mppB = compute_mpp(base_shape_b[0], base_shape_b[1], self.left_mask_px, self.radar_b_range_km)
                self._warp_meta = {
                    "cxA": float(cxA),
                    "cyA": float(cyA),
                    "cxB": float(cxB),
                    "cyB": float(cyB),
                    "mpp_A": float(mppA),
                    "mpp_B": float(mppB),
                    "W_eff_A": float(base_shape_a[1] - self.left_mask_px),
                    "W_eff_B": float(base_shape_b[1] - self.left_mask_px),
                }
        else:
            grid_np, meta = build_warp_grid(
                base_shape_a,
                base_shape_b,
                self.left_mask_px,
                self.radar_a_latlon,
                self.radar_b_latlon,
                self.radar_a_range_km,
                self.radar_b_range_km,
                center_override_a=self.center_override_a,
                center_override_b=self.center_override_b,
            )
            self._warp_base = torch.from_numpy(grid_np)
            self._warp_shape = (grid_np.shape[0], grid_np.shape[1])
            self._warp_meta = meta
            if path:
                try:
                    ensure_dir(os.path.dirname(path))
                    np.save(path, grid_np)
                except Exception:
                    pass
            if not self._warp_warned:
                print("⚠️  warp grid rebuilt on the fly", flush=True)
                self._warp_warned = True

    def _get_warp_grid(self, target_shape: Tuple[int, int]) -> torch.Tensor:
        if self._warp_base is None:
            raise RuntimeError("Warp grid not initialized; call _ensure_warp_base first.")
        if tuple(target_shape) == self._warp_shape:
            return self._warp_base.unsqueeze(0)
        key = tuple(map(int, target_shape))
        if key not in self._warp_cache:
            base = self._warp_base.permute(2, 0, 1).unsqueeze(0)
            resized = F.interpolate(base, size=target_shape, mode="bilinear", align_corners=False)
            self._warp_cache[key] = resized.squeeze(0).permute(1, 2, 0).contiguous()
        return self._warp_cache[key].unsqueeze(0)

    def _pad_to(self, tensor: torch.Tensor, th: int, tw: int, mode: str = "reflect", value: float = 0.0):
        ph = max(0, th - tensor.shape[-2])
        pw = max(0, tw - tensor.shape[-1])
        if not (ph or pw):
            return tensor

        pad_mode = mode
        if pad_mode == "reflect" and (tensor.shape[-2] <= 1 or tensor.shape[-1] <= 1):
            pad_mode = "replicate"

        pad = (0, pw, 0, ph)
        work = tensor
        needs_threshold = False
        if tensor.dtype == torch.bool and pad_mode != "constant":
            work = tensor.float()
            needs_threshold = True
        work = work.unsqueeze(0).unsqueeze(0)
        if pad_mode == "constant":
            work = F.pad(work, pad, mode="constant", value=value)
        else:
            work = F.pad(work, pad, mode=pad_mode)
        work = work.squeeze(0).squeeze(0)
        if needs_threshold:
            work = work > 0.5
        return work

    def _choose_crop(self, chs: List[torch.Tensor], y_mask: torch.Tensor, crop: int):
        H, W = chs[0].shape[-2], chs[0].shape[-1]

        def _rand_coord(limit: int):
            if limit <= 0:
                return 0
            return int(torch.randint(0, limit + 1, (1,)).item())

        if torch.rand(()).item() < self.dcfg.pos_crop_prob:
            for _ in range(self.dcfg.pos_crop_tries):
                y0 = 0 if H <= crop else _rand_coord(H - crop)
                x0 = 0 if W <= crop else _rand_coord(W - crop)
                if y_mask[..., y0:y0+crop, x0:x0+crop].float().mean().item() > self.dcfg.pos_crop_thr:
                    return y0, x0
        y0 = 0 if H <= crop else _rand_coord(H - crop)
        x0 = 0 if W <= crop else _rand_coord(W - crop)
        return y0, x0

    def __getitem__(self, idx: int):
        ts = self.ts_list[idx]
        sample = self._sample_map[ts]

        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        present_flags: List[float] = []
        for label, source in (("A", self.A), ("B", self.B)):
            for t in sample[label]:
                path = source.ts_to_path[t]
                m_np, inten_np = get_weak_label_cached(
                    path,
                    self.dcfg.hsv,
                    self.dcfg.left_mask_px,
                    device=self.dcfg.weak_label_device,
                )
                m_t = torch.from_numpy(m_np).to(torch.bool)
                inten_t = torch.from_numpy(inten_np).to(torch.float32)
                pairs.append((m_t, inten_t))
                present_flags.append(1.0)

        crop_sz = max(1, int(self.dcfg.crop))
        shapes = [p[0].shape for p in pairs]
        H = max(int(s[0]) for s in shapes)
        W = max(int(s[1]) for s in shapes)

        def resize_to_hw(pair):
            mask, inten = pair
            if mask.shape != (H, W):
                mask = self._pad_to(mask, H, W, mode="replicate")
                mask = mask[:H, :W]
            if inten.shape != (H, W):
                inten = self._pad_to(inten, H, W, mode="replicate")
                inten = inten[:H, :W]
            return mask, inten

        frames_hw: List[Tuple[torch.Tensor, torch.Tensor]] = []
        avs_hw: List[torch.Tensor] = []
        for pair, pflag in zip(pairs, present_flags):
            mask_t, inten_t = resize_to_hw(pair)
            frames_hw.append((mask_t, inten_t))
            avs_hw.append(torch.full((H, W), float(pflag), dtype=torch.float32))

        a_base_shape = None
        b_base_shape = None
        for i, pair in enumerate(pairs):
            if pair is None:
                continue
            if i < 3 and a_base_shape is None:
                a_base_shape = pair[0].shape
            if i >= 3 and b_base_shape is None:
                b_base_shape = pair[0].shape

        atlas_A = torch.from_numpy(_fit_mask_to(H, W, self.A.atlas)).to(torch.bool)
        atlas_B = torch.from_numpy(_fit_mask_to(H, W, self.B.atlas)).to(torch.bool)

        def proc_triplet(triplet, av_triplet, atlas):
            out_inten = []
            out_av = []
            out_mask = []
            mask_t = None
            inten_t = None
            clear_mask = (~atlas)
            blocked_mask = ~clear_mask
            clear_float = clear_mask.to(torch.float32)
            for i, ((mask, inten), av) in enumerate(zip(triplet, av_triplet)):
                inten = inten.masked_fill(blocked_mask, 0.0)
                mask = mask.logical_and(clear_mask)
                av = av.to(torch.float32) * clear_float
                out_inten.append(inten)
                out_av.append(av)
                out_mask.append(mask)
                if i == 1:
                    mask_t, inten_t = mask, inten
            return out_inten, out_av, out_mask, mask_t, inten_t

        A_prev, A_t, A_next = frames_hw[0:3]
        B_prev, B_t, B_next = frames_hw[3:6]
        A_prev_av, A_t_av, A_next_av = avs_hw[0:3]
        B_prev_av, B_t_av, B_next_av = avs_hw[3:6]

        intenA, avA, maskA_list, maskA_t, intenA_t = proc_triplet([A_prev, A_t, A_next], [A_prev_av, A_t_av, A_next_av], atlas_A)
        intenB, avB, maskB_list, maskB_t, intenB_t = proc_triplet([B_prev, B_t, B_next], [B_prev_av, B_t_av, B_next_av], atlas_B)
        base_shape_a = tuple(int(x) for x in (a_base_shape if a_base_shape is not None else (H, W)))
        base_shape_b = tuple(int(x) for x in (b_base_shape if b_base_shape is not None else (H, W)))
        self._ensure_warp_base(base_shape_a, base_shape_b)
        warp_grid = self._get_warp_grid((H, W)).squeeze(0)

        if self.dcfg.fuse_mode == "mean":
            availA_t = avA[1]
            availB_t = avB[1]
            avail_sum = availA_t + availB_t
            denom = torch.where(avail_sum > 0, avail_sum, torch.ones_like(avail_sum))
            y_dbz = (intenA_t * availA_t + intenB_t * availB_t) / denom
        else:
            y_dbz = torch.maximum(intenA_t, intenB_t)
        y_mask = maskA_t.logical_or(maskB_t)

        key = (H, W)
        if key not in self._hc_cache:
            center = self._helper_center(H, W)
            self._hc_cache[key] = tuple(torch.from_numpy(arr).to(torch.float32) for arr in helper_channels(H, W, center=center))
        r_norm, cos_t, sin_t = (t.clone() for t in self._hc_cache[key])

        inpaint_mask = torch.zeros((H, W), dtype=torch.bool)
        if self.do_inpaint and self.inpaint_frac > 0.0:
            allowed = (~atlas_A).clone()
            if allowed.shape[1] > self.left_mask_px:
                allowed[:, :self.left_mask_px] = False
            else:
                allowed.zero_()
            avail_combined = (avA[1] > 0.0) | (avB[1] > 0.0)
            allowed = allowed.logical_and(avail_combined > 0.0)
            if allowed.any():
                noise = torch.rand(allowed.shape, dtype=torch.float32)
                holes = noise < self.inpaint_frac
                holes = holes.logical_and(allowed)
                if holes.any():
                    intenA[1] = intenA[1].masked_fill(holes, 0.0)
                    intenB[1] = intenB[1].masked_fill(holes, 0.0)
                    avA[1] = avA[1].masked_fill(holes, 0.0)
                    avB[1] = avB[1].masked_fill(holes, 0.0)
                inpaint_mask = holes

        chs = [
            intenA[0], intenA[1], intenA[2],
            intenB[0], intenB[1], intenB[2],
            avA[0],    avA[1],    avA[2],
            avB[0],    avB[1],    avB[2],
            r_norm, cos_t, sin_t,
        ]

        view = self._rand_view()
        chs = [self._apply_view(ch, view) for ch in chs]
        y_mask = self._apply_view(y_mask, view)
        y_dbz = self._apply_view(y_dbz, view)
        inpaint_mask = self._apply_view(inpaint_mask, view)
        warp_grid = self._apply_view(warp_grid.permute(2, 0, 1), view).permute(1, 2, 0)

        target_h = max(crop_sz, max(int(ch.shape[0]) for ch in chs))
        target_w = max(crop_sz, max(int(ch.shape[1]) for ch in chs))
        chs = [self._pad_to(ch, target_h, target_w, mode="reflect") for ch in chs]
        y_mask = self._pad_to(y_mask, target_h, target_w, mode="reflect")
        y_dbz = self._pad_to(y_dbz, target_h, target_w, mode="reflect")
        inpaint_mask = self._pad_to(inpaint_mask, target_h, target_w, mode="constant", value=0.0)
        warp_grid_ch = []
        for c in range(warp_grid.shape[-1]):
            warp_grid_ch.append(self._pad_to(warp_grid[..., c], target_h, target_w, mode="replicate"))

        crop = crop_sz
        y0, x0 = self._choose_crop(chs, y_mask, crop)
        chs = [ch[y0:y0+crop, x0:x0+crop] for ch in chs]
        y_mask = y_mask[y0:y0+crop, x0:x0+crop]
        y_dbz = y_dbz[y0:y0+crop, x0:x0+crop]
        inpaint_mask = inpaint_mask[y0:y0+crop, x0:x0+crop]
        warp_grid = torch.stack([wg[y0:y0+crop, x0:x0+crop] for wg in warp_grid_ch], dim=-1)

        x = torch.stack(chs, dim=0).to(torch.float32)
        y_mask = y_mask.to(torch.float32).unsqueeze(0)
        y_dbz = y_dbz.to(torch.float32).unsqueeze(0)
        inpaint_mask = inpaint_mask.to(torch.float32).unsqueeze(0)
        warp_grid = warp_grid.to(torch.float32)

        x = _maybe_pin_memory(x)
        y_mask = _maybe_pin_memory(y_mask)
        y_dbz = _maybe_pin_memory(y_dbz)
        inpaint_mask = _maybe_pin_memory(inpaint_mask)
        warp_grid = _maybe_pin_memory(warp_grid)

        return {
            "x": x.contiguous(),
            "y_mask": y_mask.contiguous(),
            "y_dbz": y_dbz.contiguous(),
            "inpaint_mask": inpaint_mask.contiguous(),
            "warp_grid": warp_grid.contiguous(),
        }

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
    if not _cuda_is_usable(): return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 8  # Ampere+/Ada (L4 OK)

# ===========================
# TQDM helpers
# ===========================
def _gb(bytes_): 
    return bytes_ / (1024**3)

def _cuda_mem():
    if not _cuda_is_usable():
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
    if _cuda_is_usable():
        device = torch.device("cuda")
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
    ordered = []
    for rid in (RADAR_A_ID.lower(), RADAR_B_ID.lower()):
        match = next((s for s in sources if rid in os.path.basename(s.root).lower()), None)
        if match is not None and match not in ordered:
            ordered.append(match)
    if len(ordered) == 2:
        sources = ordered
    elif rank == 0:
        print("⚠️  Could not infer radar ordering from folder names; using provided order.", flush=True)
    return sources

def _seed_worker(worker_id: int, base_seed: int, rank: int) -> None:
    seed = base_seed + worker_id + rank * 1000
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(sources, device, rank, world_size, num_workers, prefetch_factor,
                 warp_path: Optional[str] = None, do_inpaint: bool = False, inpaint_frac: float = 0.0):
    weak_dev = WEAK_LABEL_DEVICE
    if weak_dev != "cpu" and not _cuda_is_usable():
        weak_dev = "cpu"
    common_cfg = dict(
        weak_label_device=weak_dev,
        warp_cache_path=warp_path,
        radar_a_latlon=RADAR_A_LATLON,
        radar_b_latlon=RADAR_B_LATLON,
        radar_a_range_km=RADAR_A_RANGE_KM,
        radar_b_range_km=RADAR_B_RANGE_KM,
        center_override_a=RADAR_A_CENTER_OVERRIDE,
        center_override_b=RADAR_B_CENTER_OVERRIDE,
    )
    train_cfg = DataConfig(do_inpaint=do_inpaint, inpaint_frac=inpaint_frac, **common_cfg)
    val_cfg = DataConfig(do_inpaint=False, inpaint_frac=0.0, **common_cfg)
    train_ds = TwoRadarFusionDataset(sources, split="train", val_split=VAL_SPLIT, dcfg=train_cfg)
    val_ds   = TwoRadarFusionDataset(sources, split="val",   val_split=VAL_SPLIT, dcfg=val_cfg)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if ddp_is_dist() else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if ddp_is_dist() else None

    g = torch.Generator(); g.manual_seed(SEED + rank)
    pin = (device.type == "cuda")
    worker_init = partial(_seed_worker, base_seed=SEED, rank=rank)

    def mk(ds, bs, shuffle, sampler):
        kwargs = dict(dataset=ds, batch_size=bs, shuffle=shuffle, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin, worker_init_fn=worker_init,
                      generator=g, drop_last=False)
        if num_workers > 0:
            kwargs["persistent_workers"] = PERSISTENT_WORKERS
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(**kwargs)

    train_loader = mk(train_ds, BATCH_SIZE, shuffle=(train_sampler is None), sampler=train_sampler)
    val_loader   = mk(val_ds,   BATCH_SIZE, shuffle=False,                   sampler=val_sampler)

    if is_main(rank):
        print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}", flush=True)
        eff_bsz = BATCH_SIZE * (world_size if ddp_is_dist() else 1) * max(1, GRAD_ACCUM_STEPS)
        print(f"Effective batch size (per optimizer step): {eff_bsz}", flush=True)

    return train_cfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler

# ===========================
# SWA helpers
# ===========================
def _update_swa_bn_stats(train_ds, device, swa_model, batch_size, rank):
    if swa_model is None or len(train_ds) == 0:
        return
    bn_modules = [m for m in swa_model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bn_modules:
        return
    try:
        from torch.optim.swa_utils import update_bn
    except Exception:
        return

    was_training = swa_model.training

    if (not ddp_is_dist()) or rank == 0:
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        class _BNLoaderWrapper:
            def __init__(self, base_loader, device):
                self.base_loader = base_loader
                self.device = device

            def __iter__(self):
                non_blocking = (self.device.type == "cuda")
                for batch in self.base_loader:
                    x_unwarped = batch["x"].to(self.device, non_blocking=non_blocking)
                    warp_grid = batch["warp_grid"].to(self.device, non_blocking=non_blocking)
                    warp_grid = warp_grid.to(x_unwarped.dtype)

                    def _warp_on_device(tensor_stack: torch.Tensor) -> torch.Tensor:
                        return F.grid_sample(
                            tensor_stack,
                            warp_grid,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=False,
                        )

                    intenA = x_unwarped[:, 0:3, :, :]
                    intenB_unwarped = x_unwarped[:, 3:6, :, :]
                    avA = x_unwarped[:, 6:9, :, :]
                    avB_unwarped = x_unwarped[:, 9:12, :, :]
                    helpers = x_unwarped[:, 12:15, :, :]

                    intenB = _warp_on_device(intenB_unwarped)
                    avB = _warp_on_device(avB_unwarped)

                    x = torch.cat([intenA, intenB, avA, avB, helpers], dim=1).contiguous()
                    yield x.to(memory_format=torch.channels_last)

            def __len__(self):
                return len(self.base_loader)

        wrapper = _BNLoaderWrapper(loader, device)
        update_bn(wrapper, swa_model, device=device)

    if ddp_is_dist() and dist.is_initialized():
        for buf in swa_model.buffers():
            dist.broadcast(buf, src=0)
        dist.barrier()

    swa_model.train(was_training)

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
    inpaint_weight = max(0.0, hparams.get("inpaint_weight", 0.0))

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
        running_inpaint = 0.0
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
            x_unwarped = batch["x"].to(device, non_blocking=(device.type=="cuda"))
            warp_grid_gpu = batch["warp_grid"].to(device, non_blocking=(device.type=="cuda"))
            warp_grid_gpu = warp_grid_gpu.to(x_unwarped.dtype)
            y_mask_raw = batch["y_mask"].to(device, non_blocking=(device.type=="cuda"))
            y_mask = y_mask_raw if LABEL_SMOOTH_EPS <= 0 else y_mask_raw*(1.0 - LABEL_SMOOTH_EPS) + 0.5*LABEL_SMOOTH_EPS
            y_dbz = batch["y_dbz"].to(device, non_blocking=(device.type=="cuda"))
            inpaint_mask = batch.get("inpaint_mask")
            if inpaint_mask is not None:
                inpaint_mask = inpaint_mask.to(device, non_blocking=(device.type=="cuda"))

            def _warp_on_device(tensor_stack: torch.Tensor) -> torch.Tensor:
                return F.grid_sample(
                    tensor_stack,
                    warp_grid_gpu,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )

            intenA = x_unwarped[:, 0:3, :, :]
            intenB_unwarped = x_unwarped[:, 3:6, :, :]
            avA = x_unwarped[:, 6:9, :, :]
            avB_unwarped = x_unwarped[:, 9:12, :, :]
            helpers = x_unwarped[:, 12:15, :, :]

            intenB = _warp_on_device(intenB_unwarped)
            avB = _warp_on_device(avB_unwarped)

            x = torch.cat([intenA, intenB, avA, avB, helpers], dim=1).contiguous()
            x = x.to(memory_format=torch.channels_last)

            loss_inpaint = torch.zeros((), device=device)
            with (torch.amp.autocast("cuda", dtype=(torch.bfloat16 if (device.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16), enabled=(device.type=="cuda" and MIXED_PRECISION)) if device.type=="cuda" else contextlib.nullcontext()):
                logits, dbz_pred = model(x)
                loss_mask = loss_focal(logits, y_mask) + dice_w * dice_loss(logits, y_mask)
                gate = (y_mask_raw > 0.5).float()
                l1 = F.smooth_l1_loss(dbz_pred, y_dbz, reduction="none")
                pos = gate.sum()
                loss_dbz = (l1 * gate).sum() / (pos + 1e-6)
                if inpaint_mask is not None and inpaint_weight > 0.0:
                    holes = inpaint_mask.to(torch.float32)
                    hole_count = holes.sum()
                    if hole_count > 0:
                        l1_all = F.l1_loss(dbz_pred, y_dbz, reduction="none")
                        loss_inpaint = (l1_all * holes).sum() / hole_count
                    else:
                        loss_inpaint = torch.zeros((), device=device)
                total_loss = loss_mask + lambda_dbz * loss_dbz
                if inpaint_mask is not None and inpaint_weight > 0.0:
                    total_loss = total_loss + inpaint_weight * loss_inpaint
                loss = total_loss / grad_accum

            if device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            steps_processed = step

            if step % grad_accum == 0:
                optimizer_step()

            loss_display = float((loss * grad_accum).item())
            running = (0.98 * running + 0.02 * loss_display) if step > 1 else loss_display
            if inpaint_weight > 0.0 and inpaint_mask is not None:
                linp_val = float(loss_inpaint.detach().item())
                running_inpaint = (0.98 * running_inpaint + 0.02 * linp_val) if step > 1 else linp_val

            if is_main(rank):
                mem_alloc, mem_res, mem_peak = _cuda_mem()
                cur_lr = opt.param_groups[0]["lr"]
                global_bsz = x.size(0) * (world_size if ddp_is_dist() else 1)
                dt = max(1e-6, time.time() - iter_t0)
                ips = global_bsz / dt
                postfix = dict(loss=f"{running:.4f}", lr=f"{cur_lr:.2e}", ips=f"{ips:.1f}/s", mem=f"{mem_alloc:.2f}G|{mem_peak:.2f}G")
                if inpaint_weight > 0.0 and inpaint_mask is not None:
                    postfix["linp"] = f"{running_inpaint:.4f}"
                bar.set_postfix(**postfix)

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
                    x_unwarped = batch["x"].to(device, non_blocking=(device.type=="cuda"))
                    warp_grid_gpu = batch["warp_grid"].to(device, non_blocking=(device.type=="cuda"))
                    warp_grid_gpu = warp_grid_gpu.to(x_unwarped.dtype)
                    y_mask_raw = batch["y_mask"].to(device, non_blocking=(device.type=="cuda"))
                    y_mask = y_mask_raw if LABEL_SMOOTH_EPS <= 0 else y_mask_raw*(1.0 - LABEL_SMOOTH_EPS) + 0.5*LABEL_SMOOTH_EPS
                    y_dbz = batch["y_dbz"].to(device, non_blocking=(device.type=="cuda"))
                    inpaint_mask = batch.get("inpaint_mask")
                    if inpaint_mask is not None:
                        inpaint_mask = inpaint_mask.to(device, non_blocking=(device.type=="cuda"))

                    def _warp_on_device(tensor_stack: torch.Tensor) -> torch.Tensor:
                        return F.grid_sample(
                            tensor_stack,
                            warp_grid_gpu,
                            mode="bilinear",
                            padding_mode="zeros",
                            align_corners=False,
                        )

                    intenA = x_unwarped[:, 0:3, :, :]
                    intenB_unwarped = x_unwarped[:, 3:6, :, :]
                    avA = x_unwarped[:, 6:9, :, :]
                    avB_unwarped = x_unwarped[:, 9:12, :, :]
                    helpers = x_unwarped[:, 12:15, :, :]

                    intenB = _warp_on_device(intenB_unwarped)
                    avB = _warp_on_device(avB_unwarped)

                    x = torch.cat([intenA, intenB, avA, avB, helpers], dim=1).contiguous()
                    x = x.to(memory_format=torch.channels_last)
                    if device.type == "cuda":
                        with torch.amp.autocast("cuda", dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), enabled=MIXED_PRECISION):
                            logits, dbz_pred = _predict_tta(m, x, do_tta=do_tta)
                    else:
                        logits, dbz_pred = _predict_tta(m, x, do_tta=do_tta)
                    loss_mask = loss_focal(logits, y_mask) + dice_w * dice_loss(logits, y_mask)
                    gate = (y_mask_raw > 0.5).float()
                    l1 = F.smooth_l1_loss(dbz_pred, y_dbz, reduction="none")
                    pos = gate.sum()
                    loss_inpaint_val = torch.zeros((), device=device)
                    if inpaint_mask is not None and inpaint_weight > 0.0:
                        holes = inpaint_mask.to(torch.float32)
                        hole_count = holes.sum()
                        if hole_count > 0:
                            l1_all = F.l1_loss(dbz_pred, y_dbz, reduction="none")
                            loss_inpaint_val = (l1_all * holes).sum() / hole_count
                    total = loss_mask + lambda_dbz * (l1 * gate).sum() / (pos + 1e-6)
                    if inpaint_mask is not None and inpaint_weight > 0.0:
                        total = total + inpaint_weight * loss_inpaint_val
                    step_loss = float(total.item())
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
            bn_bsz = getattr(train_loader, "batch_size", BATCH_SIZE) or BATCH_SIZE
            _update_swa_bn_stats(train_ds, device, swa_model, bn_bsz, rank)
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

def _prep_single(path: str, left_mask_px: int, hsvp: HSVParams, do_rgb: bool, device: str,
                 predecoded: Optional[np.ndarray] = None):
    # Compute & write weak-label cache; optionally write RGB cache
    # NOTE: STRICT_CACHE_ONLY is ignored in cpu-prep; we always compute here.
    key = _weak_cache_key_base(path, hsvp, left_mask_px)
    store = get_weak_cache_store()
    w_npz, m_npy, i_npy = _weak_paths(key)
    have_disk = os.path.isfile(w_npz) or (os.path.isfile(m_npy) and os.path.isfile(i_npy))
    have_store = False
    if store:
        try:
            have_store = store.exists(key)
        except Exception:
            have_store = False
    # Only recompute if truly missing
    if not (have_store or have_disk):
        img = predecoded if predecoded is not None else imread_rgb(path)
        m, inten = _weak_label_core(img, hsvp, left_mask_px, device=device)
        if store:
            try: store.set(key, m, inten)
            except Exception: pass
        if WEAK_CACHE_WRITE_FILES:
            if WEAK_CACHE_COMPRESS:
                _save_weak_npz(w_npz, m, inten)
            else:
                np.save(m_npy, m.astype(np.uint8))
                np.save(i_npy, inten.astype(np.float32))
    elif store and have_disk and not have_store:
        try:
            if os.path.isfile(w_npz):
                m, inten = _load_weak_npz(w_npz)
            else:
                m = np.load(m_npy, mmap_mode="r").astype(bool)
                inten = np.load(i_npy, mmap_mode="r").astype(np.float32)
            store.set(key, m, inten)
        except Exception:
            pass
    if do_rgb and (RGB_CACHE_MODE != "off"):
        if predecoded is None:
            _ = imread_rgb(path)  # will write if missing
        else:
            _write_rgb_cache(path, predecoded)
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

    store = get_weak_cache_store()

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

    store_status = "on" if store else "off"
    if store and not WEAK_CACHE_WRITE_FILES:
        files_status = "off (shared-store only)"
    elif store:
        files_status = "on (shared+files)"
    else:
        files_status = "on" if WEAK_CACHE_WRITE_FILES else "off"

    print(
        f"RGB_CACHE_MODE={RGB_CACHE_MODE} | weak: compress={WEAK_CACHE_COMPRESS} "
        f"packbits={WEAK_PACK_MASK_BITS} inten_dtype={'u8' if WEAK_INTEN_DTYPE is np.uint8 else 'f16'} "
        f"shared-store={store_status} files={files_status}",
        flush=True
    )

    # ---- index & summary ----
    sources = build_sources(rank=0)
    print("Summary per source:", flush=True)
    shapes = []
    for src in sources:
        h, w = imread_rgb(src.items[0]["path"]).shape[:2]
        shapes.append((h, w))
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

    if getattr(args, "precompute_warp", False):
        if len(shapes) != 2:
            raise RuntimeError("Need two radar sources to precompute warp.")
        warp_out = args.warp_cache_path or warp_cache_path()
        ensure_dir(os.path.dirname(warp_out))
        grid_np, meta = build_warp_grid(
            shapes[0],
            shapes[1],
            LEFT_MASK_PX,
            RADAR_A_LATLON,
            RADAR_B_LATLON,
            RADAR_A_RANGE_KM,
            RADAR_B_RANGE_KM,
            center_override_a=RADAR_A_CENTER_OVERRIDE,
            center_override_b=RADAR_B_CENTER_OVERRIDE,
        )
        np.save(warp_out, grid_np)
        print(
            f"Warp B→A saved to {warp_out}\n"
            f"  H={shapes[0][0]} W={shapes[0][1]} LEFT_MASK_PX={LEFT_MASK_PX} "
            f"W_eff={meta['W_eff_A']:.1f} cx={meta['cxA']:.2f} cy={meta['cyA']:.2f} "
            f"mpp_A={meta['mpp_A']:.4f} mpp_B={meta['mpp_B']:.4f}",
            flush=True,
        )


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

        errors = []
        max_workers = args.prep_workers or max(1, (os.cpu_count() or 2) - 1)
        omp_per_worker = max(1, args.omp_per_worker)
        print(f"🧵 Spawning {max_workers} workers (OMP threads/worker={omp_per_worker}) on {len(work)} files", flush=True)

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
    device = torch.device("cuda" if _cuda_is_usable() else "cpu")
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
    warp_path = args.warp_cache_path or warp_cache_path()
    dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler = make_loaders(
        sources,
        device,
        rank,
        world_size,
        n_workers,
        prefetch,
        warp_path=warp_path,
        do_inpaint=bool(args.do_inpaint),
        inpaint_frac=max(0.0, float(args.inpaint_frac)),
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
        "inpaint_weight": max(0.0, float(args.inpaint_weight)) if args.do_inpaint else 0.0,
    }, device, rank, local_rank, world_size,
       dcfg, train_ds, val_ds, train_loader, val_loader, train_sampler, val_sampler,
       save_ckpt_path=CKPT_PATH, verbose=True)

    if is_main(rank):
        print("Training complete. Best val loss:", final_val, flush=True)
        print("Model saved at:", CKPT_PATH, flush=True)

    ddp_cleanup()
    gc.collect()
    if _cuda_is_usable():
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
    p.add_argument("--precompute-warp", action="store_true", help="Precompute and cache the Radar B→A warp grid")
    p.add_argument("--warp-cache-path", type=str, default=None, help="Override path for the cached warp grid (.npy)")
    p.add_argument("--radar-a-lat", type=float, default=RADAR_A_LATLON[0], help="Radar A latitude (degrees)")
    p.add_argument("--radar-a-lon", type=float, default=RADAR_A_LATLON[1], help="Radar A longitude (degrees)")
    p.add_argument("--radar-b-lat", type=float, default=RADAR_B_LATLON[0], help="Radar B latitude (degrees)")
    p.add_argument("--radar-b-lon", type=float, default=RADAR_B_LATLON[1], help="Radar B longitude (degrees)")
    p.add_argument("--radar-a-range-km", type=float, default=RADAR_A_RANGE_KM, help="Radar A range to far edge (km)")
    p.add_argument("--radar-b-range-km", type=float, default=RADAR_B_RANGE_KM, help="Radar B range to far edge (km)")
    p.add_argument("--radar-a-center", type=_parse_center_tuple, default=None, help="Override radar A center pixel as 'cx,cy'")
    p.add_argument("--radar-b-center", type=_parse_center_tuple, default=None, help="Override radar B center pixel as 'cx,cy'")
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
    p.add_argument("--weak-cache-write-files", action="store_true",
               help="Force legacy per-file weak-cache artifacts even when shared store is available")
    p.add_argument("--precompute-missing-only", action="store_true",
                   help="During cpu-prep, only build weak/RGB caches that are missing.")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort cpu-prep if any file fails to cache.")
    # Training regularization
    p.add_argument("--do-inpaint", action="store_true", help="Enable inpaint objective during training")
    p.add_argument("--inpaint-frac", type=float, default=0.0, help="Fraction of meteorology pixels masked for inpaint training")
    p.add_argument("--inpaint-weight", type=float, default=0.5, help="Weight applied to masked L1 inpaint loss")


    return p.parse_args()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    if getattr(args, "weak_cache_write_files", False):
        WEAK_CACHE_WRITE_FILES = True
        WEAK_CACHE_WRITE_FILES_FORCED_ON = True
    if args.radar_dirs:
        RADAR_DIRS[:] = [p.strip() for p in args.radar_dirs.split(",") if p.strip()]
    WORK_DIR = args.work_dir
    CACHE_ROOT = args.cache_root
    CACHE_DIR = os.path.join(CACHE_ROOT, "cache")
    RGB_CACHE_DIR = os.path.join(CACHE_ROOT, "img_npy_cache")
    RADAR_A_LATLON = (args.radar_a_lat, args.radar_a_lon)
    RADAR_B_LATLON = (args.radar_b_lat, args.radar_b_lon)
    RADAR_A_RANGE_KM = float(args.radar_a_range_km)
    RADAR_B_RANGE_KM = float(args.radar_b_range_km)
    RADAR_A_CENTER_OVERRIDE = args.radar_a_center
    RADAR_B_CENTER_OVERRIDE = args.radar_b_center
    for _d in [WORK_DIR, CACHE_DIR, RGB_CACHE_DIR]:
        os.makedirs(_d, exist_ok=True)

    if args.stage == "cpu-prep":
        # In cpu-prep we might want to enforce the atlas exists too.
        # If STRICT_ATLAS_ONLY passed here and cache missing, index_source will error.
        stage_cpu_prep(args)

    elif args.stage == "gpu-train":
        stage_gpu_train(args)
