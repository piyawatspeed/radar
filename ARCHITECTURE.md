# Radar Cleaner Architecture Overview

## Top-Level Script Structure
- **Single entry point (`ddp.py`)** orchestrates both preprocessing and training stages. The script exposes a CLI flag (`--stage`) that selects between a CPU-only preparation pipeline and a GPU/DistributedDataParallel (DDP) training run.【F:ddp.py†L1486-L1649】【F:ddp.py†L1657-L1718】
- Global constants near the top of the file capture operational defaults for dataset paths, caching behavior, model hyperparameters, scheduler settings, and distributed knobs. CLI arguments can override many of these values at runtime.【F:ddp.py†L27-L221】【F:ddp.py†L1269-L1356】

## Distributed Runtime Utilities
- Helper functions (`ddp_env`, `ddp_setup`, `ddp_cleanup`, `is_main`) prepare PyTorch distributed state, choose an NCCL backend when CUDA is available, and gracefully tear down process groups. The utilities allow both single-process and multi-process runs to share the same control flow.【F:ddp.py†L223-L261】
- `weighted_reduce_sum_count` performs an all-reduce over scalar loss and count accumulators so validation metrics can be averaged consistently across workers.【F:ddp.py†L263-L268】

## Data Acquisition and Caching
- Image discovery (`list_images`, `parse_timestamp`) and timestamp helpers convert between filename timestamps and `datetime` objects, enabling time-based neighbor lookups inside the dataset.【F:ddp.py†L210-L244】
- RGB assets optionally flow through an `.npy`/`.npz` caching layer driven by `_rgb_cache_path` and `imread_rgb`. Timestamp-derived stems keep filenames stable across machines, caches are written atomically to avoid corruption, and operators can choose compressed (space-saving) or raw (fast I/O) modes.【F:ddp.py†L227-L363】
- Weak label generation combines HSV thresholding, morphological cleanup, and configurable quantization/padding. `_weak_label_core` and `_morph_open_close_torch` accept a user-selected device (CPU or CUDA with AMP), while `_weak_cache_key_base`, `_weak_cache_exists_only`, and `get_weak_label_cached` coordinate cache reuse and on-demand recomputes without violating strict policies.【F:ddp.py†L78-L573】
- Atlas building (`build_atlas_mask`, `_fit_mask_to`) summarizes static artifacts (e.g., radar occlusion regions) so downstream samples can zero out unusable pixels.【F:ddp.py†L575-L705】

## Dataset Pipeline
- `SourceData` encapsulates per-radar metadata, including sorted timestamp indices, atlas masks, and maps from timestamps to file paths.【F:ddp.py†L638-L734】
- `RadarDataset` synthesizes temporal triplets around a reference frame for two radar sources, fusing weak-label masks and intensities, handling cache lookups, performing random view augmentations, and returning tensors ready for training. It dynamically pads and crops samples to the configured crop size, ensuring consistent shapes for batching while honoring the configured weak-label device preference.【F:ddp.py†L722-L944】
- Loader construction (`make_loaders`) builds deterministic train/validation splits, sets up `DistributedSampler` instances when running under DDP, and forwards user-tunable DataLoader parameters such as worker count and prefetch depth. When GPU preprocessing is enabled, it forces worker counts to zero so CUDA stays in the main process.【F:ddp.py†L1118-L1159】

## Model Definition
- `TinyUNet` implements a lightweight encoder–decoder with GroupNorm. Building blocks (`ConvGNReLU`, `Down`, `Up`) provide reusable convolutional stages, while the network produces two heads: a segmentation mask and a reflectivity regression map.【F:ddp.py†L946-L1030】

## Losses and Regularization
- `FocalBCE` and `dice_loss` target the mask output, blending focal cross-entropy with Dice stabilization. Smooth L1 regression handles the dBZ head, gated by positive mask pixels. Optional EMA and SWA modules smooth training dynamics and evaluation checkpoints.【F:ddp.py†L1032-L1117】【F:ddp.py†L1172-L1245】

## Training Loop
- `run_training` wires together optimizer creation, optional learning-rate finder sweeps, scheduler selection (OneCycle, Cosine with optional SWA, or Plateau), mixed-precision scaling, gradient clipping, EMA/SWA updates, validation passes, and checkpointing. Gradient accumulation is handled via a helper that ensures trailing micro-batches still trigger optimizer steps.【F:ddp.py†L1119-L1265】【F:ddp.py†L1185-L1239】
- Validation leverages `_predict_tta` for optional test-time augmentation and aggregates metrics across distributed ranks with `weighted_reduce_sum_count`. Early stopping monitors validation loss and persists the best-performing state dict.【F:ddp.py†L1209-L1265】【F:ddp.py†L1265-L1337】

## CPU Preparation Stage
- `stage_cpu_prep` collects radar sources, configures cache formats (e.g., enabling fast I/O via `--fast-io`), and optionally precomputes weak labels and RGB caches in parallel with a `ProcessPoolExecutor`. `_prep_single` routes work through the requested weak-label device so cache builds can leverage CUDA when available, while progress accounting and strict reuse reporting remain unchanged.【F:ddp.py†L1456-L1585】

## GPU / DDP Training Stage
- `stage_gpu_train` initializes distributed state, validates cache availability when strict modes are enabled, assembles loaders, and finally delegates to `run_training` with the currently active hyperparameter defaults. Cleanup ensures process groups and GPU memory are released at the end of the run.【F:ddp.py†L1527-L1605】

## Command-Line Interface
- `parse_args` enumerates user-facing toggles for both stages, including cache policies, worker configuration, strictness flags, and paths. A dedicated `--prep-on-gpu` flag toggles CUDA-backed weak-label preprocessing, and `main` applies overrides before dispatching to the requested stage.【F:ddp.py†L1657-L1718】

This architecture keeps preprocessing and training in a single script while modularizing concerns—caching, dataset assembly, modeling, optimization, and distributed execution—so operators can switch between space-saving and performance-oriented workflows without modifying source code.
