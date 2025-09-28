# Radar Cleaner Architecture Overview

## Top-Level Script Structure
- **Single entry point (`ddp.py`)** orchestrates both preprocessing and training stages. A CLI switch (`--stage`) selects between a CPU preparation pipeline and the GPU/DDP training path, while shared configuration defaults live near the top of the script for easy override via command-line flags.【F:ddp.py†L1-L200】【F:ddp.py†L2603-L2674】

## Distributed Runtime Utilities
- Helper functions (`ddp_env`, `ddp_setup`, `ddp_cleanup`, `is_main`) prepare distributed state, choose an appropriate backend, and ensure process groups are torn down cleanly. `weighted_reduce_sum_count` performs cross-rank aggregation of scalar metrics so validation stays consistent under DDP.【F:ddp.py†L205-L241】

## Data Acquisition and Caching
- Image discovery (`list_images`, `parse_timestamp`) and timestamp helpers convert filename tokens into sortable integers, enabling time-based neighbor lookups inside the dataset.【F:ddp.py†L243-L294】
- RGB assets optionally flow through an `.npy`/`.npz` caching layer driven by `_rgb_cache_path` and `imread_rgb`. Timestamp-derived stems keep filenames stable across machines and caches are written atomically to avoid corruption.【F:ddp.py†L284-L360】
- Weak label generation combines HSV thresholding, morphological cleanup, and configurable quantization/padding. `_weak_cache_key_base` and `WeakCacheStore` coordinate per-file artifacts with optional shared-store backends so recomputation can be skipped when caches already exist.【F:ddp.py†L296-L757】【F:ddp.py†L598-L757】【F:ddp.py†L1805-L1959】
- Atlas building (`build_atlas_mask`, `_fit_mask_to`) summarizes static artifacts (e.g., ground clutter) so downstream samples can zero out unusable pixels.【F:ddp.py†L509-L636】

## Geometry and Radar Alignment
- `compute_center_px`, `compute_mpp`, and `build_warp_grid` derive uncropped radar centers, meters-per-pixel scaling, and the dense AEQD-based sampling grid used to reproject Radar B into Radar A’s pixel layout while accounting for the masked left strip.【F:ddp.py†L1070-L1135】
- The dataset lazily loads (or regenerates) the cached warp grid, resizes it on demand, and applies it to Radar B’s intensity, availability, and mask channels prior to fusion so both sources are co-registered during training.【F:ddp.py†L1320-L1588】
- During `cpu-prep`, enabling `--precompute-warp` rebuilds the dense warp against uncropped frame dimensions and logs the derived center, effective width, and meters-per-pixel metrics for traceability.【F:ddp.py†L2451-L2474】

## Dataset Pipeline
- `SourceData` encapsulates per-radar metadata, including sorted timestamp indices, atlas masks, and maps from timestamps to file paths.【F:ddp.py†L1170-L1239】
- `TwoRadarFusionDataset` gathers temporal triplets for both radars, retains the per-radar central masks and availability slices after augmentation, aligns Radar B via the cached warp, augments with helper channels, and prepares optional inpaint masks before padding/cropping for batching.【F:ddp.py†L1230-L1668】
- Loader construction (`make_loaders`) builds deterministic train/validation splits, sets up `DistributedSampler` instances when running under DDP, and forwards user-tunable DataLoader parameters such as worker count and prefetch depth.【F:ddp.py†L1869-L1947】

## Model Definition
- `TinyUNet` implements a lightweight encoder–decoder with GroupNorm. Building blocks (`ConvGNReLU`, `Down`, `Up`) provide reusable convolutional stages, while the network produces two heads: a segmentation mask and a reflectivity regression map.【F:ddp.py†L1670-L1775】

## Losses and Regularization
- `FocalBCE` and `dice_loss` target the mask output, blending focal cross-entropy with Dice stabilization. Smooth L1 regression handles the dBZ head, gated by positive mask pixels. Optional EMA and SWA modules smooth training dynamics and evaluation checkpoints.【F:ddp.py†L1032-L1117】【F:ddp.py†L2103-L2260】

## Inpaint Objective
- The dataset can corrupt randomly sampled meteorology pixels (excluding the label strip and atlas-masked regions), zeroing the affected channels while emitting an `inpaint_mask` tensor that tracks hole locations for reconstruction supervision.【F:ddp.py†L1604-L1667】
- Training and validation loops incorporate a masked L1 reconstruction loss weighted by `--inpaint-weight` whenever holes are present, and progress logging reports the running reconstruction loss when enabled.【F:ddp.py†L2144-L2244】
- CLI flags (`--do-inpaint`, `--inpaint-frac`, `--inpaint-weight`) control whether the objective is active and its sampling density/weighting.【F:ddp.py†L2647-L2649】

## CPU Preparation Stage
- `stage_cpu_prep` collects radar sources, configures cache formats (e.g., enabling fast I/O via `--fast-io`), optionally precomputes weak labels/RGB caches in parallel, and can prebuild the dense warp grid. Coverage reporting still reflects both store hits and disk artifacts, and strict reuse modes continue to forbid recomputation.【F:ddp.py†L2384-L2585】

## GPU / DDP Training Stage
- `stage_gpu_train` initializes distributed state, validates cache availability when strict modes are enabled, assembles loaders, and delegates to `run_training` with the active hyperparameter defaults. Cleanup ensures process groups and GPU memory are released at the end of the run.【F:ddp.py†L1949-L2098】

## Command-Line Interface
- `parse_args` enumerates user-facing toggles for both stages, including geometry overrides, cache policies, worker configuration, strictness flags, inpaint controls, and paths. `main` applies overrides (e.g., updating global directories), selects the requested stage, and invokes the corresponding driver.【F:ddp.py†L2603-L2674】

This architecture keeps preprocessing and training in a single script while modularizing concerns—caching, dataset assembly, modeling, optimization, geometry alignment, and distributed execution—so operators can switch between space-saving and performance-oriented workflows without modifying source code.【F:ddp.py†L1-L200】【F:ddp.py†L1320-L1668】
