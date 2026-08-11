# LLM Guide — SeedVR2 Batch Upscaler

> [Version française](LLM_GUIDE.fr.md) | **English**
>
> **Audience**: this document is written for an AI assistant (LLM) asked to
> understand, debug, or adapt this project to another machine. Read it fully
> before editing code. It contains the architecture, the dependency map, the
> complete history of problems already solved, and the non-regression rules.

---

## 1. Project snapshot

- **What**: a Windows desktop app that batch-upscales images with the
  ByteDance **SeedVR2-3B** one-step diffusion super-resolution model.
- **Official-code policy**: the app *wraps the official inference scripts*
  (repo `ByteDance-Seed/SeedVR`, cloned as `SeedVR/` next to `app.py`). It does
  **not** reimplement the pipeline. Only checkpoint *loading* is extended
  (safetensors FP16/FP8, GGUF), plus Windows compatibility shims.
- **Stack**: Python 3.11–3.13 (3.12 safest), torch (CUDA builds, cu128 index),
  Gradio UI, threaded batch engine. Comments and user-facing logs are in
  **French**; the Gradio UI is bilingual FR/EN (hot-switch via `i18n.py`).
- **Reference hardware** (validated): RTX 2000 Ada 16 GB, sm_89, bf16 OK,
  Windows 11, 32 GB RAM. Whole pipeline runs (GGUF Q8_0, ×4, 288 px tiles).
- **No CPU/demo mode by design**: without a CUDA GPU the app refuses to run
  and prints a repair diagnostic (see §6.6). Don't reintroduce silent fallbacks.

## 2. Repository map

```
app.py                     GUI + CLI entry point; sets PYTORCH_CUDA_ALLOC_CONF
                           (=expandable_segments:True) BEFORE any torch import.
check_install.py           Standalone diagnostics (Python, deps, GPU, repo, models).
install.bat / run_app.bat  Windows bootstrap (English console output).
requirements.txt           App-level deps — comments MUST start with '#' (a ';'
                           prefix is parsed as a pip environment marker -> #bug-1).
config/settings.json       User settings (git-ignored; auto-migrated by Settings.__post_init__).
models/                    User weights (git-ignored). SeedVR/ official repo clone (git-ignored).

seedvr2_upscaler/
├── constants.py           Paths, extensions, HF download presets, SEEDVR2_REPO env var name.
├── models.py              Typed dataclasses/enums: JobConfig, ModelInfo, ModelKind,
│                          OutputFormat/ConflictPolicy (from_label is FR/EN/canonical tolerant),
│                          Precision, TilingConfig, RunnerState, BatchStats, ImageOutcome.
├── settings.py            JSON persistence. __post_init__ migrates legacy French radio
│                          labels → canonical values, drops removed 'demo' backend,
│                          clamps language to fr/en. ADD NEW FIELDS HERE with defaults.
├── i18n.py                FR/EN string tables (same keys in both — unit-tested) + tr().
├── registry.py            scan_models/classify_model (kind+quant from filename),
│                          AuxAssets (VAE/pos/neg), ensure_aux_assets (auto-download
│                          from HF ByteDance-Seed/SeedVR2-3B), download_file (hf_hub_download).
├── naming.py              apply_suffix (off by default), resolve_output_path
│                          (overwrite/skip/rename `name (1).ext`, in-batch reservation set).
├── images.py              load_image (EXIF transpose, ICC, exif bytes, RGBA split),
│                          save_image (PNG/JPEG/WebP, quality, EXIF+ICC restore,
│                          piexif for PNG), split_alpha/merge_alpha.
├── tiling.py              compute_tiles (overlap grid), stitch (feather blending),
│                          needs_tiling (output > 4 Mpx auto).
├── gpu_check.py           detect_gpu() → GpuStatus (driver via nvidia-smi, torch build,
│                          cuda_available, bf16), cause_key() for the bilingual UI banner,
│                          French long-form help for logs/CLI. Never raises.
├── backend/
│   ├── base.py            UpscaleBackend ABC + BackendOptions; generic upscale():
│   │                      alpha split, ceil_div(·,16) rounding, tile loop with
│   │                      tile_cb(i,n) UI callback, stitch. MAX_TILE_OUT_SIDE = 1152
│   │                      caps OUTPUT tile side (training regime) — see #bug-9.
│   ├── official.py        The heart. Wraps projects.video_diffusion_sr.infer.
│   │                      VideoDiffusionInfer + configs_3b/main.yaml. See §3.
│   ├── gguf_loader.py     GGUF → state_dict via official `gguf` package; dequantizes
│   │                      Q*-types on CPU to target dtype, renames llama-style keys
│   │                      to the DiT checkpoint schema. ~6.5 GB RAM spike at load.
│   ├── flash_fallback.py  Injects fake `flash_attn` module (flash_attn_varlen_func)
│   │                      backed by torch SDPA. Uses ModuleSpec (importlib needs
│   │                      __spec__/__path__) — see #bug-7. Real package wins if present.
│   └── apex_fallback.py   Same trick for apex: FusedLayerNorm≈nn.LayerNorm,
│                          FusedRMSNorm≈nn.RMSNorm (state_dict compatible) — #bug-5.
├── worker.py              BatchRunner thread + EventBus (log/progress/preview/state/
│                          tile/finished). Pause/resume/cancel Events, ETA (5-image
│                          moving average), resume file `.seedvr2_resume.json` (config
│                          fingerprint), OOM→auto-tiling retry, parallel saves (2),
│                          append-mode log.txt, report.txt, disk-space preflight,
│                          Windows sleep inhibition (SetThreadExecutionState) and
│                          end-of-batch beep. Loads the model ONCE per batch.
└── gui_gradio.py          Gradio Blocks. 0.5 s Timer drains the EventBus. All widget
                           texts via i18n registry (hot language switch, no reload).
                           Radio widgets carry canonical values (label, value) tuples.
                           GPU banner built from cause_key() for the current language.
tests/selftest.py          21 unit tests, no GPU needed: naming, registry, tiling,
                           I/O, settings (+migration), i18n parity, batch engine with
                           an injected CPU fake backend (BatchRunner(backend_factory=…)).
docs/LLM_GUIDE*.md         This file.
```

## 3. How the official pipeline is wrapped (backend/official.py)

An image is treated as a **1-frame video** by the official scripts.

Load (`load()` → `_load_pipeline()`):
1. `gpu_check` diagnostics first; abort with repair instructions if no CUDA.
2. `torch.backends.cudnn.benchmark = True` (constant tile shapes → conv autotune).
3. Install flash/apex shims *proactively* (Windows), then `_with_autorepair`
   wraps every step: on ModuleNotFoundError, pip-installs the known package
   (`_PYPI_NAMES` map) into the *current* interpreter, once per module.
4. Load `configs_3b/main.yaml` via `common.config.load_config` with CWD
   temporarily switched to the repo root (config uses relative paths).
5. `_init_distributed_best_effort`: the repo's `@log_runtime` decorator calls
   `dist.barrier()` unconditionally, so a **real single-process gloo group** is
   created (Windows has no NCCL). Shims are the last resort.
6. dtype: bf16 if `torch.cuda.is_bf16_supported()` else fp16.
7. Assets: `ensure_aux_assets` (VAE + pos/neg text embeddings, HF auto-download).
8. DiT weights: official `.pth` → `runner.configure_dit_model(...)` verbatim;
   safetensors/GGUF → `_build_dit_from_state`: meta-device construction
   (mirrors the official branch), `load_state_dict(strict=False, assign=True)`,
   then `_materialize_buffers` (see #bug-8), `.to(device, dtype)`.
9. VAE: `configure_vae_model()` then `_tune_vae_memory_limit` overrides the
   official `conv_max_mem: 0.5 GiB` with `min(7.0, max(1.0, free*0.5)) GiB` —
   the 0.5 GiB default micro-slices decoding (17 s/tile measured) — #bug-10.
10. Diffusion: cfg 1.0, rescale 0, timesteps steps = 1, `configure_diffusion()`.
11. Optional `color_fix.wavelet_reconstruction` if the file exists in the repo.

Inference per region (`_upscale_region_impl`), mirroring
`projects/inference_seedvr2_3b.py` image branch:
`TF.to_tensor → [T=1,C,H,W] → NaResize(resolution=√(out_h·out_w), area,
downsample_only=False) → clamp01 → DivisibleCrop(16,16) → Normalize(0.5,0.5) →
Rearrange t c h w → c t h w` → `vae_encode([cond])` → `_generation_step`
(randn_like noises, per-tile `set_seed(seed, same_across_ranks=True)` with
fallback to manual_seed, cond_noise_scale=0.0, `get_condition(task='sr')`,
texts_pos/neg from cached device tensors, `runner.inference(...,
dit_offload=low_vram)` under `inference_mode`+bf16 autocast) → **output
branch exactly as official**: `video[:, None]` when `video.ndim == 3` (means
[C,H,W], time axis lost) — using `video[None]` silently transposes C/T → 1-channel
image ("not enough image data") — #bug-9. Then clamp [-1,1]→[0,255] uint8 RGB.

VRAM swap identical to the official script when `low_vram` (dit↔cpu, vae↔cuda);
auto-disabled when VRAM ≥ 14 GB, auto-enabled below 10 GB (`_tune_low_vram`).

## 4. Dependencies

App venv (`requirements.txt` + install.bat): gradio, Pillow, numpy,
safetensors, gguf, huggingface_hub, einops, omegaconf (+ optional piexif for
EXIF-in-PNG). Plus torch/torchvision from the **cu128 index** (`pip install
torch torchvision --index-url https://download.pytorch.org/whl/cu128`).

Official repo deps (curated, **not** its requirements.txt — it pins
torch==2.3.0 which would break modern Pythons and downgrade CUDA torch):
`diffusers>=0.29`, `transformers>=4.38`, `rotary-embedding-torch>=0.5`,
`opencv-python`, `mediapy`. `apex` and `flash-attn` are replaced by shims.

Python ≥ 3.14 → torch CUDA wheels may not exist yet (`No matching
distribution found for torch`) → recreate venv with `py -3.12`.

## 5. Windows compatibility layer (why each is safe)

| Missing piece | Shim | Why equivalent here |
|---|---|---|
| NCCL (no Windows build) | real **gloo** 1-proc group (`init_process_group`, tcp://127.0.0.1:29512) | only `dist.barrier()`/rank getters are exercised; world_size=1 |
| flash-attn | SDPA `flash_attn_varlen_func` (F.scaled_dot_product_attention per segment of `cu_seqlens`) | DiT v2 calls it **without** causal/window_size → SDPA is mathematically identical (validated <1e-5 fp32 vs naive attention, uniform+variable lengths) |
| apex `fusedln`/`fusedrms` (used by configs_3b norms) | `nn.LayerNorm` / `nn.RMSNorm` (eps forwarded, elementwise affine, same parameter names → state_dict compatible) | fused kernels are numerics-identical, only faster |

Both shims build real `types.ModuleType` with `ModuleSpec` + loader, and
`__path__=[]` for packages, otherwise `importlib`/`pkgutil` break (#bug-7).
If the real package exists, it is used untouched.

## 6. Bug history (symptom → root cause → fix) — READ BEFORE DEBUGGING

1. **gradio never installs; pip dies with `InvalidMarker`** — requirements.txt
   comment lines started with `;` → parsed as environment markers. Fix:
   `#`-prefixed comments only.
2. **`No matching distribution for torch` on cu124 index** — user's Python too
   new for that index's wheels. Fix: install.bat prefers `py -3.12`/`py -3.11`,
   chain cu128 → cu124 → PyPI.
3. **`ModuleNotFoundError: rotary_embedding_torch` at first load** — repo deps
   absent. Fix: curated unpinned install + runtime `_with_autorepair` (one
   auto pip attempt per module, then a clear error with the exact command).
4. **flash-attn cannot compile on Windows** → SDPA shim (§5).
5. **apex required by configs_3b (`fusedln`, `fusedrms`, `norm: fusedrms`,
   `txt_in_norm: fusedln`)** → nn.LayerNorm/RMSNorm shim (§5).
6. **`ValueError: Default process group has not been initialized`** — official
   `log_runtime` decorator barriers unconditionally; `init_torch` hardcodes
   NCCL. Fix: env RANK/LOCAL_RANK/WORLD_SIZE + real gloo group (§5).
7. **`ValueError: flash_attn.__spec__ is None`** — fake modules lacked
   importlib metadata. Fix: ModuleSpec-based module factory in both shims.
8. **Bare `AssertionError()` after weight load** — `rotary-embedding-torch`
   0.9.x registers extra non-persistent buffers (`cached_freqs` 8192×42,
   `freqs`), while the official `meta_non_persistent_buffer_init_fn` only
   materializes buffers named *dummy* then asserts. Fix: `_materialize_buffers`
   recomposes RoPE `freqs` exactly (`1/θ^(2i/d)`, freqs_for='lang', θ=10000;
   verified 0.00 vs a fresh module), zeros for cached/dummy placeholders.
9. **"not enough image data" after 7 min of correct inference + silent
   1-channel output** — my code used `video[None]` instead of official
   `video[:, None]`; ndim==3 ⇒ [C,H,W]. Transposed silently. Fix + regression
   tests on both ndim branches. **Lesson: copy the official indexation
   character for character.**
10. **4031 s/image** — three stacked causes: (a) `low_vram` swap active on a
    16 GB card → auto thresholds (§3); (b) tiles larger than the training
    regime (~1 Mpx) → attention cost quadratic → clamp OUTPUT tile side to 1152
    px (`MAX_TILE_OUT_SIDE`, logged); (c) VAE `conv_max_mem: 0.5 GiB` →
    micro-sliced decode → `_tune_vae_memory_limit`. After fixes: seconds/tile.
11. **"GPU not detected" after a reboot, and silent quality degradation** —
    the app silently fell back to a CPU demo backend. Fix: demo backend
    **removed entirely**; `gpu_check.detect_gpu()` diagnostics everywhere (bat
    line, GUI banner, load-time error, check_install.py) with the three classic
    causes (dead driver / CPU-only torch build / no reboot after driver update)
    and exact repair commands.
12. **Gradio drag&drop confusion** — dropped folders land in the OS temp dir
    → outputs appeared to "vanish" into `…\Temp\gradio\…\upscaled`. The UI
    label warns; advise explicit output paths. Also: the preview is a ≤768 px
    **thumbnail** — a user once mistook it for a "shrunken" output; the log
    prints true dimensions (`input×… → output×…`) for every image.

## 7. Configuration reference

- **Env vars**: `SEEDVR2_REPO` (repo path override),
  `PYTORCH_CUDA_ALLOC_CONF` (set by app.py, don't unset).
- **Settings file** `config/settings.json`: plain JSON; unknown keys ignored;
  canonical radio values (`x2/x4/x8/custom`, `keep/png/jpg/webp`,
  `overwrite/skip/rename`, `auto/bf16/fp16`, `auto/official`, `fr/en`).
  Legacy French labels auto-migrate (Settings.__post_init__).
- **Key constants** (`constants.py`/`base.py`): `DIVISOR=16`,
  `TILING_AUTO_THRESHOLD_MPX=4.0`, `MAX_TILE_OUT_SIDE=1152`,
  `_ETA_WINDOW=5`, gloo port 29512.

## 8. Recipes for adaptations

**Add a GUI option end-to-end (checklist — missing one place breaks silently):**
1. field in `Settings` (+ validation in `__post_init__` if enum-like);
2. add key to `_persist_settings` keys list **and** to `setting_inputs` and the
   `_on_start` unpack tuple — **same order everywhere**;
3. field in `JobConfig`; wire into the job in `_on_start`;
4. consume in `worker.py`/`backend`;
5. add i18n keys to BOTH tables of `i18n.py` (unit test enforces parity);
6. register the widget with `reg(comp, label=("s","my.key"), …)`;
7. add/extend a unit test; run `python tests/selftest.py`.

**Add a language**: add a third table in `i18n.py` with exactly the same keys;
`SUPPORTED_LANGUAGES` entry; nothing else.

**Add a weight format**: extend `registry.classify_model`; teach
`official._read_state_dict` to produce a DiT-schema state_dict; the rest is
generic.

**Never do**: reimplement pipeline math silently (wrap the official code and
cite it); change filename-preservation defaults; leak per-tile seeding
(seed+index per tile = reproducibility); swallow exceptions (logs carry full
tracebacks by design); add a CPU/demo fallback (project policy).

## 9. Validation commands (run after every change)

```bash
python -m compileall -q app.py check_install.py seedvr2_upscaler tests
python tests/selftest.py            # 21 tests, all must pass, no GPU needed
python -m seedvr2_upscaler.gpu_check --summary
python check_install.py
# Headless GUI smoke: build_app(tmp models dir) must construct ~70 components,
# and _on_language_change(session, "en") must return one update per registered
# widget (see tests history / docs).
```

## 10. Symptom → where to look

| Symptom | File |
|---|---|
| Wrong output names/suffixes | `naming.py`, `worker._process_one` |
| Tiles visible seams | `tiling.stitch` (overlap/feather) |
| Slow per-image time | log `étapes —` line; `official._tune_vae_memory_limit`, `base.MAX_TILE_OUT_SIDE`, `_tune_low_vram` |
| OOM | `BackendOptions.tiling`, low_vram, alloc env var |
| Import errors from SeedVR repo | `_with_autorepair`, `_PYPI_NAMES`, shims |
| Language not applied | widget not in `reg(...)` registry or missing i18n key (test catches) |
| Settings lost/old label | `settings.__post_init__` migrations |
| EXIF dropped | `images.save_image` (+ piexif installed?) |
| GPU vanished after reboot | `gpu_check` (run `python check_install.py`) |
