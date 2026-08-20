# SeedVR2 Batch Upscaler

> [🇫🇷 Français](README.fr.md) | **🇬🇧 English**

A **Windows / Python 3.11+** application for **batch** image upscaling with **SeedVR2-3B**
(one diffusion step — very fast): a modern **Gradio** UI **in French and English**
(hot-switchable at the top of the interface), a threaded batch engine
(Pause / Resume / Cancel), and a backend that **wraps the official inference
scripts** from the ByteDance repository instead of reimplementing the pipeline.

```
input/                    output/
├── photo001.png          ├── photo001.png      ← file names kept exactly as-is
├── vacation.jpg    ──▶   ├── vacation.jpg
└── test_image.webp       └── test_image.webp
```

## Showcase

Real comparisons (produced by this application) — **left: original,
right: SeedVR2 ×4**, displayed at identical size:

| Floating city | Backlit leaf |
|---|---|
| ![Comparison — city](docs/showcase-city.png) | ![Comparison — leaf](docs/showcase-leaf.png) |

## Features

| Area | Detail |
|---|---|
| Models | Official `.pth`, safetensors **FP16/BF16**, **FP8 e4m3fn**, **GGUF** Q3_K_M · Q4_K_M · Q5_K_M · Q6_K · Q8_0 |
| Detection | Automatic scan of `models/` at startup + dropdown + refresh button + built-in Hugging Face preset downloads |
| Names | Input file name kept **exactly** (no suffix by default); the "Same as input" option also keeps **the extension**; suffix only when explicitly enabled; conflicts → Overwrite / Skip / Auto-rename `photo (1).png` |
| Batch | Whole input folder, PNG/JPG/JPEG/WEBP/BMP/TIFF, per-image failures don't stop the batch, `log.txt` (full journal) + `report.txt` (totals, averages, errors) |
| Gradio UI | **Hot FR/EN language switch** (no reload), folder pickers, **drag & drop**, ×2/×4/×8/custom factor, output format + quality, progress bar, **ETA**, live log, **before/after preview** (comparison slider), **Start / Pause / Resume / Cancel** buttons, GPU status banner |
| VRAM | Model loaded **once**, `torch.inference_mode()`, auto BF16 (FP16 otherwise), CPU↔GPU DiT/VAE swap (like the official script), cache cleared between images, automatic **tiling** + automatic tiling fallback on OOM, `expandable_segments` allocator (anti-fragmentation), `cudnn.benchmark` (conv autotune on constant shapes), text embeddings kept on GPU, VAE memory budget calibrated to free VRAM |
| Extras | Resume of an interrupted batch (`.seedvr2_resume.json`), automatic settings persistence, parallel disk writes, EXIF + ICC profile preserved (native JPEG/WebP, PNG via piexif), alpha channel preserved, full CLI |
| Comfort | **Windows sleep blocked during a batch** (restored afterwards), completion beep, disk space preflight check, "tile i/N" sub-progress, 📂 button to open the output folder, `log.txt` in append mode (history kept) |

## 1. Installation (Windows)

```bat
install.bat        :: creates .venv (preferring Python 3.12/3.11), installs CUDA torch
                   :: (tries the cu130, cu128, then PyPI indexes), the dependencies,
                   :: clones the official repo and checks the installation
run_app.bat        :: starts the graphical interface (with pre-flight checks)
check_install.py   :: full diagnostics (versions, CUDA, models, repository)
```

> **Python 3.14**: CUDA wheels exist on the **cu130** index (torch 2.9-2.13,
> driver ≥ 580). If you ever see `No matching distribution found for torch`,
> the index in use has no wheel for your Python — recreate the environment
> with Python 3.12: `py -3.12 -m venv .venv` (install.bat already prefers it
> automatically).
> **Older NVIDIA driver (< 580)**: use the cu128 index instead (torch ≤ 2.11,
> CUDA 12.8) — replace `cu130` with `cu128` in the commands.
>
> **apex**: the official repo requires `apex`. On Windows its compilation fails —
> that is expected and **non-blocking**: the application injects a numerically
> equivalent `nn.LayerNorm`/`nn.RMSNorm` fallback (see FAQ).

Manual equivalent:

```powershell
py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt mediapy
git clone https://github.com/ByteDance-Seed/SeedVR.git SeedVR
```

The official repo is auto-located in `SeedVR/`, `seedvr2_repo/`,
`vendor/SeedVR/`, or via the `SEEDVR2_REPO` environment variable.

## 2. Models (`models/` folder)

Drop any of these weights into `models/` — they will be detected and classified
automatically (or use the interface's **Download** button):

| File | Type | Indicative VRAM |
|---|---|---|
| `seedvr2_ema_3b.pth` | Official | ~16-24 GB |
| `seedvr2_ema_3b_fp16.safetensors` | FP16 | ~12-16 GB |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | FP8 | ~8-12 GB |
| `seedvr2_ema_3b-Q4_K_M.gguf` … `-Q8_0.gguf` | GGUF | ~6-8 GB (+6.5 GB RAM for dequantization) |

The official pipeline's VAE (`ema_vae.pth`) and text embeddings
(`pos_emb.pt`, `neg_emb.pt`) are **downloaded automatically** from
`ByteDance-Seed/SeedVR2-3B` on first launch if missing from `models/`.

## 3. Usage

### Graphical interface

```powershell
python app.py           # or run_app.bat ; --share to expose on the network
```

1. Pick the **language** (selector top right), the input/output folders
   (or **drag a folder** into the dedicated zone);
2. Select a model from the list;
3. Set the factor (×2/×4/×8/custom), output format and quality;
4. **Start** — watch progress, ETA and log; **Pause/Resume/Cancel** at any
   time; the before/after preview updates on the latest image.

All settings are **saved automatically** (`config/settings.json`).
If a run is interrupted, just relaunch: finished images are skipped
("Resume an interrupted batch" option, enabled by default).

### Command line

```powershell
python app.py --cli --input input --output output --model seedvr2_ema_3b-Q4_K_M.gguf `
    --scale 4 --format keep --quality 95 --tiling
```

> ⚠️ An **NVIDIA GPU + CUDA** is mandatory (there is no CPU mode: without the
> real SeedVR2, results would be pointless anyway). If the GPU is no longer
> detected, the interface shows a **full diagnostic** (driver, torch build,
> likely cause, repair command) — see also `python check_install.py` and the FAQ.

## 4. Architecture

```
seedvr2_upscaler/
├── constants.py        # paths, extensions, HF presets, official repo
├── models.py           # typed dataclasses/enums (JobConfig, ModelInfo…)
├── settings.py         # JSON persistence + migration of legacy settings
├── i18n.py             # FR/EN interface strings (hot i18n)
├── registry.py         # model detection/classification + HF downloads
├── naming.py           # name preservation + conflict policy
├── images.py           # I/O: EXIF, ICC, alpha, PNG/JPEG/WebP, BMP/TIFF inputs
├── tiling.py           # overlapping tiles + soft (feather) merging
├── gpu_check.py        # NVIDIA driver / torch build / CUDA diagnostics (UI + CLI)
├── backend/
│   ├── base.py         # abstract contract + generic alpha + tiling
│   ├── official.py     # ✦ OFFICIAL SeedVR2 pipeline (configs_3b, transforms,
│   │                   #   VideoDiffusionInfer, 1-step DiT) + extended loading
│   │                   #   of FP16/FP8 safetensors
│   ├── gguf_loader.py  # GGUF dequantization → state_dict (official gguf package)
│   ├── flash_fallback.py # SDPA fallback for flash_attn (Windows, equivalent here)
│   └── apex_fallback.py  # LayerNorm/RMSNorm fallback for apex (Windows, equivalent)
├── worker.py           # threaded batch engine: pause/resume/cancel, ETA,
│                       # resume, OOM→tiling fallback, parallel I/O, log+report
└── gui_gradio.py       # bilingual Gradio UI (0.5 s Timer + i18n registry)
app.py                  # GUI + CLI entry point
tests/selftest.py       # 21 unit tests (naming, registry, tiles, i18n, batch…)
docs/LLM_GUIDE.md       # full guide written for LLMs (project adaptation)
docs/make_showcase.py   # README comparison image built from your real batches
```

**Official integration approach.** The backend imports
`projects.video_diffusion_sr.infer.VideoDiffusionInfer` and the
`configs_3b/main.yaml` config, then reproduces the "image" branch (one frame)
of the `projects/inference_seedvr2_3b.py` script: `NaResize → DivisibleCrop(16) →
Normalize → vae_encode → generation_step (noise/condition/1 step, cfg=1) →
vae_decode`, with the same `dit↔vae` VRAM swap and the `empty_cache()` between
files. Only **weight loading** is extended (FP16/FP8 safetensors, dequantized
GGUF), because the official script only reads `.pth` files.

## 5. FAQ / troubleshooting

- **"GPU not detected" after a reboot** → run `python check_install.py`: it
  distinguishes the three classic causes and gives the exact repair — ① silent
  NVIDIA driver (`nvidia-smi` fails, often after a Windows update → reinstall
  the driver and **reboot**); ② torch replaced by a **CPU-only** build (pip
  without the CUDA index — note: the cu128 index stopped publishing CUDA
  builds at torch 2.11, so `pip install torch` from PyPI now lands on a
  CPU-only 2.12/2.13 →
  `.venv\Scripts\python.exe -m pip install torch torchvision --index-url
  https://download.pytorch.org/whl/cu130`); ③ fresh driver without reboot or a
  misconfigured `CUDA_VISIBLE_DEVICES` variable. The interface shows the same
  diagnostic in its top banner and in the log when a batch starts.
- **FlashAttention / apex on Windows** → cannot build from source; the app
  automatically injects **equivalent native fallbacks** (PyTorch SDPA for
  `flash_attn_varlen_func` — the repo never uses it causally or windowed;
  `nn.LayerNorm`/`nn.RMSNorm` for the `fusedln`/`fusedrms` of `configs_3b`).
  If the real package is present, it is used as-is.
- **Which torch version do the SeedVR2 models need?** → None in particular:
  checkpoints (`.pth`/`.safetensors`/`.gguf`) are plain tensors, independent
  of the torch build. The official repo pins `torch==2.3.0` for its own
  training environment — incompatible with modern Pythons — so this app
  deliberately ignores that pin and runs on current torch (validated
  2.10–2.13, CUDA cu128/cu130 builds; the cu128 index stops at 2.11).
  Only a CPU-only build (`+cpu`) cannot run the pipeline.
- **OOM despite low VRAM mode** → keep "Always use tiles" enabled and lower
  the tile size (256). The engine also switches to tiling automatically after
  an OOM.
- **Output dimensions** → multiples of 16 (constraint of the official
  VAE/DivisibleCrop); the mismatch targets the requested factor (e.g. 60 px ×2
  → 128 px).
- **GGUF slow on first load** → CPU dequantization (once per batch).
- **How to go even faster?** Tile batching (2–4 per GPU pass) is under study:
  validating it requires checking in the official repo how attention handles
  multi-frame packing — it will only ship as an option disabled by default,
  after numerical tests, because cross-tile content leakage would silently
  degrade quality.
- **Log in French?** The diagnostic log messages (`log.txt`) are in French;
  the interface itself is fully bilingual FR/EN.
- The batch log and all errors live in `<output>/log.txt`; the final summary
  in `<output>/report.txt`.

## Adapting the project with an LLM

A complete guide written for AI assistants (architecture, dependencies,
already-solved Windows pitfalls, extension recipes, non-regression rules) is
provided in [`docs/LLM_GUIDE.md`](docs/LLM_GUIDE.md) —
[version française](docs/LLM_GUIDE.fr.md). Hand that file to your LLM together
with your adaptation request.

## License

This application: MIT (see `LICENSE`). SeedVR2 (code + weights): Apache 2.0 — © ByteDance.
