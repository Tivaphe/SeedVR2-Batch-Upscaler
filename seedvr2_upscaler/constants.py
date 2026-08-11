"""Constantes globales de l'application.

Centralisé ici pour éviter toute dispersion des valeurs magiques dans le code.
Compatible Windows / Linux / macOS (chemins via pathlib).
"""
from __future__ import annotations

from pathlib import Path

# --- Dossiers de l'application ------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR: Path = ROOT_DIR / "models"
CONFIG_DIR: Path = ROOT_DIR / "config"
SETTINGS_PATH: Path = CONFIG_DIR / "settings.json"

# --- Fichiers générés dans le dossier de sortie -------------------------------
LOG_FILENAME = "log.txt"               # journal complet (erreurs incluses)
REPORT_FILENAME = "report.txt"         # rapport final de traitement
RESUME_FILENAME = ".seedvr2_resume.json"  # état de reprise d'un lot interrompu

# --- Formats supportés ---------------------------------------------------------
# Entrées acceptées par le pipeline.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)
# Poids de modèles que l'on sait identifier dans models/.
MODEL_EXTENSIONS: frozenset[str] = frozenset({".safetensors", ".pth", ".pt", ".gguf"})

# --- Repo officiel SeedVR2 ------------------------------------------------------
# Variable d'environnement prioritaire, sinon recherche dans ces dossiers.
SEEDVR2_REPO_ENV = "SEEDVR2_REPO"
SEEDVR2_REPO_CANDIDATES: tuple[str, ...] = ("SeedVR", "seedvr2_repo", "vendor/SeedVR")

# --- Assets auxiliaires requis par le pipeline officiel --------------------------
# Le script officiel (projects/inference_seedvr2_3b.py) utilise :
#   - un checkpoint VAE (ema_vae.pth)
#   - des embeddings texte précalculés (pos_emb.pt / neg_emb.pt)
# Fournis par le repo HF ByteDance-Seed/SeedVR2-3B (licence Apache 2.0).
OFFICIAL_HF_REPO = "ByteDance-Seed/SeedVR2-3B"
VAE_CANDIDATE_PATTERNS: tuple[str, ...] = ("*vae*.pth", "*vae*.safetensors")
POS_EMB_CANDIDATES: tuple[str, ...] = ("pos_emb.pt",)
NEG_EMB_CANDIDATES: tuple[str, ...] = ("neg_emb.pt",)

# --- Presets de téléchargement (bouton "Télécharger" de l'UI) -------------------
# (label affiché, repo Hugging Face, nom de fichier dans le repo)
PRESET_DOWNLOADS: tuple[tuple[str, str, str], ...] = (
    ("SeedVR2-3B officiel (.pth, ~6,4 Go)", OFFICIAL_HF_REPO, "seedvr2_ema_3b.pth"),
    ("SeedVR2-3B FP8 e4m3 (~3,3 Go)", "AInVFX/SeedVR2_comfyui", "seedvr2_ema_3b_fp8_e4m3fn.safetensors"),
    ("SeedVR2-3B FP16 (~6,4 Go)", "AInVFX/SeedVR2_comfyui", "seedvr2_ema_3b_fp16.safetensors"),
    ("SeedVR2-3B GGUF Q3_K_M (~1,5 Go)", "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q3_K_M.gguf"),
    ("SeedVR2-3B GGUF Q4_K_M (~1,9 Go)", "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q4_K_M.gguf"),
    ("SeedVR2-3B GGUF Q5_K_M (~2,3 Go)", "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q5_K_M.gguf"),
    ("SeedVR2-3B GGUF Q6_K (~2,7 Go)", "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q6_K.gguf"),
    ("SeedVR2-3B GGUF Q8_0 (~3,4 Go)", "cmeka/SeedVR2-GGUF", "seedvr2_ema_3b-Q8_0.gguf"),
)

# --- Réglages numériques par défaut --------------------------------------------
DEFAULT_SEED = 666                      # seed par défaut du script officiel
MAX_SCALE = 16.0                        # borne haute du facteur personnalisé
DIVISOR = 16                            # DivisibleCrop((16, 16)) du script officiel
DEFAULT_TILE_SIZE = 512                 # taille des tuiles côté entrée (px)
DEFAULT_TILE_OVERLAP = 64               # recouvrement des tuiles côté entrée (px)
TILING_AUTO_THRESHOLD_MPX = 4.0         # au-delà (en sortie), tuiles auto
