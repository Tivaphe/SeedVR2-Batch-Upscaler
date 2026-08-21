"""Chaînes de l'interface graphique, en français et en anglais.

Périmètre : tout ce qui est **affiché dans les widgets Gradio** (libellés,
boutons, statuts, compteurs, messages d'erreur de démarrage, bandeau GPU).
Le journal technique émis par le worker/backend reste en français : ce sont
des messages de diagnostic (log.txt), pas des éléments d'interface persistants.

Ajouter une langue = ajouter un dictionnaire complet ci-dessous (les deux
tables doivent avoir exactement les mêmes clés — vérifié par tests/selftest).
"""
from __future__ import annotations

SUPPORTED_LANGUAGES: dict[str, str] = {"fr": "Français", "en": "English"}
DEFAULT_LANGUAGE = "fr"

STRINGS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ FR
    "fr": {
        "app.title": (
            "# 🎞️ SeedVR2 Batch Upscaler\n"
            "Upscaling d'images par lot via **SeedVR2-3B** (officiel, FP8 ou GGUF) — "
            "les **noms de fichiers sont conservés à l'identique**."
        ),
        "lang.label": "Langue / Language",
        # Bandeau GPU -------------------------------------------------------
        "gpu.wait": "⏳ Détection du GPU en cours…",
        "gpu.ok": ("🟢 **GPU prêt** : {name} ({vram} Go) · pilote {driver} · "
                   "torch {torch} (CUDA {cuda}) · bf16 {bf16}"),
        "gpu.bf16_no": "✗ (fp16 utilisé)",
        "gpu.ko": "🔴 **GPU CUDA non détecté — SeedVR2 ne pourra pas démarrer.**",
        "gpu.cause.driver": ("Le pilote NVIDIA ne répond pas (nvidia-smi en échec) — "
                             "souvent après une mise à jour Windows ou un pilote "
                             "réinstallé sans redémarrage."),
        "gpu.cause.torch_missing": ("PyTorch n'est pas installé dans cet "
                                    "environnement (.venv)."),
        "gpu.cause.cpu_build": ("torch est une build CPU-only (sans CUDA) — un "
                                "pip install sans index CUDA a dû remplacer la "
                                "build CUDA."),
        "gpu.cause.cuda_false": ("Le pilote répond et torch embarque CUDA, pourtant "
                                 "torch.cuda.is_available() = False."),
        "gpu.cause.unknown": "Cause indéterminée.",
        "gpu.hint.driver": ("→ Réinstallez/mettez à jour le pilote "
                            "(nvidia.fr/drivers) puis REDÉMARREZ le PC."),
        "gpu.hint.torch_missing": "→ Installez la build CUDA : "
                                  r"`.venv\Scripts\python.exe -m pip install torch torchvision "
                                  "--index-url https://download.pytorch.org/whl/cu130`",
        "gpu.hint.cpu_build": "→ Remplacez-la par la build CUDA : "
                              r"`.venv\Scripts\python.exe -m pip install torch torchvision "
                              "--index-url https://download.pytorch.org/whl/cu130`",
        "gpu.hint.cuda_false": ("→ Dans l'ordre : 1) redémarrez le PC ; "
                                "2) vérifiez la variable CUDA_VISIBLE_DEVICES ; "
                                "3) mettez à jour le pilote NVIDIA."),
        "gpu.hint.unknown": "",
        "gpu.ko.footer": ("👉 Lancez `python check_install.py` pour le détail, ou cliquez "
                          "Démarrer : le journal affichera la réparation exacte."),
        # Dossiers ------------------------------------------------------------
        "input_dir.label": "Dossier d'entrée (images PNG/JPG/WEBP/BMP/TIFF)",
        "input_dir.ph": r"ex. C:\images\input",
        "output_dir.label": "Dossier de sortie",
        "output_dir.ph": r"ex. C:\images\output",
        "open.btn": "📂 Ouvrir",
        "open.none": "📂 Aucun dossier de sortie renseigné.",
        "open.ok": "📂 Ouverture de {folder}",
        "open.fail": "📂 Impossible d'ouvrir {folder} ({err})",
        "drop.label": ("…ou glissez-déposez un dossier d'images ici (copié dans un "
                       "dossier temporaire — choisissez alors un dossier de sortie "
                       "explicite)"),
        "drop.logged": "Dossier déposé : {folder}",
        # Modèles ---------------------------------------------------------------
        "model.label": ("Modèle détecté dans models/ (aucun fichier ? utilisez "
                        "⬇️ Télécharger ci-dessous)"),
        "refresh.btn": "🔄 Actualiser",
        "preset.label": "Télécharger un modèle depuis Hugging Face",
        "download.btn": "⬇️ Télécharger",
        "dl.unknown": "⛔ Preset inconnu.",
        "dl.fail": "Échec du téléchargement : {err}",
        "dl.status": "ℹ️ {msg}",
        # Réglages --------------------------------------------------------------
        "scale.label": "Facteur d'upscale",
        "scale.x2": "x2", "scale.x4": "x4", "scale.x8": "x8",
        "scale.custom": "Personnalisé",
        "scale.custom_label": "Facteur personnalisé",
        "fmt.label": "Format de sortie",
        "fmt.keep": "Comme l'entrée",
        "fmt.png": "PNG", "fmt.jpg": "JPG", "fmt.webp": "WebP",
        "quality.label": "Qualité (JPG/WebP ; compression PNG)",
        "adv.accordion": "Options avancées",
        "suffix.note": ("Par défaut, **le nom de fichier d'entrée est conservé à "
                        "l'identique** (photo001.png → photo001.png). Cochez l'option "
                        "ci-dessous pour ajouter un suffixe."),
        "suffix.chk": "Ajouter un suffixe",
        "suffix.label": "Suffixe",
        "conflict.label": "Si le fichier de sortie existe déjà",
        "conflict.overwrite": "Écraser",
        "conflict.skip": "Ignorer",
        "conflict.rename": "Renommer (auto)",
        "tiling.md": ("**Tuilage** — activé auto si la sortie > 4 Mpx. **Important** : "
                      "la taille est automatiquement bornée pour que chaque tuile de "
                      "*sortie* reste ≈ 1-1,5 Mpx (régime d'entraînement de SeedVR2) — "
                      "au-delà : lenteur et qualité dégradées."),
        "tiling.chk": "Toujours découper en tuiles",
        "tile.size_label": "Taille de tuile max (px d'entrée)",
        "tile.overlap_label": "Recouvrement (px)",
        "engine.md": "**Moteur / précision**",
        "seed.label": "Graine (seed)",
        "precision.label": "Précision",
        "backend.label": "Backend (auto = pipeline officiel ; GPU NVIDIA obligatoire)",
        "low_vram.label": ("Mode faible VRAM (ralentit fortement ; utile < ~10 Go)"),
        "clear_cache.label": "Vider le cache GPU entre les images",
        "parallel_io.label": "Écritures disque en parallèle",
        "resume.label": "Reprendre un lot interrompu",
        "expert.accordion": "🔬 Réglages avancés du modèle (netteté / détail — pipeline officiel)",
        "expert.md": (
            "**Paramètres natifs du pipeline officiel**, normalement figés aux valeurs "
            "du script d'origine. À manier avec prudence : trop pousser le guidage ou "
            "le bruit peut donner un rendu « sur-traité » (halos, aspect plastique) — "
            "testez sur 1-2 images avant un gros lot."
        ),
        "cfg_scale.label": "Guidage (cfg_scale) — 1.0 = officiel, > 1 accentue le détail",
        "cfg_rescale.label": "Rescale du guidage (cfg_rescale) — 0.0 = officiel",
        "cond_noise_scale.label": "Bruit de condition — 0.0 = officiel, un peu plus ajoute de la texture",
        "color_fix.label": "Correction couleur (wavelet color fix officiel)",
        # Boutons / statuts -------------------------------------------------------
        "btn.start": "▶️ Démarrer",
        "btn.pause": "⏸️ Pause",
        "btn.resume": "⏯️ Reprendre",
        "btn.cancel": "⏹️ Annuler",
        "status.idle": "🟢 Prêt",
        "status.loading": "⏳ Chargement du modèle…",
        "status.running": "🔵 Traitement en cours…",
        "status.paused": "🟡 En pause",
        "status.cancelling": "🟠 Annulation en cours…",
        "status.finished": "✅ Terminé",
        "status.failed": "🔴 Échec (voir le journal)",
        "counter.images": "📦 {done}/{total} images",
        "counter.tile": " · tuile {i}/{n}",
        "eta.fmt": "~{d} restantes",
        "eta.pending": "estimation en cours…",
        # Aperçu / journal ----------------------------------------------------------
        "preview.label": ("Aperçu avant / après — VIGNETTE réduite (≤768 px) : jugez la "
                          "qualité et les dimensions sur le fichier réel du dossier de "
                          "sortie"),
        "preview.before": "Avant",
        "preview.after": "Après",
        "journal.label": "Journal de traitement",
        # Démarrage ------------------------------------------------------------------
        "err.already": "⚠️ Un traitement est déjà en cours.",
        "err.no_input": "⛔ Dossier d'entrée introuvable — vérifiez le chemin.",
        "err.no_model": ("⛔ Aucun modèle dans models/ — choisissez un preset puis "
                         "cliquez sur « ⬇️ Télécharger »."),
        "start.started": "⏳ Démarrage…",
        "summary": ("— Bilan : {ok} réussie(s), {fail} échec(s), {skip} ignorée(s), "
                    "moyenne {avg} s/image —"),
    },
    # ------------------------------------------------------------------ EN
    "en": {
        "app.title": (
            "# 🎞️ SeedVR2 Batch Upscaler\n"
            "Batch image upscaling with **SeedVR2-3B** (official, FP8 or GGUF) — "
            "**file names are kept exactly as-is**."
        ),
        "lang.label": "Language / Langue",
        # GPU banner -------------------------------------------------------
        "gpu.wait": "⏳ Detecting GPU…",
        "gpu.ok": ("🟢 **GPU ready**: {name} ({vram} GB) · driver {driver} · "
                   "torch {torch} (CUDA {cuda}) · bf16 {bf16}"),
        "gpu.bf16_no": "✗ (fp16 will be used)",
        "gpu.ko": "🔴 **No CUDA GPU detected — SeedVR2 will not be able to start.**",
        "gpu.cause.driver": ("The NVIDIA driver is not responding (nvidia-smi failed) — "
                             "often after a Windows update or a driver reinstall "
                             "without reboot."),
        "gpu.cause.torch_missing": ("PyTorch is not installed in this "
                                    "environment (.venv)."),
        "gpu.cause.cpu_build": ("torch is a CPU-only build (no CUDA) — a pip install "
                                "without the CUDA index probably replaced the "
                                "CUDA build."),
        "gpu.cause.cuda_false": ("The driver responds and torch bundles CUDA, yet "
                                 "torch.cuda.is_available() = False."),
        "gpu.cause.unknown": "Undetermined cause.",
        "gpu.hint.driver": ("→ Reinstall/update the driver (nvidia.com/drivers) "
                            "then RESTART the PC."),
        "gpu.hint.torch_missing": "→ Install the CUDA build: "
                                  r"`.venv\Scripts\python.exe -m pip install torch torchvision "
                                  "--index-url https://download.pytorch.org/whl/cu130`",
        "gpu.hint.cpu_build": "→ Replace it with the CUDA build: "
                              r"`.venv\Scripts\python.exe -m pip install torch torchvision "
                              "--index-url https://download.pytorch.org/whl/cu130`",
        "gpu.hint.cuda_false": ("→ In order: 1) restart the PC; "
                                "2) check the CUDA_VISIBLE_DEVICES variable; "
                                "3) update the NVIDIA driver."),
        "gpu.hint.unknown": "",
        "gpu.ko.footer": ("👉 Run `python check_install.py` for details, or click Start: "
                          "the log will show the exact repair."),
        # Folders -------------------------------------------------------------
        "input_dir.label": "Input folder (PNG/JPG/WEBP/BMP/TIFF images)",
        "input_dir.ph": r"e.g. C:\images\input",
        "output_dir.label": "Output folder",
        "output_dir.ph": r"e.g. C:\images\output",
        "open.btn": "📂 Open",
        "open.none": "📂 No output folder set.",
        "open.ok": "📂 Opening {folder}",
        "open.fail": "📂 Could not open {folder} ({err})",
        "drop.label": ("…or drag and drop an image folder here (copied to a "
                       "temporary folder — then set an explicit output folder)"),
        "drop.logged": "Dropped folder: {folder}",
        # Models ---------------------------------------------------------------
        "model.label": ("Model detected in models/ (nothing there? use "
                        "⬇️ Download below)"),
        "refresh.btn": "🔄 Refresh",
        "preset.label": "Download a model from Hugging Face",
        "download.btn": "⬇️ Download",
        "dl.unknown": "⛔ Unknown preset.",
        "dl.fail": "Download failed: {err}",
        "dl.status": "ℹ️ {msg}",
        # Settings --------------------------------------------------------------
        "scale.label": "Upscale factor",
        "scale.x2": "x2", "scale.x4": "x4", "scale.x8": "x8",
        "scale.custom": "Custom",
        "scale.custom_label": "Custom factor",
        "fmt.label": "Output format",
        "fmt.keep": "Same as input",
        "fmt.png": "PNG", "fmt.jpg": "JPG", "fmt.webp": "WebP",
        "quality.label": "Quality (JPG/WebP; PNG compression)",
        "adv.accordion": "Advanced options",
        "suffix.note": ("By default, **the input file name is kept exactly as-is** "
                        "(photo001.png → photo001.png). Tick the option below to "
                        "add a suffix."),
        "suffix.chk": "Add a suffix",
        "suffix.label": "Suffix",
        "conflict.label": "If the output file already exists",
        "conflict.overwrite": "Overwrite",
        "conflict.skip": "Skip",
        "conflict.rename": "Auto-rename",
        "tiling.md": ("**Tiling** — auto-enabled when output > 4 Mpx. **Important**: "
                      "sizes are automatically capped so each *output* tile stays "
                      "≈ 1–1.5 Mpx (SeedVR2's training regime) — beyond that: "
                      "slowdowns and degraded quality."),
        "tiling.chk": "Always use tiles",
        "tile.size_label": "Max tile size (input px)",
        "tile.overlap_label": "Overlap (px)",
        "engine.md": "**Engine / precision**",
        "seed.label": "Seed",
        "precision.label": "Precision",
        "backend.label": "Backend (auto = official pipeline; NVIDIA GPU required)",
        "low_vram.label": "Low VRAM mode (much slower; useful below ~10 GB)",
        "clear_cache.label": "Clear GPU cache between images",
        "parallel_io.label": "Parallel disk writes",
        "resume.label": "Resume an interrupted batch",
        "expert.accordion": "🔬 Advanced model settings (sharpness / detail — official pipeline)",
        "expert.md": (
            "**Native parameters of the official pipeline**, normally locked to the "
            "original script's values. Use with care: pushing guidance or noise too "
            "far can look \"over-processed\" (halos, plastic look) — test on 1-2 "
            "images before a large batch."
        ),
        "cfg_scale.label": "Guidance (cfg_scale) — 1.0 = official, > 1 sharpens further",
        "cfg_rescale.label": "Guidance rescale (cfg_rescale) — 0.0 = official",
        "cond_noise_scale.label": "Condition noise — 0.0 = official, a bit more adds texture",
        "color_fix.label": "Color correction (official wavelet color fix)",
        # Buttons / statuses -------------------------------------------------------
        "btn.start": "▶️ Start",
        "btn.pause": "⏸️ Pause",
        "btn.resume": "⏯️ Resume",
        "btn.cancel": "⏹️ Cancel",
        "status.idle": "🟢 Ready",
        "status.loading": "⏳ Loading model…",
        "status.running": "🔵 Processing…",
        "status.paused": "🟡 Paused",
        "status.cancelling": "🟠 Cancelling…",
        "status.finished": "✅ Done",
        "status.failed": "🔴 Failed (see log)",
        "counter.images": "📦 {done}/{total} images",
        "counter.tile": " · tile {i}/{n}",
        "eta.fmt": "~{d} left",
        "eta.pending": "estimating…",
        # Preview / journal ----------------------------------------------------------
        "preview.label": ("Before/after preview — reduced THUMBNAIL (≤768 px): judge "
                          "quality and dimensions on the real file in the output "
                          "folder"),
        "preview.before": "Before",
        "preview.after": "After",
        "journal.label": "Processing log",
        # Start -----------------------------------------------------------------------
        "err.already": "⚠️ A batch is already running.",
        "err.no_input": "⛔ Input folder not found — check the path.",
        "err.no_model": ("⛔ No model in models/ — pick a preset then click "
                         "« ⬇️ Download »."),
        "start.started": "⏳ Starting…",
        "summary": ("— Summary: {ok} succeeded, {fail} failed, {skip} skipped, "
                    "average {avg} s/image —"),
    },
}


def tr(lang: str, key: str, **fmt: object) -> str:
    """Traduit ``key`` dans ``lang`` ; repli FR → clé brute si introuvable.

    Les paramètres ``fmt`` sont interpolés avec ``str.format`` ; toute clé de
    format manquante retourne le texte brut plutôt que de lever une erreur
    (l'UI ne doit jamais planter pour une traduction).
    """
    table = STRINGS.get(lang) or STRINGS[DEFAULT_LANGUAGE]
    text = table.get(key) or STRINGS[DEFAULT_LANGUAGE].get(key) or key
    if not fmt:
        return text
    try:
        return text.format(**fmt)
    except Exception:
        return text
