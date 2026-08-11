"""Point d'entrée de SeedVR2 Batch Upscaler.

    python app.py                → lance l'interface graphique Gradio
    python app.py --cli …        → traitement par lot en ligne de commande
    python app.py --help         → aide complète des options CLI

Compatible Windows 10/11 et Python 3.11+.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from seedvr2_upscaler import APP_NAME, __version__
from seedvr2_upscaler.constants import DEFAULT_MODELS_DIR


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="seedvr2-upscaler",
        description=f"{APP_NAME} v{__version__} — upscaling par lot avec SeedVR2-3B.",
    )
    parser.add_argument("--cli", action="store_true",
                        help="Lance le traitement en ligne de commande (sans interface).")
    parser.add_argument("--input", dest="input_dir", type=str, help="Dossier d'entrée.")
    parser.add_argument("--output", dest="output_dir", type=str, help="Dossier de sortie.")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help="Dossier des modèles (défaut : ./models).")
    parser.add_argument("--model", type=str, default=None,
                        help="Nom du fichier de modèle dans --models-dir.")
    parser.add_argument("--scale", type=float, default=4.0, help="Facteur d'upscale.")
    parser.add_argument("--format", dest="output_format", type=str, default="keep",
                        choices=["keep", "PNG", "JPG", "WebP"],
                        help="Format de sortie (défaut : « keep », comme l'entrée).")
    parser.add_argument("--quality", type=int, default=95, help="Qualité 1-100.")
    parser.add_argument("--conflict", type=str, default="rename",
                        choices=["overwrite", "skip", "rename"], help="Politique de conflit.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Ajoute ce suffixe aux noms de sortie (défaut : nom inchangé).")
    parser.add_argument("--seed", type=int, default=666, help="Graine de base.")
    parser.add_argument("--precision", type=str, default="auto",
                        choices=["auto", "bf16", "fp16"], help="Précision de calcul.")
    parser.add_argument("--tiling", action="store_true",
                        help="Force le traitement par tuiles (faible VRAM).")
    parser.add_argument("--tile-size", type=int, default=512, help="Tuile (px d'entrée).")
    parser.add_argument("--tile-overlap", type=int, default=64, help="Recouvrement (px).")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "official"], help="Backend d'inférence.")
    parser.add_argument("--no-low-vram", action="store_true", help="Désactive le swap CPU/GPU.")
    parser.add_argument("--no-clear-cache", action="store_true",
                        help="N'efface pas le cache GPU entre les images.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore l'état de reprise d'un lot interrompu.")
    parser.add_argument("--share", action="store_true",
                        help="Partage Gradio en réseau (mode interface uniquement).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Segments extensibles de l'allocateur CUDA : nettement moins de
    # fragmentation de VRAM sur les longues sessions (163 tuiles × N images).
    # Posé ici, AVANT le premier import torch, sinon la variable est ignorée.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = _parse_args(argv)
    if args.cli:
        return _run_cli(args)

    from seedvr2_upscaler.gui_gradio import launch

    launch(models_dir=Path(args.models_dir), share=args.share)
    return 0


# --------------------------------------------------------------------------- #
# Mode ligne de commande                                                       #
# --------------------------------------------------------------------------- #
def _run_cli(args: argparse.Namespace) -> int:
    from seedvr2_upscaler.models import (
        ConflictPolicy,
        JobConfig,
        OutputFormat,
        Precision,
        TilingConfig,
    )
    from seedvr2_upscaler.registry import scan_models
    from seedvr2_upscaler.worker import BatchRunner, EventBus, fmt_eta

    if not args.input_dir:
        print("⛔ --input est obligatoire en mode CLI (voir --help).")
        return 2

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    models = {m.path.name: m for m in scan_models(models_dir)}
    if not models:
        print(f"⛔ Aucun modèle détecté dans {models_dir}. "
              "Déposez-y un checkpoint SeedVR2 (.pth/.safetensors/.gguf).")
        return 2
    if args.model and args.model not in models:
        print(f"⛔ Modèle « {args.model} » introuvable. Disponibles :")
        for name in models:
            print(f"   - {name}")
        return 2
    model = models[args.model] if args.model else next(iter(models.values()))
    print(f"Modèle sélectionné : {model.path.name} ({model.kind.value})")

    job = JobConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "upscaled",
        model=model,
        scale=float(args.scale),
        output_format=OutputFormat.from_label(args.output_format),
        quality=int(args.quality),
        conflict_policy=ConflictPolicy(args.conflict),
        add_suffix=args.suffix is not None,
        suffix=args.suffix or "_upscaled",
        seed=int(args.seed),
        precision=Precision(args.precision),
        low_vram=not args.no_low_vram,
        clear_cache=not args.no_clear_cache,
        resume=not args.no_resume,
        tiling=TilingConfig(args.tiling, args.tile_size, args.tile_overlap),
        backend_name=args.backend,
    )

    bus = EventBus()
    runner = BatchRunner(job, bus)

    def console_loop() -> None:
        """Affiche en continu logs, progression et temps restant estimé."""
        last_done, last_total = 0, 0
        while True:
            for event in bus.drain():
                if event.kind == "log":
                    print(event.data["message"], flush=True)
                elif event.kind == "progress":
                    last_done = event.data["done"]
                    last_total = event.data["total"]
                    eta = event.data["eta"]
                    print(f"  → {last_done}/{last_total} ({fmt_eta(eta)})", flush=True)
                elif event.kind == "finished":
                    return
            time.sleep(0.2)

    runner.start()
    try:
        console_loop()
    except KeyboardInterrupt:
        print("\nAnnulation demandée (Ctrl+C)…")
        runner.cancel()
        console_loop()
    runner._thread.join(timeout=10)  # assure la finalisation des fichiers
    print(f"Terminé — voir {job.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
