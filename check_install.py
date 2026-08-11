"""Vérification de l'installation — lancé automatiquement par install.bat.

    python check_install.py

Contrôle : version de Python, dépendances de l'application, CUDA/torch,
présence du dépôt officiel SeedVR2 et contenu du dossier models/.
Sortie 0 si l'interface peut démarrer, 1 sinon (avec conseils ciblés).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OK = "  ✅"
KO = "  ❌"
WARN = "  ⚠️ "


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    print("=" * 56)
    print("Vérification de l'installation")
    print("=" * 56)

    problems: list[str] = []

    # --- Python ----------------------------------------------------------
    py = sys.version_info
    symbol = OK if (3, 11) <= (py.major, py.minor) else KO
    print(f"{symbol} Python {py.major}.{py.minor}.{py.micro}", end="")
    if (py.major, py.minor) >= (3, 14):
        print("  (très récent : certaines wheels CUDA peuvent manquer)")
        problems.append("Python ≥ 3.14 : si torch est absent, recréez le venv en 3.12.")
    elif (py.major, py.minor) < (3, 11):
        print("  (trop ancien : Python 3.11+ requis)")
        problems.append("Python 3.11+ requis.")
    else:
        print()

    # --- Dépendances applicatives ----------------------------------------
    required = ["gradio", "PIL", "numpy", "safetensors", "gguf",
                "huggingface_hub", "einops", "omegaconf"]
    for name in required:
        present = has_module(name)
        print(f"{OK if present else KO} {name}")
        if not present:
            problems.append(f"pip install manquant : {name}")

    if not has_module("piexif"):
        print(f"{WARN} piexif absent (EXIF dans les PNG désactivé — non bloquant)")

    # --- Dépendances du dépôt officiel SeedVR2 ---------------------------
    repo_deps = {
        "rotary_embedding_torch": "rotary-embedding-torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "cv2": "opencv-python",
        "mediapy": "mediapy",
    }
    for module, pypi in repo_deps.items():
        present = has_module(module)
        print(f"{OK if present else WARN} {module}", end="")
        print("" if present else f"  → python -m pip install {pypi} (ou laissez "
                                   f"l'application le réparer automatiquement)")
    if not has_module("apex"):
        print(f"{OK} apex absent → repli LayerNorm/RMSNorm natifs automatique "
              "(normal sous Windows)")
    if not has_module("flash_attn"):
        print(f"{OK} flash_attn absent → repli SDPA automatique (normal sous Windows)")

    # --- torch / CUDA (diagnostic complet : pilote → build → disponibilité) --
    sys.path.insert(0, str(ROOT))
    gpu_ok = False
    try:
        from seedvr2_upscaler.gpu_check import detect_gpu, gpu_report_lines

        status = detect_gpu()
        gpu_ok = status.ok
        for line in gpu_report_lines(status):
            print(f"{OK if gpu_ok else KO} {line}")
        if not gpu_ok:
            for problem in status.problems:
                print(f"     cause probable : {problem}")
            for hint in status.hints:
                print(f"     → {hint}")
            problems.append("GPU CUDA indisponible (voir les pistes ci-dessus).")
    except Exception as exc:
        print(f"{KO} diagnostic GPU impossible ({exc})")
        problems.append("torch/CUDA : diagnostic impossible.")

    # --- Dépôt officiel SeedVR2 ---------------------------------------------
    repo = ROOT / "SeedVR" / "projects" / "video_diffusion_sr" / "infer.py"
    if repo.exists():
        print(f"{OK} Dépôt officiel SeedVR : {repo.parent.parent.parent}")
    else:
        print(f"{WARN} Dépôt officiel SeedVR introuvable (git clone ByteDance-Seed/SeedVR)")

    # --- Modèles --------------------------------------------------------------
    models_dir = ROOT / "models"
    weights = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.safetensors")) \
        + list(models_dir.glob("*.gguf"))
    if weights:
        for w in weights:
            print(f"{OK} Modèle détecté : {w.name} ({w.stat().st_size / 1024**3:.1f} Go)")
    else:
        print(f"{WARN} Aucun poids dans models/ (utilisez le bouton Télécharger de l'UI)")

    print("=" * 56)
    gradio_ok = has_module("gradio")
    if gradio_ok and gpu_ok:
        print("Tout est prêt : python app.py (ou run_app.bat)")
    else:
        print("Corriger d'abord :")
        for p in problems:
            print(" -", p)
        if not gradio_ok:
            print(" - .venv\\Scripts\\python.exe -m pip install -r requirements.txt")
        if not gpu_ok:
            print(" - GPU CUDA : sans lui SeedVR2 ne démarre pas (pistes ci-dessus).")
    return 0 if (gradio_ok and gpu_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
