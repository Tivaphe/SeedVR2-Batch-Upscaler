"""Détection et classification automatique des modèles présents dans models/.

Reconnaît :
    - le checkpoint officiel ByteDance (``seedvr2_ema_3b.pth``…) ;
    - les safetensors FP16 / BF16 ;
    - les safetensors FP8 e4m3fn ;
    - les GGUF quantifiés (Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0…).

Fournit aussi un téléchargeur de presets depuis Hugging Face et la
recherche des assets auxiliaires requis par le pipeline officiel
(VAE + embeddings texte précalculés du script d'inférence officiel).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .constants import (
    MODEL_EXTENSIONS,
    NEG_EMB_CANDIDATES,
    OFFICIAL_HF_REPO,
    POS_EMB_CANDIDATES,
    VAE_CANDIDATE_PATTERNS,
)
from .models import ModelInfo, ModelKind

# Ex. « ...-Q4_K_M.gguf » -> Q4_K_M ; « ...-Q6_K.gguf » -> Q6_K ; « ...-Q8_0 » -> Q8_0
_QUANT_RE = re.compile(r"(Q[3-8](?:_K(?:_[MSL])?|_0)?)", re.IGNORECASE)


def classify_model(path: Path) -> ModelInfo:
    """Déduit le type de modèle à partir du nom de fichier et de l'extension."""
    lower = path.name.lower()
    size = path.stat().st_size if path.exists() else 0
    quant: str | None = None

    if lower.endswith(".gguf"):
        match = _QUANT_RE.search(path.name)
        quant = match.group(1).upper() if match else None
        kind = ModelKind.GGUF
    elif "fp8" in lower or "e4m3" in lower or "e5m2" in lower:
        kind = ModelKind.FP8
    elif lower.endswith(".safetensors"):
        kind = ModelKind.FP16  # safetensors non annoté : on tente FP16/BF16
    else:
        kind = ModelKind.OFFICIAL  # .pth / .pt : checkpoint torch officiel

    return ModelInfo(path=path, name=path.stem, kind=kind, quant=quant, size_bytes=size)


def scan_models(models_dir: Path) -> list[ModelInfo]:
    """Liste tous les modèles exploitables, triés (GGUF et FP8 en premier car légers)."""
    if not models_dir.exists():
        return []
    found = [
        classify_model(p)
        for p in sorted(models_dir.iterdir())
        if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
    ]
    order = {ModelKind.GGUF: 0, ModelKind.FP8: 1, ModelKind.FP16: 2, ModelKind.OFFICIAL: 3}
    return sorted(found, key=lambda m: (order[m.kind], m.name.lower()))


@dataclass(frozen=True)
class AuxAssets:
    """Assets auxiliaires du pipeline officiel SeedVR2 (voir script officiel)."""

    vae: Path | None
    pos_emb: Path | None
    neg_emb: Path | None

    @property
    def complete(self) -> bool:
        return all(p is not None and p.exists() for p in (self.vae, self.pos_emb, self.neg_emb))


def find_aux_assets(models_dir: Path) -> AuxAssets:
    """Cherche le VAE et les embeddings texte dans le dossier des modèles."""

    def first(patterns: tuple[str, ...]) -> Path | None:
        for pattern in patterns:
            matches = sorted(models_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    return AuxAssets(
        vae=first(VAE_CANDIDATE_PATTERNS),
        pos_emb=first(POS_EMB_CANDIDATES),
        neg_emb=first(NEG_EMB_CANDIDATES),
    )


def download_file(
    repo_id: str,
    filename: str,
    dest_dir: Path,
    log: Callable[[str], None] = print,
) -> Path:
    """Télécharge un fichier depuis Hugging Face dans ``dest_dir``.

    Utilise ``huggingface_hub`` (reprise de téléchargement gérée nativement).
    """
    from huggingface_hub import hf_hub_download  # import paresseux (dépendance lourde)

    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"Téléchargement de {filename} depuis {repo_id}…")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,  # compatibilité Windows
    )
    log(f"Téléchargé : {path}")
    return Path(path)


def ensure_aux_assets(models_dir: Path, log: Callable[[str], None] = print) -> AuxAssets:
    """Télécharge les assets officiels manquants (petits fichiers du repo HF).

    Suit la démarche du README officiel (``snapshot_download`` sur
    ``ByteDance-Seed/SeedVR2-3B``), restreinte aux fichiers nécessaires
    à l'inférence : VAE + embeddings texte.
    """
    from huggingface_hub import hf_hub_download

    assets = find_aux_assets(models_dir)
    needed: dict[str, Path | None] = {
        "ema_vae.pth": assets.vae,
        "pos_emb.pt": assets.pos_emb,
        "neg_emb.pt": assets.neg_emb,
    }
    for filename, current in needed.items():
        if current is None or not current.exists():
            log(f"Asset manquant « {filename} » : récupération (repo officiel HF)…")
            try:
                hf_hub_download(
                    repo_id=OFFICIAL_HF_REPO,
                    filename=filename,
                    local_dir=str(models_dir),
                    local_dir_use_symlinks=False,
                )
            except Exception as exc:  # réseau coupé, repo déplacé, etc.
                raise RuntimeError(
                    f"Impossible de télécharger « {filename} » depuis {OFFICIAL_HF_REPO}. "
                    f"Télécharge-le manuellement et place-le dans {models_dir}. ({exc})"
                ) from exc
    return find_aux_assets(models_dir)
