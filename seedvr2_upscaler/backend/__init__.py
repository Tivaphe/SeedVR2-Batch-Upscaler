"""Backends d'upscaling disponibles + fabrique.

- ``official`` : pipeline officiel SeedVR2 (GPU NVIDIA **obligatoire**) ;
- ``auto``     : synonyme d'``official`` (il n'existe plus d'alternative CPU —
  un GPU indétecté déclenche un diagnostic explicite, pas un repli silencieux).
"""
from __future__ import annotations

from pathlib import Path

from .base import BackendOptions, BackendUnavailable, ProgressCb, UpscaleBackend


def create_backend(
    name: str,
    options: BackendOptions | None = None,
    models_dir: Path | None = None,
) -> UpscaleBackend:
    """Instancie le backend demandé (\"auto\" | \"official\")."""
    if name in {"auto", "official", ""}:
        from .official import SeedVR2OfficialBackend

        return SeedVR2OfficialBackend(options=options, models_dir=models_dir)
    raise ValueError(
        f"Backend inconnu : {name!r}. Les seuls backends valides sont « auto » "
        "et « official » (le retrait du mode démo est voulu : sans GPU CUDA, "
        "l'application signale le diagnostic au lieu de produire de faux résultats)."
    )


__all__ = [
    "BackendOptions",
    "BackendUnavailable",
    "ProgressCb",
    "UpscaleBackend",
    "create_backend",
]
