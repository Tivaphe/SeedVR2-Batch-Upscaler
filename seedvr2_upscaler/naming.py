"""Politique de nommage des fichiers de sortie et résolution de conflits.

Règle n°1 de l'application : **le nom de fichier d'entrée est conservé
à l'identique**. Aucun suffixe n'est ajouté sauf si l'utilisateur a
explicitement coché l'option correspondante (voir ``JobConfig.add_suffix``).

En cas de conflit dans le dossier de sortie, trois comportements :
    - OVERWRITE : on écrase le fichier existant ;
    - SKIP      : on ignore l'image (aucun traitement) ;
    - RENAME    : « photo001.png » devient « photo001 (1).png »,
                  « photo001 (2).png », etc.
"""
from __future__ import annotations

from pathlib import Path

from .models import ConflictPolicy


def apply_suffix(stem: str, suffix: str, enabled: bool) -> str:
    """Ajoute éventuellement le suffixe choisi par l'utilisateur.

    Par défaut ``enabled`` est False : le nom d'origine est donc
    retourné strictement à l'identique.
    """
    if enabled and suffix:
        return f"{stem}{suffix}"
    return stem


def resolve_output_path(
    directory: Path,
    filename: str,
    policy: ConflictPolicy,
    _reserved: set[str] | None = None,
) -> Path | None:
    """Résout le chemin de sortie en fonction de la politique de conflit.

    Args:
        directory: dossier de sortie.
        filename: nom de fichier souhaité (déjà avec suffixe éventuel).
        policy: comportement en cas d'existence du fichier.
        _reserved: ensemble interne des chemins déjà attribués pendant un
            lot (évite les collisions entre renommages dans le même lot).

    Returns:
        Le chemin final, ou ``None`` si l'image doit être ignorée (SKIP).
    """
    candidate = directory / filename
    taken: set[str] = set() if _reserved is None else _reserved

    if candidate.exists() or str(candidate).lower() in taken:
        if policy is ConflictPolicy.SKIP:
            return None
        if policy is ConflictPolicy.RENAME:
            stem, ext = candidate.stem, candidate.suffix
            index = 1
            while True:
                renamed = directory / f"{stem} ({index}){ext}"
                if not renamed.exists() and str(renamed).lower() not in taken:
                    candidate = renamed
                    break
                index += 1
        # OVERWRITE : on conserve simplement le chemin initial.
    if _reserved is not None:
        _reserved.add(str(candidate).lower())
    return candidate
