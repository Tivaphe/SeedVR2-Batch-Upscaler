"""Validation, portable presets and batch preflight (no GPU imports)."""
from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import tempfile
from pathlib import Path

from PIL import Image

from .constants import IMAGE_EXTENSIONS
from .models import JobConfig, ConflictPolicy, format_for_source
from .naming import apply_suffix


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.seedvr2-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def serialize_job(job: JobConfig) -> dict:
    def convert(value):
        if isinstance(value, enum.Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        return value
    return convert(dataclasses.asdict(job))


def scan_inputs(job: JobConfig) -> list[Path]:
    root = job.input_dir.resolve()
    output = job.output_dir.resolve()
    if job.input_files:
        candidates = job.input_files
    else:
        candidates = root.rglob('*') if job.recursive else root.iterdir()
    result = []
    for p in candidates:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            if not p.is_relative_to(root):
                raise ValueError(f'Image hors du dossier source : {p}')
            if output != root and p.is_relative_to(output):
                continue
            result.append(p)
    return sorted(set(result))


def preflight(job: JobConfig) -> dict:
    if not job.input_dir.is_dir():
        raise ValueError("Le dossier d'entrée n'existe pas.")
    if not math.isfinite(job.scale) or not 1 <= job.scale <= 16:
        raise ValueError('Le facteur doit être compris entre 1 et 16.')
    if not 1 <= job.quality <= 100:
        raise ValueError('Qualité attendue : 1 à 100.')
    if job.tiling.tile_size < 64 or not 0 <= job.tiling.overlap < job.tiling.tile_size:
        raise ValueError('Tuilage invalide : taille ≥ 64 et recouvrement < taille.')
    if job.memory_profile not in {'auto', 'eco', 'balanced', 'fast'}:
        raise ValueError('Profil mémoire inconnu.')
    for value, low, high in ((job.cfg_scale, .5, 4), (job.cfg_rescale, 0, 1),
                             (job.cond_noise_scale, 0, 1)):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError('Paramètre de diffusion hors limites.')
    if any(c in job.suffix for c in '/\\:*?"<>|'):
        raise ValueError('Le suffixe ne doit pas contenir de caractères de chemin.')
    if job.output_dir.resolve() == job.input_dir.resolve():
        raise ValueError('Choisissez un dossier de sortie différent pour protéger les originaux.')
    files = scan_inputs(job)
    if not files:
        raise ValueError('Aucune image prise en charge dans la sélection.')
    job.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=job.output_dir):
        pass  # real write permission check
    max_pixels, conflicts, errors = 0, 0, []
    destinations = set()
    for source in files:
        try:
            with Image.open(source) as image:
                w, h = image.size
                if job.crop:
                    x, y, cw, ch = job.crop
                    if min(x, y) < 0 or min(cw, ch) <= 0:
                        raise ValueError('Zone de test invalide.')
                    # Orientation may swap dimensions.
                    if image.getexif().get(274) in (5, 6, 7, 8):
                        w, h = h, w
                    if x + cw > w or y + ch > h:
                        raise ValueError("La zone dépasse l'image.")
                    w, h = cw, ch
                max_pixels = max(max_pixels, math.ceil(w * job.scale / 16) * 16 *
                                 math.ceil(h * job.scale / 16) * 16)
            with Image.open(source) as check:
                check.verify()
        except Exception as exc:
            if job.crop:
                raise ValueError(f'Zone de test invalide pour {source.name} : {exc}') from exc
            errors.append(f'{source.name}: {exc}')
        _, ext = format_for_source(source, job.output_format)
        parent = source.relative_to(job.input_dir.resolve()).parent
        destination = job.output_dir / parent / (apply_suffix(source.stem, job.suffix, job.add_suffix) + ext)
        if destination.resolve() == source or not destination.resolve().is_relative_to(job.output_dir.resolve()):
            raise ValueError(f'Destination dangereuse (lien vers une source ou hors sortie) : {destination}')
        key = str(destination).casefold()
        if key in destinations and job.conflict_policy is ConflictPolicy.OVERWRITE:
            raise ValueError(f'Deux sources écraseraient la même sortie : {destination}')
        conflicts += destination.exists() or key in destinations
        destinations.add(key)
    return {'images': len(files), 'conflicts': conflicts, 'invalid_images': errors,
            'largest_output_mpx': round(max_pixels / 1e6, 2),
            'assembly_float_gib': round(max_pixels * 16 / 1024**3, 2),
            'note': 'Estimation assemblage uniquement, hors modèle, copies et image finale.'}
