"""Conservative automatic tile profiles; no CUDA dependency in CPU tests."""
from __future__ import annotations

import dataclasses
import hashlib
import json

from .constants import CONFIG_DIR
from .jobs import atomic_json


def profile_key(backend, job) -> str | None:
    if backend.name != 'official':
        return None
    try:
        import torch
        device = torch.cuda.current_device()
        stat = job.model.path.stat()
        identity = (torch.cuda.get_device_name(device), torch.cuda.get_device_properties(device).total_memory,
                    str(job.model.path.resolve()), stat.st_size, stat.st_mtime_ns,
                    job.scale, job.precision.value, job.low_vram, job.cfg_scale)
        return hashlib.sha256(repr(identity).encode()).hexdigest()
    except (ImportError, RuntimeError, OSError):
        return None


def apply_auto_profile(backend, job, log) -> str | None:
    key = profile_key(backend, job)
    if key is None or job.memory_profile != 'auto':
        return key
    size = job.tiling.tile_size
    try:
        import torch
        free, _ = torch.cuda.mem_get_info()
        size = min(size, 192 if free < 4 * 1024**3 else 256 if free < 8 * 1024**3 else 384)
        cache = json.loads((CONFIG_DIR / 'memory_profiles.json').read_text(encoding='utf-8'))
        size = min(size, int(cache.get(key, size)))
    except (ImportError, RuntimeError, OSError, ValueError, TypeError):
        pass
    size = max(64, size)
    backend.options = dataclasses.replace(backend.options,
        tiling=dataclasses.replace(backend.options.tiling, tile_size=size,
                                   overlap=min(backend.options.tiling.overlap, size // 2)))
    log(f'Profil automatique : tuile maximale {size} px (ajustable après OOM).')
    return key


def remember_profile(key: str | None, size: int) -> None:
    if key is None:
        return
    path = CONFIG_DIR / 'memory_profiles.json'
    try:
        cache = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        cache[key] = size
        atomic_json(path, cache)
    except (OSError, ValueError, TypeError):
        pass
